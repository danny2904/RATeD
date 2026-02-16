
import os
import sys
import torch
import torch.nn as nn
import json
import numpy as np
import argparse
import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

import warnings
from transformers import logging as hf_logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from experiments.english.common.metrics import calculate_span_f1, calculate_token_metrics, calculate_auroc, calculate_bias_metrics
    from experiments.english.common.faithfulness import calculate_faithfulness_metrics
    from experiments.english.common.metrics_utils import load_bias_metadata_from_prepared
    from experiments.english.common.reporting import generate_report_string
except ImportError as e:
    print(f"❌ CRITICAL ERROR: Could not import common modules. Check path: {project_root}")
    raise e

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class HateXplainMultiTaskModel(nn.Module):
    def __init__(self, model_name_or_path, num_labels=3, dropout_rate=0.1, use_fusion=True):
        super(HateXplainMultiTaskModel, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.bert = AutoModel.from_pretrained(model_name_or_path)
        self.dropout = nn.Dropout(dropout_rate)
        self.use_fusion = use_fusion
        hidden_dim = self.config.hidden_size
        self.class_classifier = nn.Linear(hidden_dim * 2 if use_fusion else hidden_dim, num_labels)
        self.token_classifier = nn.Linear(hidden_dim, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output
        token_logits = self.token_classifier(self.dropout(sequence_output))
        if self.use_fusion:
            token_probs = torch.softmax(token_logits, dim=-1)
            toxic_probs = token_probs[:, :, 1].unsqueeze(-1)
            sum_toxic = torch.sum(toxic_probs, dim=1) + 1e-9
            rationale_vector = torch.sum(sequence_output * toxic_probs, dim=1) / sum_toxic
            features = torch.cat((pooled_output, rationale_vector), dim=1)
        else:
            features = pooled_output
        class_logits = self.class_classifier(self.dropout(features))
        return class_logits, token_logits

class HateXplainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=128, split='train'):
        self.data = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_map = {"hatespeech": 0, "normal": 1, "offensive": 2}
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                if item.get('split') == split:
                    self.data.append(item)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['comment']
        label_id = self.label_map.get(item.get('label', 'normal'), 1)
        encoding = self.tokenizer(text, max_length=self.max_len, padding="max_length", truncation=True, return_tensors="pt", return_offsets_mapping=True, return_special_tokens_mask=True)
        input_ids = encoding["input_ids"].squeeze(0)
        mask = encoding["attention_mask"].squeeze(0)
        offsets = encoding["offset_mapping"].squeeze(0)
        special_mask = encoding["special_tokens_mask"].squeeze(0)
        token_labels = torch.zeros(self.max_len, dtype=torch.long)
        unsafe_spans = item.get('unsafe_spans_indices', [])
        offsets_list = offsets.tolist()
        for idx_t, (start, end) in enumerate(offsets_list):
            if start == end: continue
            is_toxic = False
            for s_start, s_end in unsafe_spans:
                 if max(start, s_start) < min(end, s_end):
                     is_toxic = True
                     break
            if is_toxic:
                token_labels[idx_t] = 1
        return {
            "input_ids": input_ids, "attention_mask": mask, "special_mask": special_mask,
            "class_labels": torch.tensor(label_id, dtype=torch.long), "token_labels": token_labels,
            "offsets": offsets, "text": text
        }

def train(args, model, tokenizer, device):
    print(f"🚀 Loading Train Data from {args.data_path}")
    train_dataset = HateXplainDataset(args.data_path, tokenizer, split='train')
    val_dataset = HateXplainDataset(args.data_path, tokenizer, split='val')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    cls_criterion = nn.CrossEntropyLoss()
    tok_criterion = nn.CrossEntropyLoss() 
    best_cls_f1 = 0.0
    no_improve_epochs = 0
    print("🔥 Starting Training...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            cls_labels = batch['class_labels'].to(device)
            tok_labels = batch['token_labels'].to(device)
            optimizer.zero_grad()
            cls_logits, tok_logits = model(input_ids, attention_mask)
            loss_cls = cls_criterion(cls_logits, cls_labels)
            active_loss = attention_mask.view(-1) == 1
            active_logits = tok_logits.view(-1, 2)
            active_labels = torch.where(active_loss, tok_labels.view(-1), torch.tensor(tok_criterion.ignore_index).type_as(tok_labels))
            loss_tok = tok_criterion(active_logits, active_labels)
            loss = loss_cls + args.alpha * loss_tok
            if args.use_consistency and args.lambda_const > 0:
                token_probs = torch.softmax(tok_logits, dim=-1)
                toxic_probs = token_probs[:, :, 1]
                clean_mask = (cls_labels == 1)
                loss_consistency = torch.tensor(0.0).to(device)
                if clean_mask.sum() > 0:
                     clean_mask_seq = clean_mask.unsqueeze(1).expand_as(attention_mask)
                     valid_clean_tokens = (attention_mask == 1) & clean_mask_seq
                     selected_probs = toxic_probs[valid_clean_tokens]
                     if selected_probs.numel() > 0:
                         loss_consistency = torch.mean(selected_probs)
                loss += (args.lambda_const * loss_consistency)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")
        val_metrics = evaluate(model, val_loader, device)
        print(f"Validation F1: {val_metrics['cls_f1']:.4f}")
        if val_metrics['cls_f1'] > best_cls_f1:
            best_cls_f1 = val_metrics['cls_f1']
            print(f"🏆 New Best Model! Saving to {args.output_dir}")
            model.bert.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= args.patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
    print("Training Complete.")

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            cls_logits, _ = model(ids, mask)
            all_preds.extend(torch.argmax(cls_logits, 1).cpu().numpy())
            all_labels.extend(batch["class_labels"].numpy())
    cls_f1 = f1_score(all_labels, all_preds, average='macro')
    return {"cls_f1": cls_f1}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval_only", action="store_true", help="Whether to run eval only.")
    parser.add_argument("--data_path", type=str, default="experiments/english/data/hatexplain_prepared.jsonl")
    parser.add_argument("--base_model", type=str, default="xlm-roberta-base")
    parser.add_argument("--output_dir", type=str, default="experiments/english/baseline/results/XLM-RoBERTa-Base")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for token classification loss")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--use_consistency", type=int, default=1, help="Use Consistency Loss (1/0)")
    parser.add_argument("--lambda_const", type=float, default=0.1, help="Weight for Consistency Loss")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.do_train:
        print(f"Initializing model: {args.base_model}")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        model = HateXplainMultiTaskModel(args.base_model, use_fusion=False)
        model.to(device)
        train(args, model, tokenizer, device)
        print("Loading best model for final evaluation...")
        ckpt_path = os.path.join(args.output_dir, "best_model.pth")
        model.load_state_dict(torch.load(ckpt_path))
        
    elif args.do_eval_only:
        print(f"Evaluation Mode. Loading from {args.output_dir}")
        try:
            if os.path.exists(os.path.join(args.output_dir, "config.json")):
                 tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
                 config = AutoConfig.from_pretrained(args.output_dir)
                 base_name = config._name_or_path
            else:
                 base_name = args.base_model
                 tokenizer = AutoTokenizer.from_pretrained(base_name)
            model = HateXplainMultiTaskModel(base_name, use_fusion=False)
            ckpt_path = os.path.join(args.output_dir, "best_model.pth")
            if os.path.exists(ckpt_path):
                print(f"Weights found at {ckpt_path}")
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
            else:
                 bin_path = os.path.join(args.output_dir, "pytorch_model.bin")
                 if os.path.exists(bin_path):
                      print(f"Weights found at {bin_path}")
                      model.load_state_dict(torch.load(bin_path, map_location=device))
                 else:
                      print("Warning: No specific checkpoint found.")
            model.to(device)
        except Exception as e:
            print(f"Error loading model: {e}")
            return

    # Final Evaluation on Test Set
    print("\nRunning Test Evaluation (Full Metrics)...")
    if 'tokenizer' not in locals():
         tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    test_dataset = HateXplainDataset(args.data_path, tokenizer, split='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    model.eval()
    all_preds, all_labels = [], []
    all_tok_preds, all_tok_labels, all_masks, all_special_masks = [], [], [], []
    all_tok_probs = []
    toxic_probs_list = []
    faith_results = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            special_mask = batch["special_mask"].to(device)
            cls_logits, tok_logits = model(ids, mask)
            cls_probs = torch.softmax(cls_logits, dim=1)
            cls_preds = torch.argmax(cls_probs, dim=1).cpu().numpy()
            cls_labels = batch["class_labels"].numpy()
            probs_np = cls_probs.cpu().numpy()
            for p in probs_np:
                toxic_probs_list.append(float(p[0] + p[2]))
            all_preds.extend(cls_preds)
            all_labels.extend(cls_labels)
            tok_probs = torch.softmax(tok_logits, dim=2)[:, :, 1]
            tok_preds = torch.argmax(tok_logits, dim=2).cpu()
            all_tok_preds.append(tok_preds)
            all_tok_labels.append(batch["token_labels"].cpu())
            all_masks.append(mask.cpu())
            all_special_masks.append(special_mask.cpu())
            all_tok_probs.append(tok_probs.cpu())
            batch_offsets = batch['offsets'].numpy()
            batch_texts = batch['text']
            for i in range(len(batch_texts)):
                t_offsets = batch_offsets[i]
                t_preds = tok_preds[i].numpy()
                item_spans = []
                raw_text = batch_texts[i]
                for j, is_pos in enumerate(t_preds):
                    if is_pos == 1:
                        s, e = t_offsets[j]
                        if s == e: continue
                        if s < len(raw_text) and e <= len(raw_text):
                             span_str = raw_text[s:e]
                             item_spans.append(span_str)
                faith_entry = { "text": batch_texts[i], "label_id": int(cls_preds[i]), "probs": probs_np[i].tolist(), "spans": item_spans }
                faith_results.append(faith_entry)
                
    cls_acc = accuracy_score(all_labels, all_preds)
    cls_p, cls_r, cls_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    from sklearn.metrics import confusion_matrix
    cls_cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    cls_metrics = {"acc": cls_acc, "f1": cls_f1, "precision": cls_p, "recall": cls_r}
    all_tok_preds = torch.cat(all_tok_preds)
    all_tok_labels = torch.cat(all_tok_labels)
    all_masks = torch.cat(all_masks)
    all_special_masks = torch.cat(all_special_masks)
    all_tok_probs = torch.cat(all_tok_probs)
    span_f1 = calculate_span_f1(all_tok_preds, all_tok_labels, all_masks, special_tokens_mask=all_special_masks)
    tok_metrics_res = calculate_token_metrics(all_tok_preds, all_tok_labels, all_masks, all_special_masks, all_tok_probs)
    span_metrics = {
       "token_acc": tok_metrics_res['acc'], "token_p": tok_metrics_res['precision'], "token_r": tok_metrics_res['recall'],
       "token_f1": tok_metrics_res['f1'], "token_f1_pos": tok_metrics_res['f1_pos'], "token_auprc": tok_metrics_res.get('auprc', 0.0), "span_f1": span_f1
    }
    bin_gold = [1 if l != 1 else 0 for l in all_labels]
    auroc = calculate_auroc(bin_gold, toxic_probs_list)
    bias_metrics = {}
    bias_items = load_bias_metadata_from_prepared(args.data_path, len(all_labels))
    if bias_items: bias_metrics = calculate_bias_metrics(bias_items, bin_gold, toxic_probs_list)
    class MockRated:
        def __init__(self, model, tokenizer):
            self.model = model
            self.tokenizer = tokenizer
            self.device = next(model.parameters()).device
    class MockPipeline:
        def __init__(self, model, tokenizer):
            self.rated = MockRated(model, tokenizer)
    pipeline = MockPipeline(model, tokenizer)
    if 'calculate_faithfulness_metrics' in globals():
         try: faithfulness = calculate_faithfulness_metrics(pipeline, test_dataset, faith_results)
         except Exception as e:
             print(f"Faithfulness calculation failed: {e}")
             faithfulness = {'faithfulness_comprehensiveness': 0.0, 'faithfulness_sufficiency': 0.0}
    else: faithfulness = {'faithfulness_comprehensiveness': 0.0, 'faithfulness_sufficiency': 0.0}
    
    log_content = generate_report_string(
        "XLM-RoBERTa-Base (Baseline)", 
        cls_metrics, 
        span_metrics, 
        cls_cm,
        auroc=auroc,
        bias_metrics=bias_metrics,
        faithfulness_metrics=faithfulness
    )
    
    print(log_content)
    
    log_name = os.path.join(args.output_dir, "baseline_metrics.log")
    with open(log_name, "w", encoding="utf-8") as f:
        f.write(log_content)
    print(f"✅ Results saved to {log_name}")

if __name__ == "__main__":
    main()

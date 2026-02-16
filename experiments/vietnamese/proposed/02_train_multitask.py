import os
import sys
import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report
from config_loader import DATA_PATH, MODEL_DIR

from model_multitask import XLMRMultiTask as PhoBERTMultiTask # Keep alias for minimal code changes in train loop

# Logging setup
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import json
import random

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ViHOSMultiTaskDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=256, split='train', use_silver=False):
        # Load merged JSONL
        self.data = []
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        if item.get('split') == split:
                            self.data.append(item)
                    except:
                        pass
        except FileNotFoundError:
            logging.error(f"File not found: {data_path}")
                    
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.use_silver = use_silver
        logging.info(f"Loaded {len(self.data)} samples for split '{split}'")
        
    def __len__(self):
        return len(self.data)
    
    def find_sublist(self, sublist, mainlist):
        len_sub = len(sublist)
        len_main = len(mainlist)
        for i in range(len_main - len_sub + 1):
            if mainlist[i : i + len_sub] == sublist:
                return i
        return -1

    def __getitem__(self, index):
        item = self.data[index]
        text = str(item.get('comment', ''))
        
        # Label: ViHOS VIP uses 'label': 'safe' / 'unsafe'
        raw_label = item.get('label', 'safe')
        label = 1 if raw_label == 'unsafe' else 0
        
        # Rationale selection
        if self.use_silver:
            rationale_str = item.get('llm_rationale') 
            if rationale_str and rationale_str != "Clean":
                spans = [s.strip() for s in rationale_str.split(',')]
            else:
                spans = []
        else:
            # GOLD Spans (List of strings)
            spans = item.get('spans', [])
        
        # Tokenize (cần offset_mapping cho char-level alignment)
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
            return_offsets_mapping=True,
        )
        
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        offset_mapping = encoding['offset_mapping'].squeeze(0)  # (max_len, 2)
        
        # Labels Init
        token_labels = torch.zeros(self.max_len, dtype=torch.long)
        token_labels[attention_mask == 0] = -100
        
        # Alignment: ưu tiên char-level (unsafe_spans_indices) giống XLM-R baseline
        unsafe_indices = item.get('unsafe_spans_indices') or []
        if isinstance(unsafe_indices, str):
            try:
                unsafe_indices = json.loads(unsafe_indices) if unsafe_indices else []
            except Exception:
                unsafe_indices = []
        
        if label == 1 and unsafe_indices:
            for idx in range(self.max_len):
                if attention_mask[idx].item() == 0:
                    continue
                start_char, end_char = offset_mapping[idx, 0].item(), offset_mapping[idx, 1].item()
                if start_char == end_char:
                    continue
                token_label = 0
                for span_start, span_end in unsafe_indices:
                    s, e = int(span_start), int(span_end)
                    if max(start_char, s) < min(end_char, e):
                        token_label = 1 if start_char == s else 2
                        break
                token_labels[idx] = token_label
        elif label == 1 and spans:
            # Fallback: find_sublist (có thể fail với tokenizer khác context)
            input_id_list = input_ids.tolist()
            for span in spans:
                if not span:
                    continue
                span_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(span))
                start_idx = self.find_sublist(span_ids, input_id_list)
                if start_idx != -1:
                    token_labels[start_idx] = 1
                    for i in range(start_idx + 1, min(start_idx + len(span_ids), self.max_len)):
                        token_labels[i] = 2

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long),
            'token_labels': token_labels
        }

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Load separate splits from single file
    train_dataset = ViHOSMultiTaskDataset(args.data_path, tokenizer, max_len=args.max_len, split='train', use_silver=args.use_silver)
    val_dataset = ViHOSMultiTaskDataset(args.data_path, tokenizer, max_len=args.max_len, split='val', use_silver=args.use_silver)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Model
    model = PhoBERTMultiTask.from_pretrained(args.model_name, num_labels=2, use_fusion=args.use_fusion)
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.epochs
    num_warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)
    
    best_loss = float('inf')

    # Initialize lists for history
    train_losses = []
    val_losses = []
    val_accs = []
    val_f1s = []

    logging.info("Starting training...")
    
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0
        
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            token_labels = batch['token_labels'].to(device)
            
            model.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                token_labels=token_labels,
                alpha=args.alpha, # Task balancing weight
                use_consistency=args.use_consistency,
                lambda_const=args.lambda_const
            )
            
            loss = outputs['loss']
            
            if torch.isnan(loss):
                logging.error(f"NaN loss detected at epoch {epoch+1}, step {i}")
                # Optional: break or continue
            
            total_train_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            if i % 100 == 0:
                logging.info(f"Epoch {epoch+1}, Step {i}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
        avg_train_loss = total_train_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1}: Avg Train Loss = {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        total_val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                token_labels = batch['token_labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    token_labels=token_labels,
                    alpha=args.alpha,
                    use_consistency=args.use_consistency,
                    lambda_const=args.lambda_const
                )
                
                loss = outputs['loss']
                total_val_loss += loss.item()
                
                # Check metrics for classification task only roughly
                logits = outputs['cls_logits']
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        cls_acc = accuracy_score(all_labels, all_preds)
        cls_f1_macro = f1_score(all_labels, all_preds, average='macro')
        
        logging.info(f"Epoch {epoch+1}: Val Loss = {avg_val_loss:.4f} | CLS Acc = {cls_acc:.4f} | F1-Macro = {cls_f1_macro:.4f}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), args.output_dir + "/best_multitask_model.pth")
            logging.info("Saved best model.")
            
        # Update History
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accs.append(cls_acc)
        val_f1s.append(cls_f1_macro)
    
    # Save Training History
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'val_acc': val_accs,
        'val_f1': val_f1s
    }
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f)
    print(f"Training history saved to {os.path.join(args.output_dir, 'training_history.json')}")

    # --- Global Logging for Ablation ---
    # Construct a unique ID for this run
    run_id = f"vihos_fusion{int(args.use_fusion)}_const{int(args.use_consistency)}_seed{args.seed}"
    if args.alpha == 0: run_id += "_single_task"
    
    config_log = vars(args)
    # Get best metrics (approximate from history or just use last/best)
    best_f1_val = max(val_f1s) if val_f1s else 0.0
    best_acc_val = max(val_accs) if val_accs else 0.0
    
    metrics_log = {
        "val_f1_best": best_f1_val,
        "val_acc_best": best_acc_val,
        "train_loss_final": train_losses[-1] if train_losses else 0.0,
        "epochs_trained": args.epochs
    }
    
    try:
        # Add project root to path to find utils
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
        if project_root not in sys.path:
            sys.path.append(project_root)
            
        # from utils.experiment_logger import ExperimentLogger
        logger = ExperimentLogger()
        logger.log_experiment(
            run_id=run_id, 
            config=config_log, 
            metrics=metrics_log,
            task_name="Vietnamese Info Extraction (ViHOS)",
            tags=["vietnamese", "multitask", "training"]
        )
    except Exception as e:
        logging.error(f"Failed to log to global experiment logger: {e}")

    logging.info("Training completed.")

from config_loader import DATA_PATH, MODEL_DIR

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=DATA_PATH, help="Path to merged JSONL data")
    parser.add_argument("--model_name", type=str, default="xlm-roberta-base")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=2e-5)  # Optimized: giảm từ 5e-5 xuống 2e-5 để stable hơn
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--output_dir", type=str, default=MODEL_DIR)
    parser.add_argument("--use_silver", action="store_true", help="Use silver labels for training (Default: False)")
    
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=5.0, help="Weight for token classification loss")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Ablation Args
    parser.add_argument("--use_fusion", type=int, default=1, help="Use Explanation-Guided Fusion (1/0)")
    parser.add_argument("--use_consistency", type=int, default=1, help="Use Consistency Loss (1/0)")
    parser.add_argument("--lambda_const", type=float, default=0.1, help="Weight for Consistency Loss")
    
    args = parser.parse_args()
    
    # Convert bools
    args.use_fusion = bool(args.use_fusion)
    args.use_consistency = bool(args.use_consistency)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    seed_everything(args.seed)
    train(args)

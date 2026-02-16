import os
import sys
import argparse
import json
import random
import logging
import subprocess
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

# Ẩn cảnh báo DataParallel (gather scalar loss) và PyTorch không build NCCL — train vẫn chạy bình thường
warnings.filterwarnings("ignore", message=".*gather along dimension 0.*scalars.*")
warnings.filterwarnings("ignore", message=".*not compiled with NCCL support.*")


def print_gpu_info():
    """Chạy nvidia-smi để xem GPU (2 dòng lệnh: nvidia-smi rồi in số GPU PyTorch thấy)."""
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            logging.info("nvidia-smi:\n%s", out.stdout.strip())
    except Exception as e:
        logging.warning("nvidia-smi không chạy được: %s", e)
    n = torch.cuda.device_count()
    logging.info("PyTorch thấy %d GPU.", n)
    for i in range(n):
        logging.info("  GPU %d: %s", i, torch.cuda.get_device_name(i))

# --- Import Standardized Model ---
from model_multitask_en import HateXplainMultiTaskBIO

# Khớp 03_evaluate_en: model HateXplainMultiTaskBIO, num_labels=3, use_fusion, tokenizer cùng model_name (roberta-base), max_len=128
# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class HateXplainMultiTaskDataset(Dataset):
    """
    Standardized Dataset for English HateXplain.
    - gold_style='spans': token labels từ spans (string) + find_sublist (BIO). Eval với gold này cần cùng cách.
    - gold_style='unsafe_spans_indices': token labels từ unsafe_spans_indices + overlap (như baseline).
      Train với cách này thì eval (03_evaluate_en) dùng unsafe_spans_indices sẽ ra Span IoU cao như report.
    """
    def __init__(self, data_path, tokenizer, max_len=128, split='train', gold_style='unsafe_spans_indices'):
        self.data = []
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    if item.get('split') == split:
                        self.data.append(item)
        except Exception as e:
            logging.error(f"Error loading data: {e}")
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_map = {"hatespeech": 0, "normal": 1, "offensive": 2}
        self.gold_style = gold_style  # 'unsafe_spans_indices' | 'spans'
        logging.info(f"Loaded {len(self.data)} samples for split '{split}' (gold_style={gold_style})")

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
        label_id = self.label_map.get(item.get('label', 'normal'), 1)

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
        offsets = encoding['offset_mapping'].squeeze(0)

        token_labels = torch.zeros(self.max_len, dtype=torch.long)
        token_labels[attention_mask == 0] = -100

        if self.gold_style == 'unsafe_spans_indices':
            # Giống run_RATeD_E1_baseline: gold từ unsafe_spans_indices + overlap → eval 03 ra IoU cao
            unsafe_spans = item.get('unsafe_spans_indices', [])
            offsets_list = offsets.tolist()
            binary = [0] * self.max_len
            for idx_t in range(self.max_len):
                start, end = offsets_list[idx_t][0], offsets_list[idx_t][1]
                if start == end:
                    continue
                for span in unsafe_spans:
                    s_start, s_end = int(span[0]), int(span[1])
                    if max(start, s_start) < min(end, s_end):
                        binary[idx_t] = 1
                        break
            # Chuyển binary sang BIO: đầu mỗi run = 1 (B), còn lại = 2 (I)
            for idx_t in range(self.max_len):
                if binary[idx_t] == 1:
                    token_labels[idx_t] = 2 if (idx_t > 0 and binary[idx_t - 1] == 1) else 1
        else:
            # spans (string) + find_sublist
            spans = item.get('spans', [])
            if label_id != 1 and spans:
                input_id_list = input_ids.tolist()
                for span in spans:
                    if not span:
                        continue
                    span_tokens = self.tokenizer.tokenize(span)
                    span_ids = self.tokenizer.convert_tokens_to_ids(span_tokens)
                    start_idx = self.find_sublist(span_ids, input_id_list)
                    if start_idx != -1:
                        token_labels[start_idx] = 1
                        for i in range(start_idx + 1, start_idx + len(span_ids)):
                            if i < self.max_len:
                                token_labels[i] = 2

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label_id, dtype=torch.long),
            'token_labels': token_labels
        }

def train(args):
    print_gpu_info()
    n_gpu_available = torch.cuda.device_count()
    n_gpu = getattr(args, 'n_gpu', None)
    if n_gpu is None:
        # Trên Windows, DataParallel (2+ GPU) thường treo do không có NCCL; mặc định 1 GPU để chạy qua đêm ổn định
        if sys.platform == 'win32' and n_gpu_available > 0:
            n_gpu = 1
            logging.info("Windows: dùng 1 GPU để tránh treo (DataParallel). Muốn 2 GPU thì truyền --n_gpu 2.")
        else:
            n_gpu = n_gpu_available if n_gpu_available else 0
    else:
        n_gpu = min(n_gpu, n_gpu_available)
    device = torch.device('cuda' if n_gpu > 0 else 'cpu')
    logging.info("Using device: %s (n_gpu=%d)", device, n_gpu)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    gold_style = getattr(args, 'gold_style', 'unsafe_spans_indices')

    # Khớp 03_evaluate_en: cùng data_path, max_len (128), label_map, gold_style=unsafe_spans_indices
    train_dataset = HateXplainMultiTaskDataset(args.data_path, tokenizer, max_len=args.max_len, split='train', gold_style=gold_style)
    val_dataset = HateXplainMultiTaskDataset(args.data_path, tokenizer, max_len=args.max_len, split='val', gold_style=gold_style)

    # num_workers=0 tránh treo trên Windows (multiprocessing spawn)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0)
    
    # Model Setup
    model = HateXplainMultiTaskBIO.from_pretrained(args.model_name, num_labels=3, use_fusion=args.use_fusion)
    model.to(device)
    if n_gpu > 1:
        model = nn.DataParallel(model, device_ids=list(range(n_gpu)))
        logging.info("DataParallel trên %d GPU: %s", n_gpu, list(range(n_gpu)))
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
    
    best_f1 = 0.0
    
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
                alpha=args.alpha,
                use_consistency=args.use_consistency,
                lambda_const=args.lambda_const
            )
            
            loss = outputs['loss']
            loss_scalar = loss.mean() if loss.dim() > 0 else loss
            total_train_loss += loss_scalar.item()
            loss_scalar.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        model.eval()
        all_preds, all_labels = [], []
        total_val_loss = 0
        
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
                
                l = outputs['loss']
                total_val_loss += (l.mean() if l.dim() > 0 else l).item()
                preds = torch.argmax(outputs['cls_logits'], dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_f1_macro = f1_score(all_labels, all_preds, average='macro')
        val_acc = accuracy_score(all_labels, all_preds)
        
        logging.info(f"Epoch {epoch+1}: Val Loss = {avg_val_loss:.4f} | F1-Macro = {val_f1_macro:.4f} | Acc = {val_acc:.4f}")
        
        if val_f1_macro > best_f1:
            best_f1 = val_f1_macro
            state_to_save = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save(state_to_save, os.path.join(args.output_dir, "best_model.pth"))
            logging.info("🔥 Best Model Saved (F1: %.4f)", best_f1)

    logging.info("English Standardization Training Completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="experiments/english/data/hatexplain_prepared.jsonl")
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--output_dir", type=str, default="experiments/english/output_multitask_standard")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--use_fusion", type=int, default=1)
    parser.add_argument("--use_consistency", type=int, default=1)
    parser.add_argument("--lambda_const", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gold_style", type=str, default="unsafe_spans_indices", choices=["unsafe_spans_indices", "spans"],
                        help="Gold token labels: unsafe_spans_indices (eval 03 ra Span IoU cao) | spans (find_sublist)")
    parser.add_argument("--n_gpu", type=int, default=None,
                        help="Số GPU dùng (DataParallel). Mặc định: tất cả GPU nvidia-smi thấy. Ví dụ: 2")
    args = parser.parse_args()
    args.use_fusion = bool(args.use_fusion)
    args.use_consistency = bool(args.use_consistency)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    seed_everything(args.seed)
    train(args)

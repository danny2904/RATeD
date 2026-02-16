import os
import sys
import argparse
import json
import random
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from config_loader import DATA_PATH, MODEL_DIR
from model_multitask import XLMRMultiTask as PhoBERTMultiTask

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ViHOSMultiTaskDataset(Dataset):
    """Dataset giống 02_train_multitask, giữ alignment unsafe_spans_indices."""

    def __init__(self, data_path, tokenizer, max_len=256, split="train"):
        self.data = []
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        if item.get("split") == split:
                            self.data.append(item)
                    except Exception:
                        pass
        except FileNotFoundError:
            logging.error(f"File not found: {data_path}")

        self.tokenizer = tokenizer
        self.max_len = max_len
        logging.info(f"Loaded {len(self.data)} samples for split '{split}'")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = str(item.get("comment", ""))

        raw_label = item.get("label", "safe")
        label = 1 if raw_label == "unsafe" else 0

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        input_ids = encoding["input_ids"].flatten()
        attention_mask = encoding["attention_mask"].flatten()
        offset_mapping = encoding["offset_mapping"].squeeze(0)  # (max_len, 2)

        token_labels = torch.zeros(self.max_len, dtype=torch.long)
        token_labels[attention_mask == 0] = -100

        unsafe_indices = item.get("unsafe_spans_indices") or []
        if isinstance(unsafe_indices, str):
            try:
                unsafe_indices = json.loads(unsafe_indices) if unsafe_indices else []
            except Exception:
                unsafe_indices = []

        if label == 1 and unsafe_indices:
            for i in range(self.max_len):
                if attention_mask[i].item() == 0:
                    continue
                start_char, end_char = offset_mapping[i, 0].item(), offset_mapping[i, 1].item()
                if start_char == end_char:
                    continue
                tok_lab = 0
                for span_start, span_end in unsafe_indices:
                    s, e = int(span_start), int(span_end)
                    if max(start_char, s) < min(end_char, e):
                        tok_lab = 1 if start_char == s else 2
                        break
                token_labels[i] = tok_lab

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.long),
            "token_labels": token_labels,
        }


class FocalLoss(nn.Module):
    """Focal Loss cho CLS để nhấn mạnh mẫu khó (nhẹ)."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        focal = self.alpha * (1 - pt) ** self.gamma * ce
        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


def train_advanced(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset = ViHOSMultiTaskDataset(args.data_path, tokenizer, max_len=args.max_len, split="train")
    val_dataset = ViHOSMultiTaskDataset(args.data_path, tokenizer, max_len=args.max_len, split="val")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    model = PhoBERTMultiTask.from_pretrained(args.model_name, num_labels=2, use_fusion=True)
    model.to(device)

    # Losses
    if args.use_focal:
        cls_criterion = FocalLoss(gamma=2.0, alpha=0.25)
    else:
        cls_criterion = nn.CrossEntropyLoss()
    token_criterion = nn.CrossEntropyLoss(ignore_index=-100)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    best_val_f1 = 0.0
    best_epoch = -1
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            token_labels = batch["token_labels"].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                token_labels=token_labels,
                alpha=args.alpha,
                use_consistency=False,
                lambda_const=0.0,
            )

            cls_loss = cls_criterion(outputs["cls_logits"], labels)
            tok_loss = token_criterion(outputs["token_logits"].view(-1, outputs["token_logits"].size(-1)),
                                       token_labels.view(-1))
            loss = cls_loss + args.alpha * tok_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())
            pbar.set_postfix({"loss": np.mean(train_losses[-50:])})

        avg_train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        logging.info(f"Epoch {epoch}: Avg Train Loss = {avg_train_loss:.4f}")

        # Validation
        model.eval()
        all_labels, all_preds = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs["cls_logits"]
                preds = torch.argmax(logits, dim=1)

                all_labels.extend(labels.cpu().numpy().tolist())
                all_preds.extend(preds.cpu().numpy().tolist())

        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average="macro")
        logging.info(f"Epoch {epoch}: Val Acc = {val_acc:.4f} | Val F1-Macro = {val_f1:.4f}")

        # Save epoch checkpoint
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch}")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(ckpt_dir, "model.pth"))

        # Early stopping on F1
        if val_f1 > best_val_f1 + args.min_delta:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            best_path = os.path.join(args.output_dir, "best_multitask_model.pth")
            torch.save(model.state_dict(), best_path)
            logging.info(f"New best model at epoch {epoch} (F1={val_f1:.4f}) saved to {best_path}")
        else:
            patience_counter += 1
            logging.info(f"No F1 improvement. Patience {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                logging.info(f"Early stopping triggered at epoch {epoch}. Best epoch = {best_epoch}")
                break

    logging.info(f"Training finished. Best Val F1 = {best_val_f1:.4f} at epoch {best_epoch}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=DATA_PATH)
    parser.add_argument("--model_name", type=str, default="xlm-roberta-base")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(MODEL_DIR, "vihos_e1_advanced"))
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=5.0,
                        help="Weight for token classification loss in total loss.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping patience (epochs without F1 improvement).")
    parser.add_argument("--min_delta", type=float, default=1e-4,
                        help="Minimum F1 improvement to reset patience.")
    parser.add_argument("--use_focal", action="store_true",
                        help="Use Focal Loss for classification head.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_advanced(args)


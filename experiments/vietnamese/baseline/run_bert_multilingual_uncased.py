import json
import logging
import re
import numpy as np
import os
import random
import torch
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Dict, Any
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, f1_score, precision_score, recall_score
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, EarlyStoppingCallback, TrainingArguments, DataCollatorForTokenClassification
from datasets import Dataset

import argparse
import sys

# Add path to common utilities
current_dir = os.path.dirname(os.path.abspath(__file__))
common_dir = os.path.join(current_dir, "..", "common")
if common_dir not in sys.path:
    sys.path.append(common_dir)

from metrics import (
    calculate_char_metrics,
    calculate_syllable_metrics,
    calculate_token_metrics_full,
    vihatet5_standardized_span_metrics,
    get_char_indices_from_spans,
    get_char_indices_from_bio_tags,
)
from reporting import generate_report_string

# Constants
MODEL_NAME = "bert-base-multilingual-uncased"
# Adjust filenames/folder based on model_name
SAFE_MODEL_NAME = MODEL_NAME.replace("/", "_")
DATA_PATH = "c:/Projects/RATeD-V/experiments/vietnamese/data/vihos_prepared.jsonl" # Absolute path
OUTPUT_DIR = f"c:/Projects/RATeD-V/experiments/vietnamese/baseline/results/{SAFE_MODEL_NAME}"
SEED = 42

# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on test set.")
args = parser.parse_args()


# Setup Logging
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(OUTPUT_DIR, "baseline_metrics.log"), mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(SEED)

def align_labels_with_tokens(tokenizer, examples):
    tokenized_inputs = tokenizer(
        examples["comment"],
        truncation=True,
        padding='max_length', 
        max_length=256,
        is_split_into_words=False,
        return_offsets_mapping=True,
    )

    labels = []
    
    for i, offsets in enumerate(tokenized_inputs["offset_mapping"]):
        unsafe_indices = examples["unsafe_spans_indices"][i] 
        doc_labels = []
        
        for idx, (start_char, end_char) in enumerate(offsets):
            if tokenized_inputs["attention_mask"][i][idx] == 0:
                doc_labels.append(-100)
                continue
            
            if start_char == end_char:
                doc_labels.append(-100)
                continue
            
            token_label = 0 # O
            for span_start, span_end in unsafe_indices:
                if max(start_char, span_start) < min(end_char, span_end):
                    if start_char == span_start:
                         token_label = 1 # B-TOXIC
                    elif start_char > span_start:
                         token_label = 2 # I-TOXIC
                    else: 
                        if start_char < span_start and end_char > span_start:
                            token_label = 1 
                        else:
                            token_label = 2
                    break 
            
            doc_labels.append(token_label)
        
        if len(doc_labels) < 256:
             doc_labels.extend([-100] * (256 - len(doc_labels)))
             
        labels.append(doc_labels)

    return {
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "labels": labels,
    }

# --- METRIC CALCULATION (Now imported from common.metrics) ---

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    true_predictions = [
        [p for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [l for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    flat_preds = [item for sublist in true_predictions for item in sublist]
    flat_labels = [item for sublist in true_labels for item in sublist]
    
    precision, recall, f1, _ = precision_recall_fscore_support(flat_labels, flat_preds, average='macro', zero_division=0)
    
    return {
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": f1,
    }

def main():
    logger.info(f"Running Baseline Experiment: {MODEL_NAME}")
    logger.info("Loading data...")
    raw_data = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            raw_data.append(json.loads(line))
            
    # Split
    train_data = [x for x in raw_data if x.get('split') == 'train']
    val_data = [x for x in raw_data if x.get('split') in ['dev', 'val', 'validation']]
    test_data = [x for x in raw_data if x.get('split') == 'test']
    
    if not val_data and not test_data:
        logger.warning("No dev/test split found. Performing random split 80/10/10.")
        random.shuffle(train_data)
        n = len(train_data)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)
        test_data = train_data[val_end:]
        val_data = train_data[train_end:val_end]
        train_data = train_data[:train_end]

    ds_train = Dataset.from_list(train_data)
    ds_val = Dataset.from_list(val_data)
    ds_test = Dataset.from_list(test_data)
    
    logger.info(f"Train: {len(ds_train)}, Val: {len(ds_val)}, Test: {len(ds_test)}")

    # 2. Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    
    model_path = MODEL_NAME
    if args.do_predict and not args.do_train:
        # Check for checkpoints
        checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
        if checkpoints:
            # Sort by number (checkpoint-X) descending
            checkpoints.sort(key=lambda x: int(x.split("-")[1]), reverse=True)
            
            found_valid = False
            for ckpt in checkpoints:
                ckpt_path = os.path.join(OUTPUT_DIR, ckpt)
                # Check integrity (look for weight file)
                if os.path.exists(os.path.join(ckpt_path, "pytorch_model.bin")) or os.path.exists(os.path.join(ckpt_path, "model.safetensors")):
                    logger.info(f"Loading model from valid checkpoint: {ckpt_path}")
                    model_path = ckpt_path
                    found_valid = True
                    break
                else:
                    logger.warning(f"Checkpoint {ckpt} found but missing weight files. Skipping.")
            
            if not found_valid:
                 logger.warning("No valid checkpoint found (with weights). Using base model.")
        else:
            logger.warning("No checkpoint found in output directory. Using base model (metrics might be low).")

    model = AutoModelForTokenClassification.from_pretrained(
        model_path, 
        num_labels=3,
        id2label={0: "O", 1: "B-TOXIC", 2: "I-TOXIC"},
        label2id={"O": 0, "B-TOXIC": 1, "I-TOXIC": 2}
    )

    # 3. Preprocess
    remove_cols = ds_train.column_names
    tokenized_train = ds_train.map(lambda x: align_labels_with_tokens(tokenizer, x), batched=True, remove_columns=remove_cols)
    tokenized_val = ds_val.map(lambda x: align_labels_with_tokens(tokenizer, x), batched=True, remove_columns=remove_cols)
    tokenized_test = ds_test.map(lambda x: align_labels_with_tokens(tokenizer, x), batched=True, remove_columns=remove_cols)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # 4. Training
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=10, 
        weight_decay=0.01,
        save_strategy="epoch", load_best_model_at_end=True, save_total_limit=1, metric_for_best_model="loss", 
        logging_steps=100,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics, callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    if args.do_train:
        logger.info("Starting Training...")
        trainer.train()


    # 5. Final Evaluation & Prediction on Test
    if args.do_predict:
        logger.info("Predicting on Test Set...")
        predictions, labels, _ = trainer.predict(tokenized_test)
    preds = np.argmax(predictions, axis=2)

    logger.info("Calculating Advanced Metrics...")
    
    all_syl_acc = []
    all_syl_f1_macro = []
    all_syl_f1_bin = []
    all_token_acc = []
    all_token_mf1 = []
    all_token_wf1 = []
    ious = []
    all_cf1_bin = []
    all_cf1_macro = []
    all_vihatet5_acc = []
    all_vihatet5_wf1 = []
    all_vihatet5_mf1 = []

    y_true_cls = []
    y_pred_cls = []

    for i, item in enumerate(tqdm(test_data, desc="Calculating Metrics")):
        text = item['comment']
        gold_spans = item['spans']
        input_ids = tokenized_test[i]['input_ids']
        encoding = tokenizer(text, return_offsets_mapping=True, is_split_into_words=False, max_length=256, truncation=True)
        offsets = encoding['offset_mapping']

        pred_tags = preds[i]
        pred_span_strings = []
        current_span_start = -1
        current_span_end = -1

        valid_indices = [idx for idx, t_id in enumerate(input_ids) if t_id not in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]]

        for idx in valid_indices:
            if idx >= len(offsets): break
            start_char, end_char = offsets[idx]
            if start_char == end_char: continue

            tag = pred_tags[idx]
            if tag == 1:
                if current_span_start != -1:
                     pred_span_strings.append(text[current_span_start:current_span_end])
                current_span_start = start_char
                current_span_end = end_char
            elif tag == 2:
                 if current_span_start != -1:
                     current_span_end = max(current_span_end, end_char)
                 else:
                     current_span_start = start_char
                     current_span_end = end_char
            else:
                 if current_span_start != -1:
                     pred_span_strings.append(text[current_span_start:current_span_end])
                     current_span_start = -1
                     current_span_end = -1
        if current_span_start != -1:
            pred_span_strings.append(text[current_span_start:current_span_end])

        gold_indices = get_char_indices_from_spans(item.get('unsafe_spans_indices', []))
        valid_preds = [pred_tags[idx] for idx in valid_indices]
        valid_offsets = [offsets[idx] for idx in valid_indices if idx < len(offsets)]
        if len(valid_preds) > len(valid_offsets):
            valid_offsets.extend([(0, 0)] * (len(valid_preds) - len(valid_offsets)))
        elif len(valid_offsets) > len(valid_preds):
            valid_offsets = valid_offsets[:len(valid_preds)]
        pred_char_indices = get_char_indices_from_bio_tags(valid_preds, valid_offsets)
        c_acc, c_mf1, c_wf1, _ = vihatet5_standardized_span_metrics(pred_char_indices, gold_indices, text)
        all_vihatet5_acc.append(c_acc)
        all_vihatet5_wf1.append(c_wf1)
        all_vihatet5_mf1.append(c_mf1)

        iou, cf1_bin, cf1_macro = calculate_char_metrics(pred_span_strings, gold_spans, text)
        ious.append(iou)
        all_cf1_bin.append(cf1_bin)
        all_cf1_macro.append(cf1_macro)

        s_acc, s_f1_bin, s_f1_macro = calculate_syllable_metrics(pred_span_strings, gold_spans, text)
        all_syl_acc.append(s_acc)
        all_syl_f1_bin.append(s_f1_bin)
        all_syl_f1_macro.append(s_f1_macro)

        tm = calculate_token_metrics_full(pred_span_strings, gold_spans, text, tokenizer)
        if tm:
            all_token_acc.append(tm['Token Accuracy'])
            all_token_mf1.append(tm['Token mF1'])
            all_token_wf1.append(tm['Token wF1'])

        y_true_cls.append(1 if item['label'] == 'unsafe' else 0)
        y_pred_cls.append(1 if len(pred_span_strings) > 0 else 0)

    # --- COMPUTE STANDARDIZED METRICS ---
    cls_metrics = {
        "Accuracy": accuracy_score(y_true_cls, y_pred_cls),
        "F1-Macro": f1_score(y_true_cls, y_pred_cls, average='macro', zero_division=0),
        "Precision": precision_score(y_true_cls, y_pred_cls, average='macro', zero_division=0),
        "Recall": recall_score(y_true_cls, y_pred_cls, average='macro', zero_division=0)
    }

    span_metrics = {
        "Acc": np.mean(all_vihatet5_acc) if all_vihatet5_acc else 0.0,
        "WF1": np.mean(all_vihatet5_wf1) if all_vihatet5_wf1 else 0.0,
        "MF1": np.mean(all_vihatet5_mf1) if all_vihatet5_mf1 else 0.0,
        "Token Accuracy": np.mean(all_token_acc) if all_token_acc else 0.0,
        "Token mF1": np.mean(all_token_mf1) if all_token_mf1 else 0.0,
        "Token wF1": np.mean(all_token_wf1) if all_token_wf1 else 0.0,
        "Syllable Accuracy": np.mean(all_syl_acc) if all_syl_acc else 0.0,
        "Syllable F1 (Macro)": np.mean(all_syl_f1_macro) if all_syl_f1_macro else 0.0,
        "Syllable F1 (Binary)": np.mean(all_syl_f1_bin) if all_syl_f1_bin else 0.0,
        "Span IoU": np.mean(ious),
        "Char F1 (Macro)": np.mean(all_cf1_macro),
        "Char F1 (Binary)": np.mean(all_cf1_bin)
    }

    cm = confusion_matrix(y_true_cls, y_pred_cls)
    
    # --- GENERATE REPORT ---
    report_str = generate_report_string(MODEL_NAME, cls_metrics, span_metrics, cm)
    
    print(report_str)
    logger.info(report_str)



if __name__ == "__main__":
    main()


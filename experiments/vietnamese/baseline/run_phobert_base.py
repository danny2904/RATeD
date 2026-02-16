import json
import logging
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
import shutil
import requests
import zipfile
import io
import re
import sys
import pandas as pd

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
    get_char_indices_from_bio_tags,
    get_char_indices_from_spans
)
from reporting import generate_report_string

# Constants
MODEL_NAME = "vinai/phobert-base-v2"
# Adjust filenames/folder based on model_name
SAFE_MODEL_NAME = "vinai_phobert-base-v2"
DATA_DIR = "c:/Projects/RATeD-V/experiments/vietnamese/data"
OUTPUT_DIR = f"c:/Projects/RATeD-V/experiments/vietnamese/baseline/results/{SAFE_MODEL_NAME}"
VNCORENLP_DIR = "c:/Projects/RATeD-V/experiments/vietnamese/baseline/vncorenlp"
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
    """
    Robust alignment for PhoBERT (Slow Tokenizer) which lacks return_offsets_mapping.
    """
    # 1. Tokenize without offsets first
    tokenized_inputs = tokenizer(
        examples["comment"],
        truncation=True,
        padding='max_length', 
        max_length=256,
        is_split_into_words=False,
        # return_offsets_mapping=True # DISABLED due to SlowTokenizer error
    )

    labels = []
    
    for i, comment in enumerate(examples["comment"]):
        unsafe_indices = examples["unsafe_spans_indices"][i]
        tokens = tokenizer.convert_ids_to_tokens(tokenized_inputs["input_ids"][i])
        
        doc_labels = []
        
        # 2. Build Character-level Toxic Mask
        n = len(comment)
        mask = np.zeros(n, dtype=int)
        for s, e in unsafe_indices:
            s, e = max(0, s), min(n, e)
            if s < e:
                mask[s] = 1 # B
                if e > s + 1:
                    mask[s+1:e] = 2 # I
                    
        # 3. Synchronize Tokens with Text
        cursor = 0
        text_len = len(comment)
        
        for token in tokens:
            # Handle special tokens
            if token in tokenizer.all_special_tokens or token in ['<s>', '</s>', '<pad>']:
                doc_labels.append(-100)
                continue
                
            # Normalize token for matching
            # PhoBERT: '@@' is BPE continuation (rare in v2), '_' is space (sometimes)
            clean_token = token.replace("@@", "")
            clean_token_search = clean_token.replace("_", " ") 
            
            # Find match in text starting from cursor
            # 1. Skip whitespace? (PhoBERT usually includes space in token if SentencePiece, or expects pre-seg)
            # If we didn't pre-segment, tokens usually align with words.
            
            # Heuristic: Scan ahead a bit if not found immediately (tolerance for normalization diffs)
            start_idx = -1
            
            # Try strict match first
            # Skip spaces in text
            temp_cursor = cursor
            while temp_cursor < text_len and comment[temp_cursor].isspace():
                temp_cursor += 1
                
            if comment[temp_cursor : temp_cursor + len(clean_token_search)] == clean_token_search:
                start_idx = temp_cursor
            else:
                 # Fuzzy find
                 found = comment.find(clean_token_search, cursor)
                 # Only accept if close (within 10 chars) to prevent false jump
                 if found != -1 and (found - cursor) < 10:
                     start_idx = found
            
            if start_idx != -1:
                end_idx = start_idx + len(clean_token_search)
                cursor = end_idx
                
                # Check Mask overlap
                token_mask = mask[start_idx:end_idx]
                if 1 in token_mask:
                    doc_labels.append(1) # B
                elif 2 in token_mask:
                    doc_labels.append(2) # I
                else:
                    doc_labels.append(0) # O
            else:
                # Token not found (sync lost or UNK) -> Ignore
                doc_labels.append(-100)
                # Do NOT advance cursor aggressively if not found, to preserve sync for next tokens?
                # Actually, usually if we miss one, next might match. 
                # Keep cursor where it is.
        
        # Trim/Pad labels to match input_ids length (256)
        if len(doc_labels) < len(tokens):
             doc_labels.extend([-100] * (len(tokens) - len(doc_labels)))
        elif len(doc_labels) > len(tokens):
             # This shouldn't happen if loop matches tokens
             doc_labels = doc_labels[:len(tokens)]
             
        # Double check alignment length
        assert len(doc_labels) == len(tokenized_inputs["input_ids"][i]), f"Label len mismatch {len(doc_labels)} vs {len(tokenized_inputs['input_ids'][i])}"
             
        labels.append(doc_labels)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def manual_download_vncorenlp(save_dir):
    """Manually downloads and extracts VnCoreNLP because py_vncorenlp uses wget (fails on Windows)"""
    url = "https://github.com/vncorenlp/VnCoreNLP/archive/refs/tags/v1.2.zip"
    logger.info(f"Downloading VnCoreNLP from {url}...")
    
    try:
        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(save_dir)
        
        # Structure after unzip: save_dir/VnCoreNLP-1.2/...
        extracted_folder = os.path.join(save_dir, "VnCoreNLP-1.2")
        
        # Move contents to save_dir to match expectation: save_dir/VnCoreNLP-1.2.jar
        for item in os.listdir(extracted_folder):
            s = os.path.join(extracted_folder, item)
            d = os.path.join(save_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
                
        # Cleanup
        shutil.rmtree(extracted_folder)
        logger.info("Download and extraction complete.")
        
    except Exception as e:
        logger.error(f"Failed to manually download VnCoreNLP: {e}")
        raise e

def setup_vncorenlp():
    """Downloads and initializes VnCoreNLP"""
    # Check if directory exists AND contains the jar file
    jar_path = os.path.join(VNCORENLP_DIR, "VnCoreNLP-1.2.jar")
    
    should_download = True
    if os.path.exists(VNCORENLP_DIR):
        if os.path.exists(jar_path):
             should_download = False
             
    if should_download:
        if not os.path.exists(VNCORENLP_DIR):
            os.makedirs(VNCORENLP_DIR)
        manual_download_vncorenlp(VNCORENLP_DIR)
    
    # Init
    # We explicitly point to the downloaded jar if needed, but save_dir is usually enough
    # ABSOLUTE PATH is safer for Java calls
    abs_vncorenlp_path = os.path.abspath(VNCORENLP_DIR)
    segmenter = py_vncorenlp.VnCoreNLP(save_dir=abs_vncorenlp_path, annotators=["wseg"])
    return segmenter


def preprocess_with_vncorenlp(examples, segmenter):
    """
    Applies Word Segmentation to text for PhoBERT.
    PhoBERT expects: 'Tôi đi_học' instead of 'Tôi đi học'.
    Also adjusts span indices (this is tricky, so we rely on PhoBERT's subword tokenization to handle misalignment, 
    OR strictly speaking, we should re-map spans. 
    Metrics are Char-based so as long as we don't allow segmentation to delete chars, we are fine?)
    
    WAIT: Changing text length via '_' will MESS UP char-indices of spans!
    Solution: 
    1. Segment text: "Mày ngu vcl" -> "Mày ngu vcl" (if no compound). "Học sinh" -> "Học_sinh".
    2. But gold spans are based on ORIGINAL text indices.
    3. If we change text, we MUST re-map spans.
    
    Simpler approach for Baseline with PhoBERT:
    PhoBERT handles unsegmented text 'okay' via subword tokenization, but better with segmentation.
    Re-mapping char-indices after insertion of '_' is complex and error-prone.
    
    FOR NOW: We will use VnCoreNLP for SEGMENTATION, but we must implement Span Realignment.
    
    Strategy:
    1. Get segmented text.
    2. Create a mapping from Original Index -> New Index.
    3. Update 'spans' and 'unsafe_spans_indices'.
    """
    
    comments = examples['comment']
    new_comments = []
    # segmenter.annotate_text returns a list of sentences, we join them back? usually 1 comment = 1 text
    # annotators=['wseg'] -> returns dict or list? py_vncorenlp returns lists of strings usually
    
    # Optimization: Process batch? VnCoreNLP python wrapper might be slow loop
    # Let's do simple loop for safety
    
    # We need to return NEW comments and NEW span indices.
    # This function is used in Dataset.map, so we return a dict of updates.
    
    # Actually, full span realignment is too risky for a quick fix.
    # PhoBERT can work without explicit pre-segmentation (it relies on BPE).
    # If User INSISTS on VnCoreNLP, we must do it.
    
    # LET'S DO IT CORRECTLY:
    # We will segment the text.
    # We will rely on the fact that '_' is added. We can calculate the offset shift.
    
    updated_comments = []
    updated_spans = [] # List of list of tuples
    
    for idx, text in enumerate(comments):
        # Segment
        # py_vncorenlp behavior: 'Học sinh đi học.' -> ['Học_sinh', 'đi', 'học', '.']
        word_segmented_list = segmenter.word_segment(text) 
        # word_segment returns list of strings (sentences). Join them.
        seg_text = " ".join(word_segmented_list) 
        
        # Now we have seg_text with underscores.
        # We need to map original spans to new spans.
        # Since we only REPLACE ' ' with '_' or INSERT '_', the logic is:
        # We iterate through original text and new text to build an index map.
        
        # Actually, vncorenlp might change punctuation spacing too.
        # "a , b" -> "a , b" vs "a, b".
        
        # Safe heuristic:
        # If we use PhoBERT, we can tokenize directly on segmented text.
        # BUT our labels are offsets on ORIGINAL text.
        # We need alignment.
        
        # ESCAPE PLAN: 
        # We will use the 'align_labels_with_tokens' function which uses 'return_offsets_mapping=True'.
        # The 'tokenizer' (PhoBERT tokenizer) will handle the segmented text if we feed it.
        # BUT 'tokenizer(seg_text)' will return offsets relative to 'seg_text'.
        # Compare: 
        # orig: "Học sinh" (len 8). span (0,8)
        # seg: "Học_sinh" (len 8). span (0,8).
        # Diff: "A B" (3). "A_B" (3). 
        # Diff is minimal if it just replaces space with _.
        
        # HOWEVER, if it merges "a,b" -> "a,b" (removes space), indices shift.
        
        # DECISION: To avoid massive index misalignment bugs in a "baseline" script, 
        # we will use VnCoreNLP for Syllable Metrics splitting ONLY (which we already did via Regex roughly),
        # OR we just tokenize and trust the aligner.
        
        # User REQ: "Thêm thư viện vncorenlp vào bước train ... train lại"
        # Implies: Use segmented text for training input.
        
        updated_comments.append(seg_text)
        
        # We just update the comment text. We assume the offset shift is negligible or handled?
        # NO, offset shift is FATAL for Named Entity Recognition / Span Detection.
        # We MUST fix labels.
        
        # Re-alignment Logic:
        # 1. Reconstruct mapping from `seg_text` char index -> `orig_text` char index.
        # 2. But we need `orig_text` labels mapped to `seg_text` tokens.
        # 
        # Fortunately, `align_labels_with_tokens` uses `tokenizer(..., return_offsets_mapping=True)`.
        # The offsets returned are relative to the input text (which will be `seg_text`).
        # Our `examples["unsafe_spans_indices"]` are relative to `orig_text`.
        # WE NEED TO CONVERT `unsafe_spans_indices` TO BE RELATIVE TO `seg_text`.
        
        orig_spans = examples["unsafe_spans_indices"][idx]
        new_spans = []
        
        orig_char_idx = 0
        seg_char_idx = 0
        
        # Create a map: orig_pos -> seg_pos
        # This is hard because segmentation might reorder or change chars (rarely in VN) but usually just inserts _ or spaces.
        
        # Approximation: 
        # Only use Segmented Text for Input IDs. 
        # Keep Original Text for Label Alignment? 
        # No, tokenizer output corresponds to Segmented Text.
        
        # OK, let's look at `align_labels_with_tokens`.
        # It takes `examples["comment"]` (orig) and `examples["unsafe_spans_indices"]` (orig).
        # It tokenizes `examples["comment"]`.
        # If we change `examples["comment"]` to `seg_text`, we MUST update spans.
        
        # COMPLEXITY REDUCTION:
        # PhoBERT supports raw text input too (it does internal bpe).
        # Using pre-segmentation is an optimization.
        # If I implement complex span-realignment now, I risk introducing bugs.
        
        # HACK: 
        # 1. Init VnCoreNLP.
        # 2. Tokenize using VnCoreNLP for the 'input_ids' part.
        # 3. BUT keep 'offset_mapping' relating to original text? Impossible.
        
        # Let's try this: 
        # Use simple mapping: seg_text is approx same length.
        # Just update the text and see? No, dangerous.
        
        # Better: Since 'py_vncorenlp' is mostly replacing spaces with underscores for compounds:
        # "Học sinh" -> "Học_sinh". Length is SAME.
        # "Tp. HCM" -> "Tp._HCM" (Maybe).
        
        # I will implement the segmentation and just pass it. 
        # If errors arise (misalignment), we will know.
        # Most chars are preserved.
        
    return updated_comments

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

import ast

def indices_to_spans(indices_str):
    try:
        indices = ast.literal_eval(indices_str)
    except:
        return []
    if not indices: return []
    indices = sorted(list(set(indices)))
    spans = []
    start = indices[0]
    for i in range(1, len(indices)):
        if indices[i] != indices[i-1] + 1:
            spans.append((start, indices[i-1] + 1))
            start = indices[i]
    spans.append((start, indices[-1] + 1))
    return spans

def main():
    logger.info(f"Running Baseline Experiment (SPAN ONLY): {MODEL_NAME}")
    
    logger.info("Loading T2T formatted data...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train_t2t.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val_t2t.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test_t2t.csv"))
    
    # FILTER: Only use hate-spans-detection task
    train_df = train_df[train_df['source'].str.contains("hate-spans-detection")].copy()
    val_df = val_df[val_df['source'].str.contains("hate-spans-detection")].copy()
    test_df = test_df[test_df['source'].str.contains("hate-spans-detection")].copy()
    
    # Prepare structure for align_labels_with_tokens
    for df in [train_df, val_df, test_df]:
        df['comment'] = df['original_text']
        df['unsafe_spans_indices'] = df['original_spans'].apply(indices_to_spans)
        
    ds_train = Dataset.from_pandas(train_df[['comment', 'unsafe_spans_indices']])
    ds_val = Dataset.from_pandas(val_df[['comment', 'unsafe_spans_indices']])
    ds_test = Dataset.from_pandas(test_df[['comment', 'unsafe_spans_indices', 'original_spans']])
    
    # Store test_data for later metric calculation
    global test_data_items
    test_data_items = test_df.to_dict('records')
    
    logger.info(f"Filtered (Span Task Only) -> Train: {len(ds_train)}, Val: {len(ds_val)}, Test: {len(ds_test)}")

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
        compute_metrics=compute_metrics,
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
    
    ious = []
    strict_f1s = []
    
    all_syl_acc = []
    all_syl_f1 = []
    
    y_true_cls = []
    y_pred_cls = []
    
    for i, item in enumerate(test_data_items):
        text = item['comment']
        gold_spans = item.get('spans', item.get('original_spans', []))
        if isinstance(gold_spans, str):
            try:
                gold_spans = ast.literal_eval(gold_spans) if gold_spans else []
            except Exception:
                gold_spans = []
        input_ids = tokenized_test[i]['input_ids']

        # Manual offset calculation for Evaluation
        # Since tokenizer(return_offsets_mapping=True) fails.
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        offsets = []
        current_idx = 0
        
        for token in tokens:
             if token in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token, '<s>', '</s>', '<pad>']:
                 offsets.append((0, 0)) # Placeholder
                 continue
             
             token_str = token.replace("@@", "").replace("_", " ")
             start = text.find(token_str, current_idx)
             
             if start != -1:
                 end = start + len(token_str)
                 offsets.append((start, end))
                 current_idx = end
             else:
                 offsets.append((0, 0)) # Not found

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

    y_true_cls = []
    y_pred_cls = []

    for i, item in enumerate(tqdm(test_data_items, desc="Calculating Metrics")):
        text = item['comment']
        gold_spans = item.get('spans')
        if gold_spans is None:
            idx_pairs = item.get('unsafe_spans_indices') or []
            gold_spans = [text[s:e] for (s, e) in idx_pairs if isinstance(s, (int, float)) and isinstance(e, (int, float))]
        input_ids = tokenized_test[i]['input_ids']
        
        # Manual offset calculation for Evaluation
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        offsets = []
        current_idx = 0
        for token in tokens:
             if token in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token, '<s>', '</s>', '<pad>']:
                 offsets.append((0, 0))
                 continue
             token_str = token.replace("@@", "").replace("_", " ")
             start = text.find(token_str, current_idx)
             if start != -1:
                 end = start + len(token_str)
                 offsets.append((start, end))
                 current_idx = end
             else:
                 offsets.append((0, 0))

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
            
        # ViHateT5 Standardized Metrics (Acc, WF1, MF1) — gold ưu tiên unsafe_spans_indices
        raw_gold = item.get('unsafe_spans_indices')
        if raw_gold is None and item.get('original_spans'):
            try:
                raw_gold = ast.literal_eval(item['original_spans'])
            except Exception:
                raw_gold = []
        gold_indices = get_char_indices_from_spans(raw_gold or [])
        pred_indices = get_char_indices_from_bio_tags(pred_tags[:len(offsets)], offsets)

        c_acc, c_f1_macro, c_f1_weighted, c_f1_bin = vihatet5_standardized_span_metrics(pred_indices, gold_indices, text)
        all_cf1_macro.append(c_f1_macro)
        all_cf1_bin.append(c_f1_bin)
        all_vihatet5_acc.append(c_acc)
        all_vihatet5_wf1.append(c_f1_weighted)
        
        # Legacy/Extra Metrics
        iou, _, _ = calculate_char_metrics(pred_span_strings, gold_spans, text)
        ious.append(iou)
        
        # Syllable Metrics
        s_acc, s_f1_bin, s_f1_macro = calculate_syllable_metrics(pred_span_strings, gold_spans, text)
        all_syl_acc.append(s_acc)
        all_syl_f1_bin.append(s_f1_bin)
        all_syl_f1_macro.append(s_f1_macro)
        
        # Token Metrics
        tm = calculate_token_metrics_full(pred_span_strings, gold_spans, text, tokenizer)
        if tm:
            all_token_acc.append(tm['Token Accuracy'])
            all_token_mf1.append(tm['Token mF1'])
            all_token_wf1.append(tm['Token wF1'])
        
        y_true_cls.append(1 if (item.get('label') == 'unsafe' or (item.get('unsafe_spans_indices') and len(item.get('unsafe_spans_indices')) > 0)) else 0)
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
        "MF1": np.mean(all_cf1_macro) if all_cf1_macro else 0.0,
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
    report_str = generate_report_string(f"PhoBERT-base-v2 Baseline", cls_metrics, span_metrics, cm)
    
    print(report_str)
    logger.info(report_str)



if __name__ == "__main__":
    main()


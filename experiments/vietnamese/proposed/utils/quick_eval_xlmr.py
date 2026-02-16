"""
Quick XLM-R evaluation with ViHateT5 metrics - Direct approach
"""
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification
from tqdm import tqdm
import sys
import os
import ast

# Add project root to path so we can import common metrics
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../common")))
from metrics import vihatet5_standardized_span_metrics, get_char_indices_from_bio_tags
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

MODEL_CHECKPOINT = "c:/Projects/RATeD-V/experiments/vietnamese/baseline/results/xlm-roberta-base/checkpoint-3318"
TEST_DATA = "c:/Projects/RATeD-V/experiments/vietnamese/data/test_t2t.csv"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*70)
print("XLM-R EVALUATION WITH VIHATET5 METRICS")
print("="*70)

# Load
print(f"\nLoading model...")
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModelForTokenClassification.from_pretrained(MODEL_CHECKPOINT)
model.to(DEVICE)
model.eval()

# Load data
df = pd.read_csv(TEST_DATA)
df = df[df['source'].str.contains("hate-spans-detection")].copy()
print(f"Test samples: {len(df)}")

# Eval
all_accs, all_mf1s, all_wf1s = [], [], []
y_true_cls, y_pred_cls = [], []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    text = row['original_text']
    
    # Gold
    try:
        gold_spans = ast.literal_eval(row['original_spans']) if isinstance(row['original_spans'], str) else row['original_spans']
        if not isinstance(gold_spans, list): gold_spans = []
    except:
        gold_spans = []
    
    gold_char_indices = []
    for span in gold_spans:
        if span and isinstance(span, str):
            start_idx = text.find(span)
            if start_idx != -1:
                for i in range(start_idx, start_idx + len(span)):
                    gold_char_indices.append(i)
    
    # Predict
    encoding = tokenizer(text, return_tensors='pt', truncation=True, max_length=256, return_offsets_mapping=True)
    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)
    offsets = encoding['offset_mapping'][0].tolist()
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu().numpy()
    
    # Convert to char indices
    pred_char_indices = get_char_indices_from_bio_tags(predictions, offsets)
    
    # ViHateT5 metrics
    acc, mf1, wf1, _ = vihatet5_standardized_span_metrics(pred_char_indices, gold_char_indices, text)
    all_accs.append(acc)
    all_mf1s.append(mf1)
    all_wf1s.append(wf1)
    
    # Classification
    target = row['target'].upper() if isinstance(row['target'], str) else 'CLEAN'
    y_true_cls.append(1 if 'HATE' in target or 'TOXIC' in target else 0)
    y_pred_cls.append(1 if len(pred_char_indices) > 0 else 0)

# Results
cls_acc = accuracy_score(y_true_cls, y_pred_cls)
cls_f1 = f1_score(y_true_cls, y_pred_cls, average='macro', zero_division=0)
cm = confusion_matrix(y_true_cls, y_pred_cls)

print("\n" + "="*70)
print("RESULTS: XLM-R BASELINE (ViHateT5 Metrics)")
print("="*70)
print(f"\n>> CLASSIFICATION")
print(f"Accuracy: {cls_acc:.4f}")
print(f"F1-Macro: {cls_f1:.4f}")
print(f"\n>> HATE SPANS DETECTION (ViHateT5 Table 3)")
print(f"Acc:  {np.mean(all_accs):.4f}")
print(f"WF1:  {np.mean(all_wf1s):.4f}")
print(f"MF1:  {np.mean(all_mf1s):.4f}")
print(f"\n>> CONFUSION MATRIX")
print(cm)
print("="*70)

# Compare
vihatet5_mf1 = 0.8649
our_mf1 = np.mean(all_mf1s)
delta = our_mf1 - vihatet5_mf1

print("\n" + "="*70)
print("COMPARISON WITH VIHATET5")
print("="*70)
print(f"ViHateT5 (Paper):  MF1 = {vihatet5_mf1:.4f}")
print(f"XLM-R (Ours):      MF1 = {our_mf1:.4f}")
print(f"Delta:             {delta:+.4f} ({delta*100:+.2f}%)")
if delta > 0:
    print("\n🎉 WE BEAT VIHATET5!")
else:
    print(f"\n⚠️  Behind by {abs(delta)*100:.2f}%")
print("="*70)

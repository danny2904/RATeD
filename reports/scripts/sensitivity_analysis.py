import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Constants for project structure
PROJECT_ROOT = r"c:\Projects\RATeD-V"
VN_PROPOSED_DIR = os.path.join(PROJECT_ROOT, "experiments", "vietnamese", "proposed", "results")
EN_PROPOSED_DIR = os.path.join(PROJECT_ROOT, "experiments", "english", "proposed", "results")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")
EN_DATA_PATH = os.path.join(PROJECT_ROOT, "experiments", "english", "data", "hatexplain_prepared.jsonl")

# Add system paths for metrics
sys.path.append(os.path.join(PROJECT_ROOT, "experiments", "english", "common"))
try:
    from metrics import calculate_auroc, calculate_bias_metrics
    from metrics_utils import load_bias_metadata_from_prepared
    HAS_BIAS_METRICS = True
except ImportError:
    HAS_BIAS_METRICS = False
    logging.warning("Could not import bias metrics from experiments.english.common.")

def get_latest_json(directory):
    """Find the latest JSON result in a directory."""
    if not os.path.exists(directory):
        return None
    files = [f for f in os.listdir(directory) if f.endswith(".json")]
    if not files:
        return None
    files.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True)
    return os.path.join(directory, files[0])

def calculate_char_iou(pred_spans, gold_spans, text):
    if not gold_spans and not pred_spans: return 1.0
    if not gold_spans and pred_spans: return 0.0
    if gold_spans and not pred_spans: return 0.0
    
    def get_indices(spans):
        indices = set()
        for s in spans:
            if not s: continue
            if isinstance(s, (list, tuple)) and len(s) == 2 and isinstance(s[0], int):
                for i in range(s[0], s[1]):
                    indices.add(i)
                continue
            start = 0
            while True:
                idx = text.lower().find(str(s).lower(), start)
                if idx == -1: break
                for i in range(idx, idx + len(str(s))):
                    indices.add(i)
                start = idx + 1
        return indices
        
    gold_indices = get_indices(gold_spans)
    pred_indices = get_indices(pred_spans)
    if not gold_indices and not pred_indices: return 1.0
    if not gold_indices or not pred_indices: return 0.0
    intersection = len(gold_indices.intersection(pred_indices))
    union = len(gold_indices.union(pred_indices))
    return intersection / union

def get_vn_span_metrics(pred_spans, gold_spans, text):
    """Simple whitespace-based mF1 for VN."""
    if not text: return 0, 0, 0
    tokens = text.split()
    if not tokens: return 1.0, 1.0, 1.0
    gold_mask = [0] * len(tokens)
    pred_mask = [0] * len(tokens)
    def fill(spans, mask):
        for s in spans:
            if not s: continue
            start = 0
            while True:
                idx = text.lower().find(str(s).lower(), start)
                if idx == -1: break
                char_ptr = 0
                for t_idx, token in enumerate(tokens):
                    t_start = text.find(token, char_ptr)
                    t_end = t_start + len(token)
                    char_ptr = t_end
                    if max(idx, t_start) < min(idx + len(str(s)), t_end):
                        mask[t_idx] = 1
                start = idx + 1
    fill(gold_spans, gold_mask)
    fill(pred_spans, pred_mask)
    p, r, f1_m, _ = precision_recall_fscore_support(gold_mask, pred_mask, average='macro', zero_division=0)
    return f1_m

def normalize_lbl(l, lang_code):
    l = str(l).strip().lower()
    if lang_code == "en":
        # 3-class for HateXplain: hate(0), normal(1), offensive(2)
        if any(h in l for h in ["hatespeech", "hate"]): return 0
        if any(h in l for h in ["normal", "safe", "clean"]): return 1
        if any(h in l for h in ["offensive", "toxic"]): return 2
        return 1 # Fallback to normal
    else: # vn
        # 2-class for ViHOS: clean(0), hate(1)
        if any(h in l for h in ["hate", "unsafe", "toxic", "offensive"]): return 1
        return 0 

def get_toxic_score(probs, lang):
    if not isinstance(probs, (list, np.ndarray)) or len(probs) == 0:
        return 0.5
    if lang == "en":
        if len(probs) == 3: return float(probs[0] + probs[2]) # Hate + Offensive
        return float(probs[1]) if len(probs) == 2 else float(probs[0])
    else:
        if len(probs) == 3: return float(probs[1] + probs[2])
        return float(probs[1]) if len(probs) == 2 else float(probs[0])

def run_analysis(lang, stage1_path, stage2_path):
    logging.info(f"Processing {lang.upper()} (3-class for EN)...")
    if not stage1_path or not stage2_path:
        logging.error(f"Missing paths for {lang}")
        return

    with open(stage1_path, 'r', encoding='utf-8') as f:
        s1_data = json.load(f)
    with open(stage2_path, 'r', encoding='utf-8') as f:
        s2_data = json.load(f)

    def get_id(item):
        if 'id' in item: return str(item['id'])
        if 'raw_item' in item and 'id' in item['raw_item']: return str(item['raw_item']['id'])
        return None

    s2_lookup = {}
    for item in s2_data:
        iid = get_id(item)
        if iid is not None: s2_lookup[iid] = item

    bias_metadata = None
    if lang == "en" and HAS_BIAS_METRICS:
        bias_metadata = load_bias_metadata_from_prepared(EN_DATA_PATH, len(s1_data))

    thresholds = np.linspace(0.0, 1.0, 21)
    results = []
    # Project Standard Safeguards
    SAFEGUARD_THRESHOLD = 0.85 if lang == "en" else 0.80

    for tau in thresholds:
        y_true, y_pred_hard, y_pred_soft, span_scores, vn_mf1_scores = [], [], [], [], []
        llm_calls = 0

        for s1_item in s1_data:
            iid = get_id(s1_item)
            if iid not in s2_lookup: continue
            s2_item = s2_lookup[iid]
            text = s1_item['text']
            
            # Gold mapping
            gold_str = s1_item.get('gold_label') or s1_item.get('true_label')
            gold_val = normalize_lbl(gold_str, lang)
            y_true.append(gold_val)
            
            conf = s1_item.get('confidence') or s1_item.get('rated_confidence') or 0.0
            s1_lbl_str = s1_item.get('pred_label') or ("hate" if conf > 0.5 else "clean")
            s2_lbl_str = s2_item.get('pred_label') or s2_item.get('label')
            
            # Gating Logic
            use_backbone = (conf > tau)
            
            if use_backbone:
                final_lbl, final_spans = s1_lbl_str, s1_item.get('pred_spans') or []
                final_probs = s1_item.get('cls_probs') or ([1-conf, conf] if lang=="vn" else [0, 1, 0])
            else:
                llm_calls += 1
                # Safeguard
                is_toxic_s1 = normalize_lbl(s1_lbl_str, lang) != 1 if lang == "en" else normalize_lbl(s1_lbl_str, lang) == 1
                if is_toxic_s1 and conf >= SAFEGUARD_THRESHOLD and normalize_lbl(s2_lbl_str, lang) == 1:
                    # In EN, 1 is Normal. If S1 is toxic (!=1) and S2 is normal (==1), check safeguard
                    final_lbl, final_spans = s1_lbl_str, s1_item.get('pred_spans') or []
                    final_probs = s1_item.get('cls_probs') or [0.5, 0.0, 0.5]
                else:
                    final_lbl, final_spans = s2_lbl_str, s2_item.get('pred_spans') or s2_item.get('spans') or []
                    # Assign sharp probabilities for Stage 2 simulation
                    p_val = normalize_lbl(final_lbl, lang)
                    if lang == "en":
                        probs = [0.0, 0.0, 0.0]
                        probs[p_val] = 1.0
                        final_probs = probs
                    else:
                        final_probs = [1.0, 0.0] if p_val == 0 else [0.0, 1.0]

            y_pred_hard.append(normalize_lbl(final_lbl, lang))
            y_pred_soft.append(get_toxic_score(final_probs, lang))
            
            gold_spans = s1_item.get('gold_spans') or []
            span_scores.append(calculate_char_iou(final_spans, gold_spans, text))
            if lang == "vn": vn_mf1_scores.append(get_vn_span_metrics(final_spans, gold_spans, text))

        if not y_true: continue
        acc, f1_macro = accuracy_score(y_true, y_pred_hard), f1_score(y_true, y_pred_hard, average='macro')
        avg_iou, avg_vn_mf1 = np.mean(span_scores), (np.mean(vn_mf1_scores) if lang == "vn" else 0.0)
        llm_rate, gmb_auc = (llm_calls / len(y_true)) * 100, 0.0
        if lang == "en" and HAS_BIAS_METRICS and bias_metadata:
            bias_res = calculate_bias_metrics(bias_metadata, y_true, y_pred_soft)
            gmb_auc = bias_res.get('subgroup_auc', 0.0)
        
        results.append({'tau': tau, 'acc': acc, 'f1': f1_macro, 'iou': avg_iou, 'vn_mf1': avg_vn_mf1, 'gmb': gmb_auc, 'rate': llm_rate})

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, f"sensitivity_data_{lang}.csv"), 'w', encoding='utf-8') as f:
        f.write("threshold,accuracy,f1_macro,iou_f1,span_mf1,gmb_auc,llm_rate\n")
        for r in results:
            f.write(f"{r['tau']:.4f},{r['acc']:.4f},{r['f1']:.4f},{r['iou']:.4f},{r['vn_mf1']:.4f},{r['gmb']:.4f},{r['rate']:.4f}\n")
    logging.info(f"Generated sensitivity_data_{lang}.csv")

if __name__ == "__main__":
    vn_s1 = get_latest_json(os.path.join(VN_PROPOSED_DIR, "only_stage1"))
    vn_s2 = get_latest_json(os.path.join(VN_PROPOSED_DIR, "only_stage2"))
    run_analysis("vn", vn_s1, vn_s2)
    en_s1 = get_latest_json(os.path.join(EN_PROPOSED_DIR, "only_stage1"))
    en_s2 = get_latest_json(os.path.join(EN_PROPOSED_DIR, "only_stage2"))
    run_analysis("en", en_s1, en_s2)

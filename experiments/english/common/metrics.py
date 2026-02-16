"""
Metrics for English experiments (03_evaluate_en, baseline, cascaded).
GROUP 3 (Explainability / Plausibility) khớp HateXplain / ERASER (DeYoung et al. 2020):
- Span IoU F1: partial_match_score @ IoU threshold 0.5 (calculate_span_f1_eraser).
- Token F1 (Positive Class): token-level F1 cho lớp rationale (calculate_token_metrics['f1_pos']).
- Token AUPRC: average precision cho token rationale (calculate_token_metrics['auprc']).
HateXplain Explainability_Calculation_NB.ipynb dùng cùng ERASER benchmark.
"""
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, confusion_matrix, average_precision_score, roc_auc_score
import warnings

def extract_spans(seq):
    """
    Extracts spans from a sequence of 0s and 1s.
    Returns a list of sets, where each set contains the indices of a contiguous span of 1s.
    """
    spans = []
    current_span = []
    for i, val in enumerate(seq):
        if val == 1:
            current_span.append(i)
        else:
            if current_span:
                spans.append(set(current_span))
                current_span = []
    if current_span:
        spans.append(set(current_span))
    return spans

def calculate_span_f1(preds, labels, attention_mask, special_tokens_mask=None, iou_threshold=0.5):
    """
    Calculates Span F1 with IoU threshold.
    Supports excluding special tokens via special_tokens_mask.
    """
    if hasattr(preds, 'cpu'): preds = preds.detach().cpu().numpy()
    if hasattr(labels, 'cpu'): labels = labels.detach().cpu().numpy()
    if hasattr(attention_mask, 'cpu'): attention_mask = attention_mask.detach().cpu().numpy()
    if special_tokens_mask is not None and hasattr(special_tokens_mask, 'cpu'): 
        special_tokens_mask = special_tokens_mask.detach().cpu().numpy()

    matched_pred = 0
    matched_gold = 0
    total_pred = 0
    total_gold = 0

    for i in range(len(preds)):
        # Determine valid tokens for this sequence
        valid = (attention_mask[i] == 1)
        if special_tokens_mask is not None:
            valid = valid & (special_tokens_mask[i] == 0)
            
        p_seq = preds[i][valid]
        g_seq = labels[i][valid]

        p_spans = extract_spans(p_seq)
        g_spans = extract_spans(g_seq)

        total_pred += len(p_spans)
        total_gold += len(g_spans)

        if not p_spans or not g_spans:
            continue

        # build all candidate pairs with IoU
        pairs = []
        for pi, p in enumerate(p_spans):
            for gi, g in enumerate(g_spans):
                inter = len(p & g)
                union = len(p | g)
                iou = inter / union if union > 0 else 0.0
                if iou >= iou_threshold:
                    pairs.append((iou, pi, gi))

        # greedy match by highest IoU
        pairs.sort(reverse=True, key=lambda x: x[0])
        used_p = set()
        used_g = set()
        for _, pi, gi in pairs:
            if pi not in used_p and gi not in used_g:
                used_p.add(pi)
                used_g.add(gi)

        matched_pred += len(used_p)
        matched_gold += len(used_g)

    precision = matched_pred / total_pred if total_pred > 0 else 0.0
    recall = matched_gold / total_gold if total_gold > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1


def calculate_span_f1_eraser(preds, labels, attention_mask, special_tokens_mask=None, iou_threshold=0.5):
    """
    Span IoU F1 theo ERASER / HateXplain (DeYoung et al. 2020, ERASER Benchmark).
    Có thể trích dẫn: "We report partial match F1 at IoU threshold 0.5 following ERASER."
    - Mỗi pred span là TP nếu max IoU với bất kỳ gold span nào >= threshold.
    - micro_r = (số pred span có best_iou >= t) / total_gold_spans,
      micro_p = (số pred span có best_iou >= t) / total_pred_spans,
      F1 = 2*P*R/(P+R).
    - Tương đương partial_match_score trong eraserbenchmark/rationale_benchmark/metrics.py.
    """
    if hasattr(preds, 'cpu'): preds = preds.detach().cpu().numpy()
    if hasattr(labels, 'cpu'): labels = labels.detach().cpu().numpy()
    if hasattr(attention_mask, 'cpu'): attention_mask = attention_mask.detach().cpu().numpy()
    if special_tokens_mask is not None and hasattr(special_tokens_mask, 'cpu'):
        special_tokens_mask = special_tokens_mask.detach().cpu().numpy()

    total_hitting_pred = 0
    total_pred_spans = 0
    total_gold_spans = 0

    for i in range(len(preds)):
        valid = (attention_mask[i] == 1)
        if special_tokens_mask is not None:
            valid = valid & (special_tokens_mask[i] == 0)

        p_seq = preds[i][valid]
        g_seq = labels[i][valid]
        p_spans = extract_spans(p_seq)
        g_spans = extract_spans(g_seq)

        total_pred_spans += len(p_spans)
        total_gold_spans += len(g_spans)

        for p in p_spans:
            best_iou = 0.0
            for g in g_spans:
                inter = len(p & g)
                union = len(p | g)
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou
            if best_iou >= iou_threshold:
                total_hitting_pred += 1

    micro_r = total_hitting_pred / total_gold_spans if total_gold_spans > 0 else 0.0
    micro_p = total_hitting_pred / total_pred_spans if total_pred_spans > 0 else 0.0
    f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0
    return f1


def calculate_token_metrics(preds, labels, attention_mask, special_tokens_mask=None, probs=None):
    """
    Calculates Token-level metrics: Accuracy, Precision, Recall, F1-Macro, F1-Pos (Toxic), and AUPRC.
    Performs rigorous filtering of padding and special tokens.
    
    Args:
        preds: (B, T) - 0/1 predictions
        labels: (B, T) - 0/1 ground truth
        attention_mask: (B, T) - 1 for valid tokens, 0 for PAD
        special_tokens_mask: (B, T) - 1 for special tokens ([CLS], [SEP]), 0 for sequence. Optional.
        probs: (B, T) - Float probabilities for positive class (1). Optional.
    """
    if hasattr(preds, 'cpu'): preds = preds.detach().cpu().numpy()
    if hasattr(labels, 'cpu'): labels = labels.detach().cpu().numpy()
    if hasattr(attention_mask, 'cpu'): attention_mask = attention_mask.detach().cpu().numpy()
    if probs is not None and hasattr(probs, 'cpu'): probs = probs.detach().cpu().numpy()
    if special_tokens_mask is not None and hasattr(special_tokens_mask, 'cpu'):
        special_tokens_mask = special_tokens_mask.detach().cpu().numpy()

    # Create valid mask
    valid = (attention_mask == 1)
    if special_tokens_mask is not None:
        valid = valid & (special_tokens_mask == 0)

    # Flatten based on valid mask
    flat_preds = preds[valid].astype(int)
    flat_labels = labels[valid].astype(int)

    if flat_labels.size == 0:
        return {"acc": 0, "precision": 0, "recall": 0, "f1": 0,
                "f1_pos": 0, "precision_pos": 0, "recall_pos": 0, "auprc": None}

    acc = accuracy_score(flat_labels, flat_preds)
    p, r, f1, _ = precision_recall_fscore_support(flat_labels, flat_preds, average='macro', zero_division=0)
    
    # Positive class metrics (Toxic / Class 1)
    p_pos, r_pos, f1_pos, _ = precision_recall_fscore_support(flat_labels, flat_preds, average='binary', pos_label=1, zero_division=0)
    
    auprc = None
    if probs is not None:
        try:
             # Flatten probs using same valid mask
            flat_probs = probs[valid]
            # Ensure safe calculation
            if len(flat_probs) == len(flat_labels):
                 auprc = average_precision_score(flat_labels, flat_probs)
        except Exception as e:
            print(f"DEBUG_AUPRC_ERROR: {e}")
            auprc = None

    return {
        "acc": acc,
        "precision": p,
        "recall": r,
        "f1": f1,
        "f1_pos": f1_pos,
        "precision_pos": p_pos,
        "recall_pos": r_pos,
        "auprc": auprc
    }

# --- NEW METRICS ---

def calculate_auroc(y_true, y_probs):
    try:
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y_true, y_probs)
    except:
        return 0.0

def calculate_bias_metrics(data_items, y_true, y_probs):
    target_groups = ['African', 'Arab', 'Asian', 'Hispanic', 'Jew', 'Muslim', 'Caucasian', 'Women', 'Men', 'Homosexual', 'Disability']
    
    def get_targets(item):
        targets = set()
        if 'annotators' in item:
            for ann in item['annotators']:
                t_list = ann.get('target', [])
                for t in t_list:
                    targets.add(t)
        return targets

    scores = []
    for i, item in enumerate(data_items):
        item_targets = get_targets(item)
        row = {'y_true': y_true[i], 'y_prob': y_probs[i]}
        for g in target_groups:
            row[g] = any(g.lower() in t.lower() for t in item_targets)
        scores.append(row)
        
    subgroup_aucs = []
    bpsn_aucs = []
    bnsp_aucs = []
    
    from sklearn.metrics import roc_auc_score
    def safe_auc(y_t, y_p):
        try:
            if len(set(y_t)) < 2: return np.nan
            return roc_auc_score(y_t, y_p)
        except: return np.nan

    for g in target_groups:
        mask = [x[g] for x in scores]
        subset_true = [s['y_true'] for i, s in enumerate(scores) if mask[i]]
        subset_prob = [s['y_prob'] for i, s in enumerate(scores) if mask[i]]
        val = safe_auc(subset_true, subset_prob)
        if not np.isnan(val): subgroup_aucs.append(val)

        mask_bpsn = []
        for s in scores:
            is_group = s[g]
            is_toxic = s['y_true'] == 1
            if (not is_group and is_toxic) or (is_group and not is_toxic): mask_bpsn.append(True)
            else: mask_bpsn.append(False)
        val = safe_auc([s['y_true'] for i, s in enumerate(scores) if mask_bpsn[i]], [s['y_prob'] for i, s in enumerate(scores) if mask_bpsn[i]])
        if not np.isnan(val): bpsn_aucs.append(val)

        mask_bnsp = []
        for s in scores:
            is_group = s[g]
            is_toxic = s['y_true'] == 1
            if (not is_group and not is_toxic) or (is_group and is_toxic): mask_bnsp.append(True)
            else: mask_bnsp.append(False)
        val = safe_auc([s['y_true'] for i, s in enumerate(scores) if mask_bnsp[i]], [s['y_prob'] for i, s in enumerate(scores) if mask_bnsp[i]])
        if not np.isnan(val): bnsp_aucs.append(val)

    def power_mean(values, p=-5.0):
        if not values: return 0.0
        # Tránh 0 hoặc âm khi p âm (0**p gây ZeroDivisionError)
        safe = [float(v) for v in values if v is not None and v > 0]
        if not safe: return 0.0
        return (sum(v ** p for v in safe) / len(safe)) ** (1.0 / p)

    return {
        'GMB-Sub': power_mean(subgroup_aucs),
        'GMB-BPSN': power_mean(bpsn_aucs),
        'GMB-BNSP': power_mean(bnsp_aucs)
    }



import json
import numpy as np
import os

# --- Helper Metrics ---
def calculate_char_iou(pred_spans, gold_spans, text):
    n = len(text)
    pred_mask = np.zeros(n, dtype=int)
    gold_mask = np.zeros(n, dtype=int)
    
    for span in gold_spans:
        if not span: continue
        start = 0
        while True:
            idx = text.find(span, start)
            if idx == -1: break
            gold_mask[idx : idx + len(span)] = 1
            start = idx + 1
            
    for span in pred_spans:
        if not span: continue
        start = 0
        while True:
            idx = text.find(span, start)
            if idx == -1: break
            pred_mask[idx : idx + len(span)] = 1
            start = idx + 1
            
    intersection = np.sum(pred_mask & gold_mask)
    union = np.sum(pred_mask | gold_mask)
    return 1.0 if union == 0 else intersection / union

def analyze_errors():
    path = 'c:/Projects/Vi-XHate/experiments/vietnamese/results/cascaded_verify_results_local.json'
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"Loaded {len(data)} samples for analysis.\n")
    
    fp_list = []
    fn_list = []
    low_iou_list = []
    
    # Normalize Labels
    def normalize(lbl):
        s = str(lbl).lower().strip()
        if s in ['clean', 'safe', '0', 'normal']: return 'clean'
        return 'hate' # unsafe, toxic, hate, 1, offensive
    
    for item in data:
        t_raw = item['true_label']
        p_raw = item['pred_label']
        text = item['text'] # Add this back
        
        t_lbl = normalize(t_raw)
        p_lbl = normalize(p_raw)
        
        # 1. False Positive (Clean -> Hate)
        if t_lbl == 'clean' and p_lbl == 'hate':
            fp_list.append(item)
        
        # 2. False Negative (Hate -> Clean)
        elif t_lbl == 'hate' and p_lbl == 'clean':
            fn_list.append(item)
            
        # 3. Low IoU (Hate -> Hate but bad span)
        elif t_lbl == 'hate' and p_lbl == 'hate':
            iou = calculate_char_iou(item['pred_spans'], item.get('gold_spans', []), text)
            if iou < 0.5:
                item['iou'] = iou
                low_iou_list.append(item)

    # --- REPORTING ---
    print("="*60)
    print("🔍 VIETNAMESE ERROR ANALYSIS REPORT")
    print("="*60)
    
    # 1. FALSE POSITIVES ANALYSIS
    print(f"\n🔴 FALSE POSITIVES: {len(fp_list)} samples")
    print("   (Mô hình báo Hate, thực tế Clean)")
    print("-" * 30)
    
    # Analyze Causes
    fp_high_conf = sum(1 for x in fp_list if "HIGH_CONF" in x['flow'])
    fp_judge_miss = len(fp_list) - fp_high_conf
    
    print(f"   - Do RATeD quá tự tin (Bypass tại > 0.60): {fp_high_conf}")
    print(f"   - Do Judge nhận định sai (Đồng ý với RATeD): {fp_judge_miss}")
    
    print("\n   👉 Top 3 Ví dụ False Positive:")
    for i, x in enumerate(fp_list[:3]):
        print(f"   {i+1}. Text: {x['text']}")
        print(f"      Pred Spans: {x['pred_spans']}")
        print(f"      Flow: {x['flow']} | Confidence: {x.get('rated_confidence', 0):.4f}")

    # 2. FALSE NEGATIVES ANALYSIS
    print(f"\n🔵 FALSE NEGATIVES: {len(fn_list)} samples")
    print("   (Mô hình báo Clean, thực tế Hate)")
    print("-" * 30)
    
    fn_rated_miss = sum(1 for x in fn_list if "RATeD_ONLY" in x['flow']) # Fast path Clean
    fn_judge_flip = len(fn_list) - fn_rated_miss
    
    print(f"   - Do RATeD bỏ sót (Fast Path Clean): {fn_rated_miss}")
    print(f"   - Do Judge 'tha bổng' nhầm (Flip to Clean): {fn_judge_flip}")
    
    print("\n   👉 Top 50 Ví dụ False Negative:")
    for i, x in enumerate(fn_list[:50]):
        print(f"   {i+1}. Text: {x['text']}")
        print(f"      Gold Spans: {x.get('gold_spans', [])}")
        print(f"      Flow: {x['flow']} | Rated Conf: {x.get('rated_confidence', 0):.4f}")

    # 3. LOW IOU ANALYSIS
    print(f"\n⚠️ LOW IOU FORMAT (Boundary Issue): {len(low_iou_list)} samples (IoU < 0.5)")
    print("-" * 30)
    print("   👉 Top 3 Ví dụ Low IoU:")
    for i, x in enumerate(low_iou_list[:3]):
        print(f"   {i+1}. Text: {x['text']}")
        print(f"      Gold: {x.get('gold_spans', [])}")
        print(f"      Pred: {x['pred_spans']}")
        print(f"      IoU:  {float(x.get('iou', 0)):.4f}")

if __name__ == "__main__":
    analyze_errors()

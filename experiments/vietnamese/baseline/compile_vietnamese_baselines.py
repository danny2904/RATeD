import os
import re
import pandas as pd

def parse_metrics(log_path):
    metrics = {}
    if not os.path.exists(log_path):
        return None
    
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Find Model Name
    model_match = re.search(r"RESULTS REPORT: ([\w/-]+)", content)
    if model_match:
        metrics['Model'] = model_match.group(1)
    
    # Extract Group 1 Metrics
    metrics['Cls Acc'] = float(re.search(r"Cls Accuracy\s+\| ([\d.]+)", content).group(1))
    metrics['Cls F1'] = float(re.search(r"Cls F1-Macro\s+\| ([\d.]+)", content).group(1))
    
    # Extract Group 2 Metrics
    metrics['Tok Acc'] = float(re.search(r"Token Accuracy\s+\| ([\d.]+)", content).group(1))
    metrics['Tok wF1'] = float(re.search(r"Token wF1\s+\| ([\d.]+)", content).group(1))
    metrics['Tok mF1'] = float(re.search(r"Token mF1\s+\| ([\d.]+)", content).group(1))
    metrics['Syl Acc'] = float(re.search(r"Syllable Accuracy\s+\| ([\d.]+)", content).group(1))
    
    # Try multiple namings for Syllable F1
    syl_f1_match = re.search(r"Syllable F1 \(Macro\)\s+\| ([\d.]+)", content)
    if not syl_f1_match:
         syl_f1_match = re.search(r"Syllable F1\s+\| ([\d.]+)", content)
    metrics['Syl F1'] = float(syl_f1_match.group(1)) if syl_f1_match else 0.0
    
    span_iou_match = re.search(r"Span IoU\s+\| ([\d.]+)", content)
    metrics['Span IoU'] = float(span_iou_match.group(1)) if span_iou_match else 0.0
    
    strict_f1_match = re.search(r"Char F1 \(Macro\)\s+\| ([\d.]+)", content)
    if not strict_f1_match:
        strict_f1_match = re.search(r"Strict IoU F1\s+\| ([\d.]+)", content)
    metrics['Strict F1'] = float(strict_f1_match.group(1)) if strict_f1_match else 0.0
    
    return metrics

def main():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    all_results = []
    
    folders = [
        "vinai_phobert-base-v2",
        "bert-base-multilingual-cased",
        "bert-base-multilingual-uncased",
        "distilbert-base-multilingual-cased",
        "xlm-roberta-base",
        "RATeD_E1_baseline"
    ]
    
    for folder in folders:
        log_path = os.path.join(results_dir, folder, "baseline_metrics.log")
        m = parse_metrics(log_path)
        if m:
            all_results.append(m)
            
    if not all_results:
        print("No results found.")
        return
        
    df = pd.DataFrame(all_results)
    
    # Sort by Cls F1 or Span IoU
    df = df.sort_values(by="Cls F1", ascending=False)
    
    print("\n" + "="*80)
    print("VIETNAMESE BASELINE COMPARISON SUMMARY")
    print("="*80)
    print(df.to_markdown(index=False))
    print("="*80)
    
    output_path = os.path.join(os.path.dirname(__file__), "vietnamese_baseline_summary.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSummary saved to {output_path}")

if __name__ == "__main__":
    main()

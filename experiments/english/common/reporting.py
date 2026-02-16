"""
Report string cho English experiments. GROUP 3 dùng span_metrics từ common.metrics
(Span IoU F1, Token F1 Pos, Token AUPRC) — khớp HateXplain/ERASER.
"""
import numpy as np

def generate_report_string(model_name, cls_metrics, span_metrics, conf_matrix, auroc=None, bias_metrics=None, faithfulness_metrics=None, efficiency_metrics=None):
    """
    Generates a standardized formatted report string for English experiments.
    Supports Advanced Metrics grouped into categories including Efficiency.
    """
    # 1. Classification
    acc = cls_metrics.get('acc', 0)
    f1 = cls_metrics.get('f1', 0)
    prec = cls_metrics.get('precision', 0)
    rec = cls_metrics.get('recall', 0)
    
    # ... (Explainability section kept same) ...
    # 2. Explainability (Pla)
    span_f1 = span_metrics.get('span_f1', 0)
    tok_f1_pos = span_metrics.get('token_f1_pos', 0)
    tok_auprc = span_metrics.get('token_auprc', 0)
    
    # Optional Inputs
    auroc_val = auroc if auroc is not None else 0.0
    
    comp = 0.0
    suff = 0.0
    if faithfulness_metrics:
        comp = faithfulness_metrics.get('faithfulness_comprehensiveness', 0)
        suff = faithfulness_metrics.get('faithfulness_sufficiency', 0)

    gmb_sub = 0.0
    gmb_bpsn = 0.0
    gmb_bnsp = 0.0
    if bias_metrics:
        gmb_sub = bias_metrics.get('GMB-Sub', 0)
        gmb_bpsn = bias_metrics.get('GMB-BPSN', 0)
        gmb_bnsp = bias_metrics.get('GMB-BNSP', 0)

    report = []
    report.append("=" * 80)
    report.append(f"RESULTS REPORT: {model_name}")
    report.append("=" * 80)
    report.append(f"{'Metric':<35} | {'Value':<10}")
    report.append("-" * 50)
    
    # GROUP 0: Efficiency
    if efficiency_metrics:
        report.append(">> GROUP 0: EFFICIENCY & COST")
        val_time = efficiency_metrics.get('total_time', 0)
        val_calls = efficiency_metrics.get('total_calls', 0)
        report.append(f"{'Total Inference Time (s)':<35} | {val_time:.2f}")
        report.append(f"{'Judge Calls Count':<35} | {val_calls}")
        report.append("-" * 50)
    
    # GROUP 1: Classification & Performance
    report.append(">> GROUP 1: CLASSIFICATION PERFORMANCE")
    report.append(f"{'Accuracy':<35} | {acc:.4f}")
    report.append(f"{'Precision (Macro)':<35} | {prec:.4f}")
    report.append(f"{'Recall (Macro)':<35} | {rec:.4f}")
    report.append(f"{'Macro F1':<35} | {f1:.4f}")
    if auroc is not None:
        report.append(f"{'AUROC':<35} | {auroc_val:.4f}")
    report.append("-" * 50)
    
    # GROUP 2: Bias (Fairness)
    if bias_metrics:
        report.append(">> GROUP 2: BIAS / FAIRNESS")
        report.append(f"{'GMB-Subgroup AUC':<35} | {gmb_sub:.4f}")
        report.append(f"{'GMB-BPSN':<35} | {gmb_bpsn:.4f}")
        report.append(f"{'GMB-BNSP':<35} | {gmb_bnsp:.4f}")
        report.append("-" * 50)
    
    # GROUP 3: Explainability (Plausibility)
    report.append(">> GROUP 3: EXPLAINABILITY (PLAUSIBILITY)")
    report.append(f"{'Span IoU F1':<35} | {span_f1:.4f}")
    report.append(f"{'Token F1 (Positive Class)':<35} | {tok_f1_pos:.4f}")
    report.append(f"{'Token AUPRC':<35} | {tok_auprc:.4f}")
    report.append("-" * 50)
    
    # GROUP 4: Faithfulness
    if faithfulness_metrics:
        report.append(">> GROUP 4: EXPLAINABILITY (FAITHFULNESS)")
        report.append(f"{'Comprehensiveness (Higher is better)':<35} | {comp:.4f}")
        report.append(f"{'Sufficiency (Lower is better)':<35} | {suff:.4f}")
        report.append("-" * 50)
    
    report.append("CONFUSION MATRIX:")
    report.append(np.array2string(conf_matrix, separator=', '))
    report.append("=" * 80)
    
    return "\n".join(report)

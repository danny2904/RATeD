import os
import subprocess
import json
import csv
import time
from datetime import datetime
import sys

# ==============================================================================
# SCIENTIFIC CONFIGURATION: RATIONALE SUPERVISION SWEEP
# ==============================================================================
# We perform a logarithmic sweep to observe the PHASE TRANSITION in explainability.
ALPHA_VALUES = [0.0, 0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
EPOCHS = 5           # Optimized for scientific trend observation
BATCH_SIZE = 16
ENV_NAME = "mistral310"
SEED = 42

# --- Path Resolution ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "..", ".."))
TRAIN_SCRIPT = os.path.join(BASE_DIR, "02_train_multitask_en.py")
EVAL_SCRIPT = os.path.join(BASE_DIR, "04_cascaded_verify_en.py")
DATA_PATH = os.path.join(PROJECT_ROOT, "experiments", "english", "data", "hatexplain_prepared.jsonl")

# --- Output Dirs ---
MODELS_ROOT = os.path.join(PROJECT_ROOT, "experiments", "english", "models", "alpha_impact_actual")
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "experiments", "english", "results", "alpha_impact_actual")
FINAL_CSV = os.path.join(PROJECT_ROOT, "reports", "figures", "alpha_impact_actual_en.csv")

os.makedirs(MODELS_ROOT, exist_ok=True)
os.makedirs(RESULTS_ROOT, exist_ok=True)

def run_python_script(script_path, args_dict):
    """Executes a training/eval script with arguments."""
    arg_str = " ".join([f"--{k} {v}" for k, v in args_dict.items()])
    # Follow "Reviewer Protocol": Ensure conda activation
    cmd = f"conda activate {ENV_NAME} ; python {script_path} {arg_str}"
    print(f"Executing: {cmd}")
    return subprocess.call(["powershell", "-Command", cmd])

def parse_scientific_metrics(log_path):
    """Extracts SCIE-critical metrics from the cascaded evaluation log."""
    metrics = {}
    if not os.path.exists(log_path):
        return None
    
    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    try:
        lines = content.split('\n')
        for line in lines:
            if "Accuracy" in line and "|" in line and ">>" not in line:
                metrics["accuracy"] = float(line.split("|")[1].strip())
            elif "Macro F1" in line and "|" in line:
                metrics["f1_macro"] = float(line.split("|")[1].strip())
            elif "Span IoU F1" in line and "|" in line:
                metrics["iou_f1"] = float(line.split("|")[1].strip())
            elif "Token F1 (Positive Class)" in line and "|" in line:
                metrics["token_f1"] = float(line.split("|")[1].strip())
            elif "GMB-Subgroup AUC" in line and "|" in line:
                metrics["gmb_auc"] = float(line.split("|")[1].strip())
    except Exception as e:
        print(f"[ERR] Metric extraction failed for {log_path}: {e}")
        return None
        
    return metrics

def main():
    print("=" * 80)
    print("RELENTLESS PURSUIT OF SOTA: ALPHA IMPACT SWEEP (ENGLISH)")
    print("=" * 80)
    print(f"Sweep Range: {ALPHA_VALUES}")
    
    sweep_history = []
    
    for alpha in ALPHA_VALUES:
        print(f"\n[PHASE] Testing Alpha = {alpha}")
        output_dir = os.path.join(MODELS_ROOT, f"alpha_{alpha}")
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. TRAIN: Joint Optimization with variable rationale weight
        train_args = {
            "alpha": alpha,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "output_dir": output_dir,
            "data_path": DATA_PATH,
            "seed": SEED
        }
        
        start_time = time.time()
        ret = run_python_script(TRAIN_SCRIPT, train_args)
        duration = (time.time() - start_time) / 60
        
        if ret != 0:
            print(f"[ERR] Training failed for alpha={alpha}. Skipping.")
            continue
            
        print(f"[INFO] Alpha={alpha} training complete in {duration:.1f} mins.")
        
        # 2. EVAL: Extract performance and explainability chokepoints
        model_path = os.path.join(output_dir, "best_model.pth")
        eval_args = {
            "model_path": model_path,
            "only_stage1": "", # Flags are handled slightly differently, we add as key with empty val
            "batch_size": 32
        }
        
        # Capture generated log correctly
        eval_log_dir = os.path.join(PROJECT_ROOT, "experiments", "english", "proposed", "results", "only_stage1")
        before_logs = set(os.listdir(eval_log_dir)) if os.path.exists(eval_log_dir) else set()
        
        # Special handling for flags in our dict wrapper
        eval_cmd = f"conda activate {ENV_NAME} ; python {EVAL_SCRIPT} --model_path {model_path} --only_stage1 --batch_size 32"
        subprocess.call(["powershell", "-Command", eval_cmd])
        
        after_logs = set(os.listdir(eval_log_dir))
        new_logs = list(after_logs - before_logs)
        
        if new_logs:
            new_logs.sort(key=lambda x: os.path.getmtime(os.path.join(eval_log_dir, x)))
            latest_log = os.path.join(eval_log_dir, new_logs[-1])
            
            metrics = parse_scientific_metrics(latest_log)
            if metrics:
                metrics["alpha"] = alpha
                sweep_history.append(metrics)
                print(f"[RESULT] Alpha={alpha} -> F1: {metrics.get('f1_macro')}, IoU: {metrics.get('iou_f1')}")
            else:
                print(f"[ERR] Metrics missing in log: {latest_log}")
        else:
            print(f"[ERR] No evaluation log generated for alpha={alpha}")

    # 3. CONSOLIDATE: Export to CSV for visual excellence
    if sweep_history:
        # Sort by alpha to ensure clean plotting
        sweep_history.sort(key=lambda x: x['alpha'])
        
        fieldnames = ["alpha", "accuracy", "f1_macro", "iou_f1", "token_f1", "gmb_auc"]
        with open(FINAL_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in sweep_history:
                # Fill missing keys with 0.0
                clean_row = {k: row.get(k, 0.0) for k in fieldnames}
                writer.writerow(clean_row)
                
        print(f"\n[OK] Alpha Sweep Analysis Saved to: {FINAL_CSV}")
        print("Data is now ready for scientific visualization script.")
    else:
        print("\n[ERR] Sweep completed with 0 valid results. Check logs.")

if __name__ == "__main__":
    main()

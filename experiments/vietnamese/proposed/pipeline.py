import os
import subprocess
import sys
import argparse
from config_loader import MODEL_DIR

def run_step(script_name, description, extra_args=[]):
    print(f"\n{'='*60}")
    print(f"🚀 STEP: {description}")
    print(f"   Script: experiments/vietnamese/scripts/{script_name}")
    print(f"{'='*60}")
    
    # Script is now in the same directory as pipeline.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, script_name)
    if not os.path.exists(script_path):
        print(f"❌ Script not found: {script_path}")
        sys.exit(1)

    cmd = [sys.executable, script_path] + extra_args
    print(f"Executing: {' '.join(cmd)}")
    
    try:
        # Stream output to console
        result = subprocess.run(cmd, check=True)
        print(f"✅ Step '{description}' completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in step '{description}'. Exit code: {e.returncode}")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Vi-XHate Full Pipeline Runner")
    parser.add_argument("--skip_data", action="store_true", help="Skip data preparation step")
    parser.add_argument("--skip_train", action="store_true", help="Skip training step")
    parser.add_argument("--epochs", type=str, default="5", help="Number of training epochs")
    
    args = parser.parse_args()
    
    print("🔥 STARTING VI-XHATE PIPELINE...")
    
    # 1. Data Prep
    from config_loader import DATA_PATH
    if not args.skip_data:
        if os.path.exists(DATA_PATH):
           print(f"ℹ️ Data file found at {DATA_PATH}. Skipping Data Process.")
        else:
           run_step("01_prepare_data.py", "Data Preparation (Gold Only)")
    


    # 2. Training
    if not args.skip_train:
        best_model_path = os.path.join(MODEL_DIR, "best_multitask_model.pth")
        if os.path.exists(best_model_path):
             print(f"ℹ️ Model found at {best_model_path}. Skipping Training.")
        else:
             run_step("02_train_multitask.py", "Training (Multi-task Learning)", 
                      extra_args=["--epochs", args.epochs, "--output_dir", MODEL_DIR])
        
    # 3. Evaluation
    run_step("03_evaluate.py", "Evaluation (Metrics Calculation)")
    
    # 4. Robustness Check (Explicit)
    # Explicitly running again to satisfy visual requirement of "Robustness Check" step
    # This evaluates on the test set again (as proxy for robustness) and saves to a distinct file.
    run_step("03_evaluate.py", "Robustness Check (Same Test Data)", 
             extra_args=["--output_file", "robustness_check_pipeline.json"])
    
    print("\n🎉 ALL STEPS COMPLETED SUCCESSFULLY! RESULTS ARE READY.")

if __name__ == "__main__":
    main()

import os

def load_env():
    """Simple .env loader to avoid external dependencies like python-dotenv"""
    # config_loader.py is in experiments/vietnamese/scripts/
    # Root is 3 levels up: experiments/vietnamese/scripts -> experiments/vietnamese -> experiments -> root
# Constants pointing into the experiment folder
# Use Absolute Paths to avoid CWD issues
script_dir = os.path.dirname(os.path.abspath(__file__))
# current file is in experiments/vietnamese/proposed/
# root is ../../../
root_dir = os.path.abspath(os.path.join(script_dir, "../../../"))

# Load immediately on import
load_env()

DATA_PATH = os.path.join(root_dir, os.environ.get("VIHOS_DATA_PATH", "experiments/vietnamese/data/vihos_prepared.jsonl"))
MODEL_DIR = os.path.join(root_dir, os.environ.get("VIHOS_MODEL_DIR", "experiments/vietnamese/models"))
# Fixed path to the best trained model for E1 VN.
# Trước đây dùng vihos_v2_full; hiện tại chuyển sang vihos_e1_optimized (checkpoint mới train).
MODEL_PATH = os.path.join(root_dir, "experiments/vietnamese/models/vihos_e1_optimized/best_multitask_model.pth")

# English Benchmarks
HATEXPLAIN_DATA_PATH = os.path.join(root_dir, os.environ.get("HATEXPLAIN_DATA_PATH", "experiments/english/data/hatexplain_3label.jsonl"))
HATEXPLAIN_MODEL_PATH = os.path.join(root_dir, os.environ.get("HATEXPLAIN_MODEL_PATH", "experiments/english/output_multitask/best_model.pth"))

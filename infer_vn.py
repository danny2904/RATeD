import os
import subprocess
import sys

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def run_command(cmd):
    print("\n" + "="*70)
    print(f"RUNNING: {' '.join(cmd)}")
    print("="*70 + "\n")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERR] Script execution failed with exit code {e.returncode}")
    except KeyboardInterrupt:
        print("\n[INFO] Inference interrupted by user.")

def main():
    while True:
        clear_screen()
        print("          RATeD-V VIETNAMESE - RESEARCH PIPELINE")
        print("="*60)
        print("1. Cascaded Pipeline (Backbone + Judge)")
        print("2. Stage 1 Only (Backbone/RATeD-V Only)")
        print("3. Stage 2 Only (Judge Only Ablation)")
        print("4. Exit")
        print("-" * 60)
        
        choice = input("Select Mode (1-4): ")
        if choice == '4': break
        if choice not in ['1', '2', '3']: continue

        # Configuration
        provider = "local"
        if choice in ['1', '3']:
            print("\nJudge Provider:")
            print("  1. Qwen-2.5-7B (Local Specialist)")
            print("  2. Gemini 2.5 Flash Lite (API)")
            p_choice = input("Choice (1-2): ")
            if p_choice == '2': provider = "gemini"

        limit = input("\nLimit samples? (Enter for FULL, or number): ")
        
        # Build Command
        script = "experiments/vietnamese/proposed/04_run_proposed_cascaded_verify.py"
        # ViHOS specialist judge model path
        hf_model = "experiments/vietnamese/models/qwen2.5-7b-vihos-specialist"
        
        cmd = [sys.executable, script, "--provider", provider, "--hf_model", hf_model]
        
        if choice == '2': cmd.append("--only_stage1")
        elif choice == '3': cmd.append("--only_stage2")
        
        if limit.isdigit():
            cmd.extend(["--limit", limit])
            
        run_command(cmd)
        input("\nPress Enter to return to menu...")

if __name__ == "__main__":
    main()

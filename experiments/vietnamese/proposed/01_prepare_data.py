import json
import os
import sys
from config_loader import DATA_PATH

def main():
    # Define paths
    source_dir = os.path.join("experiments", "vietnamese", "data", "dataset_vihatexplain_raw")
    output_path = DATA_PATH
    
    # Original ViHOS filenames
    splits = ['train', 'validation', 'test']
    
    # Check source directory
    if not os.path.exists(source_dir):
        # Fallback: Check if they are still in root data (migration safety)
        if os.path.exists(os.path.join("data", "train.jsonl")):
            source_dir = "data"
        else:
            print(f"❌ Error: Source directory '{source_dir}' not found. Please ensure raw jsonl files are present.")
            sys.exit(1)

    all_data = []
    print(f"🚀 Preparing ViHOS Data from: {source_dir}")

    for filename in splits:
        file_path = os.path.join(source_dir, f"{filename}.jsonl")
        
        if not os.path.exists(file_path):
            print(f"⚠️ Warning: File not found: {file_path}")
            continue
            
        # Map filename to standard split name (validation -> val)
        split_label = 'val' if filename == 'validation' else filename
        
        print(f"   Processing {filename} -> split='{split_label}'...")
        
        count = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                
                try:
                    item = json.loads(line)
                    
                    # 1. Enforce Split Label
                    item['split'] = split_label
                    
                    # 2. Cleanup Unused Fields (Legacy Silver/Rewrites)
                    item.pop('rewrites', None)
                    item.pop('llm_rationale', None)
                    
                    all_data.append(item)
                    count += 1
                except json.JSONDecodeError:
                    print(f"   ❌ Error parsing JSON line in {filename}")

        print(f"   -> Mobile {count} items.")

    # Sort checks? No need, usually preserve order.
    
    # Write output
    print(f"\n💾 Saving {len(all_data)} prepared samples to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print("✅ Data Preparation Completed!")

if __name__ == "__main__":
    main()

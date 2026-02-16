import json
import os

def prepare_sft_en(input_file, output_file):
    """
    Standardized SFT Data Preparation for English Specialist Judge (Qwen).
    Aligned with Vietnamese gold standard (05_prepare_sft_data.py).
    """
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]
    
    sft_data = []
    for s in samples:
        # HateXplain format: 'comment' and 'spans' (list of strings)
        text = s.get("comment", "")
        spans = s.get("spans", [])
        
        # Format spans for output: comma-separated or SAFE
        # Align with Vietnamese output format
        output_text = ", ".join(spans) if spans else "SAFE"
        
        # Specialist Expert Persona (English version)
        system_prompt = "You are a specialist in analyzing social media hate speech and toxic content. Extract all toxic, offensive, or hateful words/phrases from the given text exactly as they appear. If the text is safe, return only 'SAFE'."
        
        entry = {
            "instruction": system_prompt,
            "input": text,
            "output": output_text
        }
        sft_data.append(entry)
    
    # Save as JSON (Unsloth/SFTTrainer friendly)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)
    
    print(f"Prepared {len(sft_data)} English SFT samples -> {output_file}")

if __name__ == "__main__":
    # Standard locations
    train_in = "experiments/english/data/hatexplain_prepared.jsonl"
    sft_out = "experiments/english/data/hatexplain_sft_train.json"
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(sft_out), exist_ok=True)
    
    prepare_sft_en(train_in, sft_out)

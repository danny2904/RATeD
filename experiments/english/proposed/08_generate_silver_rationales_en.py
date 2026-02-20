from unsloth import FastLanguageModel
import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

# Configuration
PROJECT_ROOT = r"c:\Projects\RATeD-V"
DATA_PATH = os.path.join(PROJECT_ROOT, "experiments", "english", "data", "hatexplain_prepared.jsonl")
MODEL_PATH = os.path.join(PROJECT_ROOT, "experiments", "english", "models", "qwen2.5-7b-hatexplain-specialist-3class")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "experiments", "english", "data", "hatexplain_silver_rationales.jsonl")

def generate_silver():
    print("🚀 Initializing Silver Rationale Generation (LLM Distillation - TURBO)...")
    
    # Load Model (Unsloth for speed)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=512,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    
    # Ensure left padding for batch generation
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Data
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        all_data = [json.loads(line) for line in f]
    
    train_data = [item for item in all_data if item.get("split") == "train"]
    print(f"Total Train Samples: {len(train_data)}")

    # OPTIMIZED Prompt: Short and sweet to minimize output tokens
    PROMPT_STYLE = "Comment: {text}\nClassify and find toxic spans.\nOutput: LABEL: [label], SPANS: [list]"

    # Open file for incremental writing
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f_out:
        batch_size = 16 # Reduced from 32 to fit in VRAM comfortably
        for i in tqdm(range(0, len(train_data), batch_size), desc="Distilling Rationales (Turbo mode)"):
            batch = train_data[i : i + batch_size]
            texts = [PROMPT_STYLE.format(text=item["comment"]) for item in batch]
            
            inputs = tokenizer(texts, return_tensors="pt", padding=True).to("cuda")
            outputs = model.generate(
                **inputs, 
                max_new_tokens=64,
                use_cache=True,
                temperature=0.01,
                pad_token_id=tokenizer.pad_token_id
            )
            
            input_len = inputs.input_ids.shape[1]
            decoded = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
            
            for item, full_out in zip(batch, decoded):
                silver_spans = []
                try:
                    if "SPANS:" in full_out:
                        span_part = full_out.split("SPANS:")[1].strip()
                        if "[" in span_part and "]" in span_part:
                            list_str = "[" + span_part.split("[")[1].split("]")[0] + "]"
                            silver_spans = eval(list_str)
                except:
                    pass

                res = {
                    "id": item.get("id"),
                    "silver_spans": silver_spans
                }
                f_out.write(json.dumps(res, ensure_ascii=False) + "\n")
            
            f_out.flush() # Ensure it's written disk
    
    print(f"✅ Silver rationales incremental save completed: {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_silver()

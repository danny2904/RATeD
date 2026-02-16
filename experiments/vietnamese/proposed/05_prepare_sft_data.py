import json
import os

def prepare_sft(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]
    
    sft_data = []
    for s in samples:
        text = s["comment"]
        spans = s.get("spans", [])
        
        # Format spans for output
        output_text = ", ".join(spans) if spans else "SAFE"
        
        # System prompt for consistency
        system_prompt = "Bạn là chuyên gia phân tích ngôn từ thù ghét trên mạng xã hội Việt Nam. Hãy trích xuất chính xác tất cả các từ hoặc cụm từ độc hại, xúc phạm hoặc thù ghét trong văn bản. Nếu văn bản an toàn, chỉ trả về 'SAFE'."
        
        entry = {
            "instruction": system_prompt,
            "input": text,
            "output": output_text
        }
        sft_data.append(entry)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)
    
    print(f"Prepared {len(sft_data)} samples -> {output_file}")

if __name__ == "__main__":
    train_in = "experiments/vietnamese/data/vihos_prepared.jsonl"
    prepare_sft(train_in, "experiments/vietnamese/data/vihos_sft_train.json")
    
    # Also prepare the full data for maximum coverage if needed
    # But for now, we follow split convention

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

model_path = "experiments/vietnamese/models/qwen2.5-7b-vihos-specialist"
print(f"Loading model from: {model_path}")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    print("Model loaded successfully!")
    
    prompt = "Cái thứ súc vật này."
    inputs = tokenizer(f"Câu sau có phải là ngôn ngữ thù ghét không? Trả lời 'SAFE' hoặc 'TOXIC'.\nCâu: {prompt}\nKết quả:", return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=20)
    print("Response:", tokenizer.decode(outputs[0], skip_special_tokens=True))
    
except Exception as e:
    print(f"Error: {e}")

import os
import sys
import torch
import json
import numpy as np
import argparse
import time
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from transformers import AutoTokenizer
import openai
import google.generativeai as genai
from torch.utils.data import DataLoader, Dataset
from dotenv import load_dotenv

# Load env variables from .env file if present
load_dotenv()

# Add path to import local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
proposed_dir = os.path.abspath(os.path.join(current_dir, "../proposed"))
if proposed_dir not in sys.path: sys.path.append(proposed_dir)
from config_loader import DATA_PATH

# --- IMPORT COMMON METRICS & REPORTING ---
common_dir = os.path.abspath(os.path.join(current_dir, "../common"))
if common_dir not in sys.path: sys.path.append(common_dir)

from metrics import (
    vihatet5_standardized_span_metrics, 
    get_char_indices_from_bio_tags,
    get_char_indices_from_spans,
    calculate_syllable_metrics,
    calculate_token_metrics_full
)
from reporting import generate_report_string
import ast

# --- METRIC FUNCTIONS REMOVED (Now imported from common/metrics.py) ---

# --- METRIC FUNCTIONS REMOVED (Now imported from common/metrics.py) ---

class InferenceDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=256, split='test', limit=None):
        self.data = []
        if data_path.endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(data_path)
            if 'source' in df.columns:
                df = df[df['source'].str.contains("hate-spans-detection")].copy()
            self.data = df.to_dict('records')
        else:
            try:
                with open(data_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        item = json.loads(line)
                        if split is None or item.get('split') == split:
                            self.data.append(item)
            except FileNotFoundError:
                print(f"File not found: {data_path}")
        
        if limit:
            self.data = self.data[:limit]
            
        self.tokenizer = tokenizer
        self.max_len = max_len
        print(f"Loaded {len(self.data)} samples for split '{split}'")
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        text = str(item.get('comment', ''))
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'text': text
        }

# --- OpenAI Judge Class ---
class OpenAIJudge:
    def __init__(self, api_key=None, model_name="gpt-4o-mini", max_workers=10):
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")
        self.max_workers = max_workers
        self.model_name = model_name
        if not api_key:
            print("WARNING: No OPENAI_API_KEY found. OpenAI Judge will fail.")
            self.client = None
        else:
            self.client = openai.OpenAI(api_key=api_key)
            print(f"OpenAI Judge ({model_name}) initialized (parallel workers={max_workers}).")

    def _call_one(self, text):
        """Single call to OpenAI for one text."""
        if not self.client:
            return "SAFE"
        
        prompt = f"""
        Nhiệm vụ: Bạn là một chuyên gia phân tích độc hại (Toxic Detection).
        Hãy xác định xem văn bản sau có chứa ngôn từ thù ghét, xúc phạm, tục tĩu hoặc quấy rối không.

        Văn bản cần kiểm tra: "{text}"

        Yêu cầu:
        1. Nếu văn bản HOÀN TOÀN sạch: Trả về "SAFE".
        2. Nếu chứa nội dung độc hại (dù là ẩn ý, tiếng lóng, hay châm biếm):
           Hãy trích xuất CHÍNH XÁC (vẫn giữ nguyên teencode) các từ/cụm từ độc hại đó.
           Nếu có nhiều cụm từ, hãy liệt kê chúng cách nhau bởi dấu phẩy.

        Trả về "SAFE" hoặc các cụm từ độc hại. Không giải thích gì thêm.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            return response.choices[0].message.content.strip() or "SAFE"
        except openai.AuthenticationError:
            print(f"CRITICAL: OpenAI Authentication Error (Invalid API Key). Check your .env file.", flush=True)
            return "AUTH_ERROR"
        except Exception as e:
            print(f"OpenAI Error: {e}", flush=True)
            return "SAFE"

    def verify_batch(self, texts):
        """Call OpenAI parallel for multiple texts."""
        if not texts:
            return []
        if not self.client:
            return ["SAFE"] * len(texts)
        n = len(texts)
        res = [None] * n
        workers = min(self.max_workers, n)
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self._call_one, t): i for i, t in enumerate(texts)}
            for fut in as_completed(futures):
                idx = futures[fut]
                raw = (fut.result() or "").strip()
                if raw == "AUTH_ERROR":
                    print("🛑 Stopping batch due to Authentication Error.")
                    # Fill the rest with ERROR and exit this loop if possible, or just mark
                    res[idx] = "ERROR: AUTH_FAILED"
                elif "SAFE" in raw.upper() or "KHÔNG" in raw.upper():
                    res[idx] = "SAFE"
                else:
                    res[idx] = raw.strip(' ".\'')
        return res

# --- Gemini Judge Class ---
class GeminiJudge:
    def __init__(self, api_key=None, model_name="gemini-2.5-flash-lite", max_workers=5):
        if not api_key:
            api_key = os.environ.get("GOOGLE_API_KEY")
        self.max_workers = max_workers
        if not api_key:
            print("WARNING: No GOOGLE_API_KEY found. Gemini Judge will fail.")
            self.model = None
        else:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
            print(f"Gemini Judge ({model_name}) initialized (parallel workers={max_workers}).")

    def _call_one(self, text):
        """Single call to Gemini for one text."""
        if not self.model:
            return "SAFE"
        
        prompt = f"""
        Nhiệm vụ: Bạn là một chuyên gia phân tích độc hại (Toxic Detection).
        Hãy xác định xem văn bản sau có chứa ngôn từ thù ghét, xúc phạm, tục tĩu hoặc quấy rối không.

        Văn bản cần kiểm tra: "{text}"

        Yêu cầu:
        1. Nếu văn bản HOÀN TOÀN sạch: Trả về "SAFE".
        2. Nếu chứa nội dung độc hại (dù là ẩn ý, tiếng lóng, hay châm biếm):
           Hãy trích xuất CHÍNH XÁC (vẫn giữ nguyên teencode) các từ/cụm từ độc hại đó.
           Nếu có nhiều cụm từ, hãy liệt kê chúng cách nhau bởi dấu phẩy.

        Trả về "SAFE" hoặc các cụm từ độc hại. Không giải thích gì thêm.
        """
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(temperature=0.0)
            )
            return (response.text or "").strip()
        except Exception as e:
            print(f"Gemini Error: {e}", flush=True)
            return "SAFE"

    def verify_batch(self, texts):
        """Call Gemini parallel for multiple texts."""
        if not texts:
            return []
        if not self.model:
            return ["SAFE"] * len(texts)
        n = len(texts)
        res = [None] * n
        workers = min(self.max_workers, n)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self._call_one, t): i for i, t in enumerate(texts)}
            for fut in as_completed(futures):
                idx = futures[fut]
                raw = (fut.result() or "").strip()
                if "SAFE" in raw.upper() or "KHÔNG" in raw.upper():
                    res[idx] = "SAFE"
                else:
                    res[idx] = raw.strip(' ".\'')
        return res

# --- Local Judge Class (Transformers) ---
class LocalJudge:
    def __init__(self, model_repo="Qwen/Qwen2.5-7B-Instruct", load_in_4bit=True):
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
        print(f"Loading local model: {model_repo} (Offline Inference UI)...")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA detected. Available GPUs: {torch.cuda.device_count()}")
        else:
            print("❌ CUDA NOT DETECTED! Local inference will be extremely slow on CPU.")

        is_local_path = os.path.exists(model_repo)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_repo, 
            trust_remote_code=True,
            local_files_only=is_local_path
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
            
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        ) if load_in_4bit else None
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_repo, 
            device_map="auto",
            quantization_config=bnb_config,
            local_files_only=is_local_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 
        )
        
        print(f"✅ Model distributed across: {self.model.hf_device_map}")
        print(f"Local Judge initialized successfully.")

    def verify_batch(self, texts):
        if not texts: return []
        
        instruction = "Bạn là chuyên gia phân tích ngôn từ thù ghét trên mạng xã hội Việt Nam. Hãy trích xuất chính xác tất cả các từ hoặc cụm từ độc hại, xúc phạm hoặc thù ghét trong văn bản. Nếu văn bản an toàn, chỉ trả về 'SAFE'."
        
        formatted_inputs = []
        for text in texts:
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": text}
            ]
            text_input = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            formatted_inputs.append(text_input)
            
        try:
            inputs = self.tokenizer(formatted_inputs, return_tensors="pt", padding=True).to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            input_len = inputs.input_ids.shape[1]
            generated_tokens = outputs[:, input_len:]
            answers = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            
            results = []
            for answer in answers:
                ans = answer.strip()
                results.append("SAFE" if "SAFE" in ans.upper() else ans)
            return results
            
        except Exception as e:
            print(f"Local Model Batch Error: {e}")
            return ["SAFE"] * len(texts)

    def verify_span(self, text, span):
        if not self.model:
            return span
        raw = self._call_one(text)
        if "SAFE" in raw.upper() or "KHÔNG" in raw.upper():
            return None
        refined_span = raw.strip(' ".\'')
        if refined_span and refined_span not in text:
            return span
        return refined_span if refined_span else span

# --- Removed Redundant Classes (HFJudge, LocalJudge, CascadedPipeline) ---
                    
def main():
    parser = argparse.ArgumentParser(description="Vietnamese Hate Speech Evaluation Script for Multiple LLMs (Zero-Shot Baseline).")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for testing (default: all).")
    parser.add_argument("--output", type=str, default=None, help="Output JSON filename (default: auto-generated in results/).")
    parser.add_argument("--provider", type=str, default="openai", choices=["openai", "gemini", "local"], 
                        help="LLM provider: 'openai' (GPT-4o-mini), 'gemini' (Gemini 2.5 Flash Lite), or 'local' (Qwen 2.5-7B via Transformers).")
    parser.add_argument("--model", type=str, default=None, help="Override default model name (e.g., 'gpt-4o', 'gemini-1.5-flash', or local path).")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size or number of parallel workers (default: 10).")
    parser.add_argument("--load_4bit", action="store_true", default=True, help="Load local model in 4-bit quantization (default: True).")
    args = parser.parse_args()
    
    # Defaults and Initialization
    if args.provider == "openai":
        model_name = args.model or "gpt-4o-mini"
        judge = OpenAIJudge(model_name=model_name, max_workers=args.batch_size)
    elif args.provider == "gemini":
        model_name = args.model or "gemini-2.5-flash-lite"
        judge = GeminiJudge(model_name=model_name, max_workers=args.batch_size)
    elif args.provider == "local":
        model_name = args.model or "Qwen/Qwen2.5-7B-Instruct"
        judge = LocalJudge(model_repo=model_name, load_in_4bit=args.load_4bit)
    else:
        raise ValueError(f"Unknown provider: {args.provider}")
    
    # Load Data (Generic XLM-R Tokenizer for metric calculations that need it)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    
    dataset = InferenceDataset(DATA_PATH, tokenizer, split='test', limit=args.limit)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    results = []
    start_time = time.time()
    
    # --- LOGGER SETUP ---
    class DualLogger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "w", encoding='utf-8')
            
        def print(self, message):
            self.terminal.write(message + "\n")
            self.log.write(message + "\n")
            self.terminal.flush()
            self.log.flush()

        def close(self):
            self.log.close()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    provider_tag = args.provider
    model_tag = re.sub(r'[^a-zA-Z0-9]', '_', model_name.split('/')[-1])
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Create model-specific subdirectory inside results/
    model_results_dir = os.path.join(script_dir, "results", model_tag)
    os.makedirs(model_results_dir, exist_ok=True)

    log_name = f"baseline_{provider_tag}_{model_tag}_vn_{timestamp}.log"
    log_filename = os.path.join(model_results_dir, log_name)
    
    logger = DualLogger(log_filename)
    logger.print(f"[Provider: {args.provider}]")
    logger.print(f"[Model: {model_name}]")
    logger.print(f"[Log file: {log_name}]")

    # Output JSON path initialization
    custom_output = args.output
    if not custom_output:
        custom_output = f"results_{provider_tag}_{model_tag}_{timestamp}.json"
    
    # If not absolute, put it inside the model-specific results folder
    if os.path.isabs(custom_output):
        out_path = custom_output
    else:
        out_path = os.path.join(model_results_dir, custom_output)

    print(f"Logging analysis to: {log_filename}")

    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        texts = batch['text']
        
        # Call OpenAI parallel
        judge_results = judge.verify_batch(texts)
        
        for i, judge_out in enumerate(judge_results):
            global_idx = batch_idx * args.batch_size + i
            raw_item = dataset.data[global_idx]
            
            is_error = "ERROR" in judge_out
            is_toxic = judge_out != "SAFE" and not is_error
            pred_spans = []
            pred_indices = []
            
            if is_toxic:
                pred_spans = [s.strip() for s in judge_out.split(",") if s.strip()]
                for s in pred_spans:
                    start = 0
                    while True:
                        found_idx = texts[i].find(s, start)
                        if found_idx == -1: break
                        for char_idx in range(found_idx, found_idx + len(s)):
                            if char_idx not in pred_indices: pred_indices.append(char_idx)
                        start = found_idx + 1
            elif is_error:
                pred_spans = [judge_out]
            
            results.append({
                "text": texts[i],
                "true_label": raw_item.get('label', 'N/A'),
                "pred_label": "error" if is_error else ("hate" if is_toxic else "clean"),
                "pred_spans": pred_spans,
                "pred_indices": pred_indices,
                "gold_spans": raw_item.get('spans', []),
                "raw_item": raw_item
            })

    total_time = time.time() - start_time
    
    # Stats Summary
    logger.print("\n" + "="*40)
    logger.print(f"{args.provider.upper()} EVALUATION RESULTS")
    logger.print("="*40)
    logger.print(f"Total Processed: {len(results)}")
    logger.print(f"Time Taken: {total_time:.2f}s ({total_time/len(results):.2f}s/sample)")
    
    # --- CALCULATE METRICS VS GROUND TRUTH ---
    from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
    
    
    # Metric Arrays
    y_true = []
    y_pred_e1 = []
    y_pred_e11_cls = []
    # Span Metric Aggregators
    all_syl_acc = []
    all_syl_f1 = []
    all_token_acc = []
    all_token_p = []
    all_token_r = []
    all_token_mf1 = []
    all_span_mf1 = []
    all_span_wf1 = []
    all_span_acc = []
    all_strict_f1 = []

    valid_entries = 0
    for r in results:
        # Normalize labels for ViHOS (safe/unsafe) or generic toxic labels
        # 1 = Toxic/Unsafe, 0 = Normal/Safe
        def normalize_lbl(l):
            l_str = str(l).strip().lower()
            if any(hit in l_str for hit in ["unsafe", "hate", "toxic", "offensive", "1"]):
                return 1
            return 0
            
        t_val = normalize_lbl(r['true_label'])
        p_val_e11 = normalize_lbl(r['pred_label'])
        
        y_true.append(t_val)
        y_pred_e11_cls.append(p_val_e11)
        
        # --- COLLECT METRICS ---
        # 1. Syllable
        s_acc, s_f1_bin, s_f1_macro = calculate_syllable_metrics(r['pred_spans'], r.get('gold_spans', []), r['text'])
        all_syl_acc.append(s_acc)
        all_syl_f1.append(s_f1_macro) # We use Macro F1 for consistency
        
        # 2. Token
        tok_m = calculate_token_metrics_full(r['pred_spans'], r.get('gold_spans', []), r['text'], tokenizer)
        if tok_m:
            all_token_acc.append(tok_m['Token Accuracy'])
            all_token_p.append(tok_m['Token Precision'])
            all_token_r.append(tok_m['Token Recall'])
            all_token_mf1.append(tok_m['Token mF1'])
            
        # 3. Standardized Metric (ViHateT5 Style)
        # Use absolute character indices for precision
        text = r['text']
        
        # FIX: Prioritize unsafe_spans_indices which is standard in our jsonl
        raw_item = r.get('raw_item', {})
        if 'unsafe_spans_indices' in raw_item:
            gold_indices = get_char_indices_from_spans(raw_item['unsafe_spans_indices'])
        elif 'original_spans' in raw_item:
            try:
                gold_indices = set(ast.literal_eval(raw_item['original_spans']))
            except:
                gold_indices = set()
        else:
            gold_indices = set()
        
        # Mapping predicted indices
        pred_indices = set(r.get('pred_indices', []))
        
        c_acc, c_mf1, c_wf1, c_f1_bin = vihatet5_standardized_span_metrics(pred_indices, gold_indices, text)
        all_span_acc.append(c_acc)
        all_span_mf1.append(c_mf1)
        all_span_wf1.append(c_wf1)
        all_strict_f1.append(c_f1_bin)
        
        valid_entries += 1
        
    if valid_entries > 0:
        # --- AGGREGATE METRICS ---
        cls_metrics = {
            "Accuracy": accuracy_score(y_true, y_pred_e11_cls),
            "F1-Macro": f1_score(y_true, y_pred_e11_cls, average='macro', zero_division=0),
            "Precision": precision_score(y_true, y_pred_e11_cls, average='macro', zero_division=0),
            "Recall": recall_score(y_true, y_pred_e11_cls, average='macro', zero_division=0)
        }
        
        # Only report the 3 metrics from Table 3 (Hate Spans Detection)
        span_metrics = {
            "Acc": np.mean(all_span_acc) if all_span_acc else 0.0,
            "WF1": np.mean(all_span_wf1) if all_span_wf1 else 0.0,
            "MF1": np.mean(all_span_mf1) if all_span_mf1 else 0.0
        }
        
        cm = confusion_matrix(y_true, y_pred_e11_cls)
        
        # --- GENERATE STANDARDIZED REPORT ---
        report_name = f"{model_name} Zero-Shot Baseline (VN)"
        report_str = generate_report_string(report_name, cls_metrics, span_metrics, cm)
        
        logger.print("\n" + report_str)
        
        # Also print time stats separately as they are pipeline specific
        logger.print(f"\nTime Stats: {total_time:.2f}s total | {total_time/len(results):.2f}s/avg")

    
    # Save results
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.print(f"Results saved to {out_path}")
    logger.close()

if __name__ == "__main__":
    main()

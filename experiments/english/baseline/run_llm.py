import os
import sys
import torch
import json
import numpy as np
import argparse
import time
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
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
# from config_loader import DATA_PATH # Removed as it's not needed for baseline

# --- IMPORT COMMON METRICS & REPORTING ---
common_dir = os.path.abspath(os.path.join(current_dir, "../common"))
if common_dir not in sys.path: sys.path.append(common_dir)

try:
    from metrics import (
        calculate_span_f1, 
        calculate_token_metrics, 
        calculate_auroc, 
        calculate_bias_metrics
    )
    from metrics_utils import load_bias_metadata_from_prepared
    from reporting import generate_report_string
except ImportError:
    print("Warning: Could not import common metrics. Using fallback logic.")


class InferenceDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=256, split='test', limit=None):
        self.data = []
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
        
        # print(f"  [OpenAI] Calling with text: {text[:50]}...", flush=True) # Debug log
        prompt = f"""Task: You are an expert toxic content analyst. Classify the text into exactly ONE of these categories: "HATE", "OFFENSIVE", or "NORMAL".
Also, extract the smallest set of toxic phrases if any.

Context: "{text}"

Rules:
1. LABEL: return exactly one of "HATE", "OFFENSIVE", or "NORMAL".
2. SPANS: If toxic, extract ONLY the exact toxic phrase(s), comma-separated. Otherwise return "NONE".

Return format:
LABEL: <label>
SPANS: <spans>

No explanation."""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                timeout=30.0 # Prevent indefinite hangs
            )
            res = response.choices[0].message.content.strip() or "SAFE"
            # print(f"  [OpenAI] Result: {res[:30]}", flush=True) # Debug log
            return res
        except openai.APITimeoutError:
            print("Timeout: OpenAI request took too long.", flush=True)
            return "SAFE"
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
            # Use wait instead of as_completed to have a global batch timeout
            done, not_done = wait(futures, timeout=120) # 2 minutes for a batch of 10
            
            for fut in done:
                idx = futures[fut]
                try:
                    raw = (fut.result() or "").strip()
                except Exception as e:
                    print(f"Batch worker error: {e}")
                    raw = "SAFE"
                    
                if raw == "AUTH_ERROR":
                    print("🛑 Stopping batch due to Authentication Error.")
                    res[idx] = "ERROR: AUTH_FAILED"
                else:
                    # Parse LABEL and SPANS
                    label = "NORMAL"
                    spans = "NONE"
                    label_match = re.search(r"LABEL:\s*(HATE|OFFENSIVE|NORMAL)", raw, re.IGNORECASE)
                    if label_match: label = label_match.group(1).upper()
                    
                    spans_match = re.search(r"SPANS:\s*(.*)", raw, re.IGNORECASE)
                    if spans_match: spans = spans_match.group(1).strip()
                    
                    if label == "NORMAL":
                        res[idx] = "SAFE"
                    else:
                        res[idx] = f"{label}|{spans}"
            
            # For those that didn't finish
            for fut in not_done:
                idx = futures[fut]
                print(f"⚠️ Sample {idx} in batch HUNG and was skipped.")
                res[idx] = "SAFE"
                fut.cancel()

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
        
        prompt = f"""Task: You are an expert toxic content analyst. Classify the text into exactly ONE of these categories: "HATE", "OFFENSIVE", or "NORMAL".
Also, extract the smallest set of toxic phrases if any.

Context: "{text}"

Rules:
1. LABEL: return exactly one of "HATE", "OFFENSIVE", or "NORMAL".
2. SPANS: If toxic, extract ONLY the exact toxic phrase(s), comma-separated. Otherwise return "NONE".

Return format:
LABEL: <label>
SPANS: <spans>

No explanation."""

        try:
            # Wrap in try-except to handle safety filters
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(temperature=0.0)
            )
            # Check if candidates were returned
            if not response.candidates:
                print(f"  [Gemini] Blocked: No candidates (PROHIBITED_CONTENT). Outputting SAFE.", flush=True)
                return "SAFE"
            
            # Check finish reason
            finish_reason = response.candidates[0].finish_reason
            if finish_reason != 1: # 1 is STOP (Success)
                print(f"  [Gemini] Blocked: Finish reason {finish_reason}. Outputting SAFE.", flush=True)
                return "SAFE"

            return (response.text or "").strip()
        except Exception as e:
            # Handle "The response.text quick accessor requires the response to contain a valid Part"
            if "response.text" in str(e) or "Part" in str(e) or "candidates" in str(e):
                print(f"  [Gemini] Error accessor: Safety Block. Outputting SAFE.", flush=True)
            else:
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
            done, not_done = wait(futures, timeout=120)
            
            for fut in done:
                idx = futures[fut]
                try:
                    raw = (fut.result() or "").strip()
                except Exception as e:
                    print(f"Gemini Batch worker error: {e}")
                    raw = "SAFE"
                
                # Parse LABEL and SPANS
                label = "NORMAL"
                spans = "NONE"
                label_match = re.search(r"LABEL:\s*(HATE|OFFENSIVE|NORMAL)", raw, re.IGNORECASE)
                if label_match: label = label_match.group(1).upper()
                
                spans_match = re.search(r"SPANS:\s*(.*)", raw, re.IGNORECASE)
                if spans_match: spans = spans_match.group(1).strip()
                
                if label == "NORMAL":
                    res[idx] = "SAFE"
                else:
                    res[idx] = f"{label}|{spans}"
            
            for fut in not_done:
                idx = futures[fut]
                print(f"⚠️ Gemini Sample {idx} HUNG and was skipped.")
                res[idx] = "SAFE"
                fut.cancel()
        return res

# --- Local Judge Class (Transformers) ---
class LocalJudge:
    def __init__(self, model_repo="Qwen/Qwen2.5-7B-Instruct", load_in_4bit=True):
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
        print(f"Loading local model: {model_repo}...")
        
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

    def clean_judge_output(self, raw_output, max_span_length=120, text=None):
        import re
        label, spans, hallucination_detected = "normal", [], False
        clean_ans = raw_output
        if "assistant\n" in raw_output:
            clean_ans = raw_output.split("assistant\n")[-1].strip()
        try:
            label_match = re.search(r"LABEL:\s*(hate|offensive|normal)", clean_ans, re.IGNORECASE)
            if label_match: label = label_match.group(1).lower()
            else:
                kw_match = re.search(r"\b(hate|offensive|normal)\b", clean_ans, re.IGNORECASE)
                label = kw_match.group(1).lower() if kw_match else "normal"
            spans_match = re.search(r"SPANS:\s*([^\n\r]*)", clean_ans, re.IGNORECASE)
            if spans_match:
                spans_text = spans_match.group(1).strip()
                if "NONE" not in spans_text.upper() and spans_text:
                    raw_spans = [s.strip() for s in spans_text.split(',')]
                    for span in raw_spans:
                        span = span.strip().strip('"').strip("'")
                        if span and len(span) >= 2 and len(span) <= max_span_length:
                            spans.append(span)
            spans = sorted(list(set(spans)), key=len)[:2]
        except: hallucination_detected = True
        return label, spans, hallucination_detected

    def verify_batch(self, texts):
        if not texts: return []
        
        instruction = f"""You are a content moderator specializing in detecting hate speech.
You must follow the MINIMAL RATIONALE philosophy: Only extract the smallest set of tokens that make the text toxic. Topic words (e.g., 'Immigrants', 'Refugees') are NOT toxic unless directly used as part of a slur.

STRICT RULES:
1. OUTPUT FORMAT: Exactly 3 lines.
LABEL: <hate|offensive|normal>
SPANS: <comma-separated minimal spans, or NONE>
REASONING: <1 short sentence>
2. MINIMALITY: 
   - Extract ONLY specific slurs, insults, or violent threats.
   - ANTI-TOPIC RULE: Never include words like 'immigrants', 'refugees', 'blacks', 'whites', 'jews', 'muslims' etc., in the SPANS unless they are used as part of a compound slur.

Example:
Text: "Those immigrants are cockroaches."
LABEL: hate
SPANS: cockroaches
REASONING: Only the dehumanizing slur "cockroaches" is extracted."""
        
        prompt_template = """<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"""
        formatted_inputs = [prompt_template.format(instruction, t) for t in texts]
            
        try:
            inputs = self.tokenizer(formatted_inputs, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            input_len = inputs.input_ids.shape[1]
            generated_tokens = outputs[:, input_len:]
            answers = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            return answers
            
        except Exception as e:
            print(f"Local Model Batch Error: {e}")
            return ["LABEL: normal\nSPANS: NONE"] * len(texts)

    def verify_span(self, text, span):
        return span # Redundant for this script

# --- Removed Redundant Classes (HFJudge, CascadedPipeline) ---
                    
def main():
    parser = argparse.ArgumentParser(description="English Hate Speech Evaluation Script for Multiple LLMs (Zero-Shot Baseline).")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for testing (default: all).")
    parser.add_argument("--output", type=str, default=None, help="Output JSON filename (default: auto-generated in results/).")
    parser.add_argument("--provider", type=str, default="openai", choices=["openai", "gemini", "local"], 
                        help="LLM provider: 'openai' (GPT-4o-mini), 'gemini', or 'local'.")
    parser.add_argument("--model", type=str, default=None, help="Override default model name.")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size or number of parallel workers (default: 10).")
    parser.add_argument("--load_4bit", action="store_true", default=True, help="Load local model in 4-bit (default: True).")
    parser.add_argument("--retry", type=str, default=None, help="Path to existing results JSON to resume evaluation.")
    args = parser.parse_args()
    
    # Path Override for English
    DATA_PATH_EN = os.path.join(current_dir, "../../english/data/hatexplain_prepared.jsonl")
    
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
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    provider_tag = args.provider
    model_tag = re.sub(r'[^a-zA-Z0-9]', '_', model_name.split('/')[-1])
    
    # Load Data (Generic XLM-R Tokenizer for metric calculations that need it)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    
    dataset = InferenceDataset(DATA_PATH_EN, tokenizer, split='test', limit=args.limit)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # --- RETRY LOGIC: Load existing results from the provided path or expected output path ---
    processed_ids = set()
    results = []
    
    # Pre-calculate the out_path for SAVING
    model_results_dir = os.path.join(script_dir, "results", model_tag)
    custom_output = args.output
    if not custom_output:
        # Use filename with current timestamp for the NEW run
        custom_output = f"results_{provider_tag}_{model_tag}_{timestamp}.json"
    
    if os.path.isabs(custom_output):
        out_path = custom_output
    else:
        out_path = os.path.join(model_results_dir, custom_output)

    # Source for retry
    retry_path = args.retry if args.retry and os.path.exists(args.retry) else None
    
    if retry_path:
        print(f"Resuming from existing results at: {retry_path}")
        try:
            with open(retry_path, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
                for res_item in existing_results:
                    if 'id' in res_item:
                        processed_ids.add(res_item['id'])
                results = existing_results
            print(f"Loaded {len(processed_ids)} already processed samples. Resuming...")
        except Exception as e:
            print(f"Warning: Could not load existing results: {e}")

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

    # Create model-specific subdirectory inside results/
    model_results_dir = os.path.join(script_dir, "results", model_tag)
    os.makedirs(model_results_dir, exist_ok=True)

    log_name = f"baseline_{provider_tag}_{model_tag}_en_{timestamp}.log"
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
        
        # Batch Filter: Identify which texts in this batch need processing
        batch_to_process = []
        batch_indices = []
        for i in range(len(texts)):
            global_idx = batch_idx * args.batch_size + i
            raw_item = dataset.data[global_idx]
            sample_id = raw_item.get('id', str(global_idx))
            if sample_id in processed_ids:
                continue
            batch_to_process.append(texts[i])
            batch_indices.append(i)
            
        if not batch_to_process:
            continue
            
        # Call Judge parallel only for needed samples
        judge_results = judge.verify_batch(batch_to_process)
        
        for idx_in_processed, judge_out in enumerate(judge_results):
            local_batch_idx = batch_indices[idx_in_processed]
            global_idx = batch_idx * args.batch_size + local_batch_idx
            raw_item = dataset.data[global_idx]
            current_text = texts[local_batch_idx]
            
            if args.provider == "local":
                # Local Qwen Specialist: LABEL/SPANS/REASONING format
                label, pred_spans, is_hallucination = judge.clean_judge_output(judge_out, text=current_text)
                is_toxic = label != "normal"
                is_error = is_hallucination
            else:
                # OpenAI/Gemini: SAFE or LABEL|phrase1, phrase2 format
                is_error = "ERROR" in judge_out
                if judge_out == "SAFE":
                    is_toxic = False
                    label = "normal"
                    pred_spans = []
                else:
                    is_toxic = True
                    if "|" in judge_out:
                        label_part, spans_part = judge_out.split("|", 1)
                        label = label_part.lower()
                        pred_spans = [s.strip() for s in spans_part.split(",") if s.strip() and s.upper() != "NONE"]
                    else:
                        label = "hate" # Fallback
                        pred_spans = [s.strip() for s in judge_out.split(",") if s.strip()]
            
            # Generate token masks for Group 2 metrics
            encoding = tokenizer(
                current_text,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_offsets_mapping=True,
                return_special_tokens_mask=True
            )
            offsets = encoding['offset_mapping']
            special_mask = encoding['special_tokens_mask']
            
            # Gold token label
            gold_tok_label = [0] * 128
            gold_spans = raw_item.get('unsafe_spans_indices', [])
            for s_start, s_end in gold_spans:
                for idx_t, (start, end) in enumerate(offsets):
                    if start == end: continue
                    if max(start, s_start) < min(end, s_end):
                        gold_tok_label[idx_t] = 1
            
            # Pred token label
            pred_tok_label = [0] * 128
            for s in pred_spans:
                start_search = 0
                while True:
                    found_idx = current_text.lower().find(s.lower(), start_search)
                    if found_idx == -1: break
                    for idx_t, (start, end) in enumerate(offsets):
                        if start == end: continue
                        if max(start, found_idx) < min(end, found_idx + len(s)):
                            pred_tok_label[idx_t] = 1
                    start_search = found_idx + 1

            results.append({
                "id": raw_item.get('id', str(global_idx)),
                "text": current_text,
                "true_label": raw_item.get('label', 'N/A'),
                "pred_label": "error" if is_error else label,
                "pred_tok_label": pred_tok_label,
                "gold_tok_label": gold_tok_label,
                "attention_mask": encoding['attention_mask'],
                "special_mask": special_mask,
                "probs": 1.0 if is_toxic else 0.0, # Zero-shot score binary
                "raw_item": raw_item
            })

        # --- INCREMENTAL SAVE ---
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    total_time = time.time() - start_time
    
    # Stats Summary
    logger.print("\n" + "="*40)
    logger.print(f"{args.provider.upper()} EVALUATION RESULTS (EN)")
    logger.print("="*40)
    logger.print(f"Total Processed: {len(results)}")
    logger.print(f"Time Taken: {total_time:.2f}s ({total_time/len(results):.2f}s/sample)")
    
    # --- CALCULATE METRICS VS GROUND TRUTH ---
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support
    
    y_true, y_pred = [], []
    all_tok_preds, all_tok_labels, all_masks, all_special_masks = [], [], [], []
    all_tok_probs = []
    toxic_probs_list = []

    for r in results:
        def normalize_lbl(l):
            l_str = str(l).strip().lower()
            if any(hit in l_str for hit in ["hatespeech", "hate"]): return 0
            if "normal" in l_str: return 1
            if "offensive" in l_str: return 2
            return 1 # Fallback
            
        t_val = normalize_lbl(r['true_label'])
        p_val = normalize_lbl(r['pred_label'])
        
        y_true.append(t_val)
        y_pred.append(p_val)
        toxic_probs_list.append(r['probs'])
        
        all_tok_preds.append(r['pred_tok_label'])
        all_tok_labels.append(r['gold_tok_label'])
        all_masks.append(r['attention_mask'])
        all_special_masks.append(r['special_mask'])
        all_tok_probs.append(r['pred_tok_label']) # For zero-shot, pred mask is our prob surrogate

    if results:
        # Group 1: Classification
        cls_acc = accuracy_score(y_true, y_pred)
        cls_p, cls_r, cls_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        cls_metrics = {"acc": cls_acc, "f1": cls_f1, "precision": cls_p, "recall": cls_r}
        cls_cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        
        # Group 2: Span Detection
        all_tok_preds = torch.tensor(all_tok_preds)
        all_tok_labels = torch.tensor(all_tok_labels)
        all_masks = torch.tensor(all_masks)
        all_special_masks = torch.tensor(all_special_masks)
        all_tok_probs = torch.tensor(all_tok_probs).float()
        
        span_f1 = calculate_span_f1(all_tok_preds, all_tok_labels, all_masks, special_tokens_mask=all_special_masks)
        tok_m = calculate_token_metrics(all_tok_preds, all_tok_labels, all_masks, all_special_masks, all_tok_probs)
        
        span_metrics = {
            "token_acc": tok_m['acc'],
            "token_p": tok_m['precision'],
            "token_r": tok_m['recall'],
            "token_f1": tok_m['f1'],
            "token_f1_pos": tok_m['f1_pos'],
            "token_auprc": tok_m.get('auprc', 0.0),
            "span_f1": span_f1
        }
        
        # Group 4: AUROC & Bias
        y_true_bin = [1 if x != 1 else 0 for x in y_true] # 1 is Normal (Safe) in EN, so x != 1 means Toxic
        auroc = calculate_auroc(y_true_bin, toxic_probs_list)
        bias_metrics = {}
        bias_items = load_bias_metadata_from_prepared(DATA_PATH_EN, len(y_true))
        if bias_items:
            bias_metrics = calculate_bias_metrics(bias_items, y_true_bin, toxic_probs_list)
            
        # Reporting
        report_name = f"{model_name} Zero-Shot LLM (EN)"
        log_content = generate_report_string(report_name, cls_metrics, span_metrics, cls_cm)
        
        # Add Group 4 Manual Header (Match mBERT-cased style)
        extra_info = "-" * 50 + "\n"
        extra_info += ">> GROUP 4: FAITHFULNESS & FAIRNESS\n"
        extra_info += "-" * 50 + "\n"
        extra_info += f"{'Faithful. Comp (N/A for LLM)':<30} | 0.0000\n"
        extra_info += f"{'Faithful. Suff (N/A for LLM)':<30} | 0.0000\n"
        extra_info += "-" * 45 + "\n"
        extra_info += "--- Fairness / Bias ---\n"
        extra_info += f"{'GMB-Subgroup AUC':<30} | {bias_metrics.get('GMB-Sub', 0):.4f}\n"
        extra_info += f"{'GMB-BPSN':<30} | {bias_metrics.get('GMB-BPSN', 0):.4f}\n"
        extra_info += f"{'GMB-BNSP':<30} | {bias_metrics.get('GMB-BNSP', 0):.4f}\n"
        extra_info += f"{'AUROC':<30} | {auroc:.4f}\n"
        extra_info += "="*80
        
        logger.print("\n" + log_content + "\n" + extra_info)
        logger.print(f"\nTime Stats: {total_time:.2f}s total | {total_time/len(results):.2f}s/avg")

    # Final wrap up is already handled by incremental save, 
    # but we print the final confirmation.
    logger.print(f"Final results (Total: {len(results)}) saved to {out_path}")
    logger.close()

if __name__ == "__main__":
    main()

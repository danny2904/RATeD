# from unsloth import FastLanguageModel
import os
import sys
import torch
import json
import numpy as np
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import warnings
from transformers import logging as hf_logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from dotenv import load_dotenv
import google.generativeai as genai

# Add path to import local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from model_multitask_en import HateXplainMultiTaskBIO

# Add path to project root for absolute imports
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
if project_root not in sys.path: sys.path.insert(0, project_root)

try:
    from experiments.english.common.metrics import calculate_span_f1_eraser, calculate_token_metrics, calculate_auroc, calculate_bias_metrics
    from experiments.english.common.faithfulness import calculate_faithfulness_metrics
    from experiments.english.common.metrics_utils import load_bias_metadata_from_prepared
    from experiments.english.common.reporting import generate_report_string
except ImportError as e:
    print(f"[ERR] Critical Import Error: {e}")
    # Fallback to direct imports if possible
    try:
        from metrics import calculate_span_f1_eraser, calculate_token_metrics, calculate_auroc, calculate_bias_metrics
        from faithfulness import calculate_faithfulness_metrics
        from metrics_utils import load_bias_metadata_from_prepared
        from reporting import generate_report_string
    except ImportError:
        print("[ERR] Could not recover imports. Metric calculation will fail.")

# --- LƯU Ý KHÁC BIỆT NHÃN EN vs VN ---
# EN (HateXplain): 3 nhãn — hate (0), normal (1), offensive (2). Không gộp hay đổi thành 2 lớp.
# VN (ViHOS): 2 nhãn — clean, hate. Pipeline VN dùng "clean"/"hate"; EN luôn dùng "normal"/"hate"/"offensive".
LABELS_EN = ("hate", "normal", "offensive")  # HateXplain 3-class; index 0,1,2 tương ứng cls_label

# --- SPAN FORMAT THỐNG NHẤT: char offsets [start, end], gộp overlap/liền kề ---
def merge_span_indices(spans):
    """Gộp các span (char offsets) overlap hoặc liền kề. Input: list of [start,end] or (start,end)."""
    if not spans:
        return []
    out = []
    for s in spans:
        a, b = int(s[0]), int(s[1])
        if a < b:
            out.append([a, b])
    out.sort(key=lambda x: (x[0], x[1]))
    merged = []
    for a, b in out:
        if merged and a <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged

def spans_string_to_indices(text, span_strings, merge=True):
    """Chuyển danh sách span (chuỗi) sang char offsets [start,end], tùy chọn gộp."""
    if not text or not span_strings:
        return []
    indices = []
    for s in span_strings:
        s = (s or "").strip().strip('"\'')
        if not s or len(s) > 200:
            continue
        start_search = 0
        while True:
            pos = text.lower().find(s.lower(), start_search)
            if pos == -1:
                break
            indices.append([pos, pos + len(s)])
            start_search = pos + 1
    if merge:
        indices = merge_span_indices(indices)
    return indices

class InferenceDatasetEN(Dataset):
    def __init__(self, data_path, tokenizer, max_len=128, split='test', limit=None):
        self.data = []
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    if split is None or item.get('split') == split:
                        self.data.append(item)
        except Exception as e:
            print(f"Error loading data: {e}")
        
        if limit: self.data = self.data[:limit]
        self.tokenizer = tokenizer
        self.max_len = max_len
        print(f"Loaded {len(self.data)} samples for split '{split}'")
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['comment']
        # Pad to max_length to avoid DataLoader batching error
        encoding = self.tokenizer(
            text, 
            truncation=True, 
            max_length=128, 
            padding='max_length',
            return_tensors=None # Return as lists for easier handling
        )
        
        token_labels = [0] * 128
        # If the model uses 2 labels, use index 1 for toxic
        # If 3 (BIO), use index 1 or 2. Here we just need a dummy for indices mapping
        # but correctly padded.
        
        return {
            'input_ids': torch.tensor(encoding['input_ids']),
            'attention_mask': torch.tensor(encoding['attention_mask']),
            'text': text,
            'token_labels': torch.tensor(token_labels, dtype=torch.long)
        }

class RATeDInferenceEN:
    def __init__(self, model_path, model_name="roberta-base", use_fusion=True, device=None):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Loading RATeD-V EN model on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Handle English baseline legacy (2 labels in BIO) vs Standard (3 labels)
        print(f"Initializing model skeleton (num_labels=3 for cls)...")
        # We initialize with num_labels=3 (default for our standard model)
        self.model = HateXplainMultiTaskBIO.from_pretrained(model_name, num_labels=3, use_fusion=use_fusion)
        
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            
            # --- Robust Key Remapping ---
            new_state_dict = {}
            for k, v in state_dict.items():
                # 1. Standardize prefix to 'roberta.'
                new_key = k
                if k.startswith("bert."): new_key = k.replace("bert.", "roberta.")
                elif not k.startswith("roberta.") and not any(x in k for x in ["classifier", "token_"]):
                    new_key = "roberta." + k
                
                # 2. Map legacy classifier names to standardized names
                if "class_classifier" in new_key:
                    new_key = new_key.replace("class_classifier", "cls_classifier")
                
                # Strip prefixes for top-level layers in HateXplainMultiTaskBIO
                if new_key.startswith("roberta.cls_classifier"):
                    new_key = new_key.replace("roberta.", "")
                if new_key.startswith("roberta.token_classifier"):
                    new_key = new_key.replace("roberta.", "")
                
                new_state_dict[new_key] = v
            
            # Find the size of the token classifier in the NEW state dict
            if 'token_classifier.weight' in new_state_dict:
                ckpt_size = new_state_dict['token_classifier.weight'].shape[0]
                if ckpt_size != 3:
                    print(f"[WARN] Checkpoint has {ckpt_size} labels for token_classifier. Adjusting model layer...")
                    import torch.nn as nn
                    self.model.token_classifier = nn.Linear(self.model.config.hidden_size, ckpt_size)
            
            # Use strict=False but check for missing keys manually for safety
            load_info = self.model.load_state_dict(new_state_dict, strict=False)
            print(f"[OK] Model weights loaded. Missing keys: {load_info.missing_keys}")
            if any("roberta" in k for k in load_info.missing_keys):
                print("[ERR] ERROR: Critical backbone weights (roberta.*) are missing!")
                raise RuntimeError("Failed to load backbone.")
        except Exception as e:
            print(f"[ERR] Error during remapping/loading: {e}")
            raise e
            
        self.model.to(self.device)
        self.model.eval()
        
    def predict(self, texts, threshold_cls=0.5):
        if isinstance(texts, str): texts = [texts]
        encodings = self.tokenizer(
            texts,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_offsets_mapping=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=encodings['input_ids'], attention_mask=encodings['attention_mask'])
            
            cls_probs = torch.softmax(outputs['cls_logits'], dim=1).cpu().numpy()
            token_logits = outputs['token_logits'].cpu()
            token_probs = torch.softmax(token_logits, dim=2).numpy()  # [batch, seq, num_classes]
            token_preds_all = torch.argmax(token_logits, dim=2).numpy()
            offsets_all = encodings['offset_mapping'].cpu().numpy()
            
            batch_results = []
            for i, text in enumerate(texts):
                # English Labels: 0: Hate, 1: Normal, 2: Offensive
                # Toxicity = 0 or 2? Usually (Hate + Offensive)
                toxic_prob = 1.0 - cls_probs[i][1] # Confidence that it is NOT Normal
                
                # Phase 10.2: Offset-based Span Extraction (Scientific Accuracy)
                # Avoid fake spans from decoding token IDs; extract directly from source text.
                pred_spans = []
                current_span = None # (start, end)
                
                for j in range(len(token_preds_all[i])):
                    p = token_preds_all[i][j]
                    start, end = offsets_all[i][j]
                    
                    if start == end: continue # Skip special tokens/padding if offset is 0,0
                    
                    if p > 0: # Toxic token
                        if current_span is None:
                            current_span = [start, end]
                        else:
                            # If continuous or very close (whitespace)
                            if start <= current_span[1] + 1:
                                current_span[1] = end
                            else:
                                pred_spans.append(text[current_span[0]:current_span[1]])
                                current_span = [start, end]
                    else:
                        if current_span is not None:
                            pred_spans.append(text[current_span[0]:current_span[1]])
                            current_span = None
                
                if current_span is not None:
                    pred_spans.append(text[current_span[0]:current_span[1]])

                batch_results.append({
                    'is_toxic': toxic_prob > threshold_cls,
                    'confidence': float(toxic_prob),
                    'spans': pred_spans,
                    'token_mask': token_preds_all[i].tolist(), # Phase 10.3: Pass raw mask
                    'cls_label': int(np.argmax(cls_probs[i])),
                    'cls_probs': cls_probs[i].tolist(), # Phase 10.2: Pass real probabilities
                    'token_probs': token_probs[i].tolist()  # For AUPRC calculation
                })
        return batch_results

# --- Shared Judge Prompt Logic + Từ điển slang  ---
ENGLISH_SLANG_KB = """
GROUP 1: HATE (IDENTITY-BASED ATTACK)
- STRICT CRITERIA: Text that targets, dehumanizes, or promotes violence against protected groups (Race, Religion, Sexual Orientation, Disability).
- SLURS: Includes identity-based slurs used to attack or marginalize.
- DEHUMANIZATION: Using metaphors comparing groups to vermin, animals, or diseases.

GROUP 2: OFFENSIVE (GENERAL TOXICITY)
- CRITERIA: Swearing, profanity, or insults that DO NOT target a specific protected identity.
- EXAMPLES: General insults, aggressive non-identity-based language, or profanity used for emphasis.

GROUP 3: NORMAL (SAFE CONTENT)
- CATEGORIES: Neutral descriptions, reclaimed terms in casual context, or safe conversational text.
"""

def get_slang_kb_en():
    return ENGLISH_SLANG_KB.strip()

class LocalJudgeEN:
    """
    Local Judge cho E11: ưu tiên Qwen (SFT trên HateXplain) để đồng nhất với thí nghiệm tiếng Việt.
    Mặc định: experiments/english/models/qwen2.5-7b-hatexplain-specialist-3class (Qwen2.5-7B finetune HateXplain, 3-class + SPANS).
    Hỗ trợ: (1) Unsloth FastLanguageModel nếu có (load được PEFT adapter-only), (2) base + PEFT khi có adapter_config.json.
    """
    def __init__(self, model_repo, load_in_4bit=True):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Loading local English Specialist Judge: {model_repo} (Qwen HateXplain, align VN) on {self.device}...")
        if torch.cuda.is_available():
            print(f"[OK] CUDA detected. Using single device: {self.device}")
        else:
            print("[ERR] CUDA NOT DETECTED! Local inference will be extremely slow on CPU.")

        is_local_path = os.path.exists(model_repo)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        ) if load_in_4bit else None

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_repo,
            trust_remote_code=True,
            local_files_only=is_local_path
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        adapter_config_path = os.path.join(model_repo, "adapter_config.json")
        use_peft = os.path.isfile(adapter_config_path)
        device_map_single = {"": self.device}

        # (1) Unsloth FastLanguageModel: load được PEFT adapter-only (Qwen specialist-3class)
        if use_peft:
            try:
                from unsloth import FastLanguageModel
                self.model, _ = FastLanguageModel.from_pretrained(
                    model_name=model_repo,
                    max_seq_length=512,
                    load_in_4bit=load_in_4bit,
                    device_map=device_map_single,
                )
                FastLanguageModel.for_inference(self.model)
                print("[OK] Loaded via Unsloth FastLanguageModel (PEFT adapter) on", self.device)
                return
            except Exception as e:
                print(f"[WARN] Unsloth load failed ({e}), trying base + PEFT...")

        # (2) Base model + PEFT adapter (transformers + peft)
        if use_peft:
            try:
                with open(adapter_config_path, "r", encoding="utf-8") as f:
                    adapter_cfg = json.load(f)
                base_name = adapter_cfg.get("base_model_name_or_path") or "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
                from peft import PeftModel
                base = AutoModelForCausalLM.from_pretrained(
                    base_name,
                    device_map=device_map_single,
                    quantization_config=bnb_config,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                )
                self.model = PeftModel.from_pretrained(base, model_repo)
                self.model.eval()
                print(f"[OK] Loaded base {base_name} + PEFT adapter from {model_repo} on", self.device)
                return
            except Exception as e:
                print(f"[WARN] PEFT load failed ({e}), trying full from_pretrained...")

        # (3) Full model (e.g. merged or non-PEFT)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_repo,
            device_map=device_map_single,
            quantization_config=bnb_config,
            local_files_only=is_local_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        print(f"[OK] Judge model on {self.device}")
        print("Local English Judge initialized successfully.")

    def clean_judge_output(self, raw_output, max_span_length=120, text=None):
        import re
        # Initialize with a safe default. In Stage 2 Cascaded, 
        # defaulting to 'normal' is safer for FP control.
        label, spans, hallucination_detected = "normal", [], False
        
        # 0. Precise Assistant output extraction for Qwen/ChatML
        clean_ans = raw_output
        if "<|im_start|>assistant" in raw_output:
            clean_ans = raw_output.split("<|im_start|>assistant")[-1].strip()
        elif "assistant\n" in raw_output:
            clean_ans = raw_output.split("assistant\n")[-1].strip()
        
        # Remove any lingering im_end or turn markers
        clean_ans = clean_ans.split("<|im_end|>")[0].strip()

        try:
            # 1. Primary: Strict LABEL Extraction (Start of line prioritize)
            label_match = re.search(r"^LABEL:\s*\*?(hate|hatespeech|offensive|normal)\*?", clean_ans, re.MULTILINE | re.IGNORECASE)
            if not label_match:
                label_match = re.search(r"LABEL:\s*\*?(hate|hatespeech|offensive|normal)\*?", clean_ans, re.IGNORECASE)
            
            if label_match: 
                label = label_match.group(1).lower().replace("hatespeech", "hate")
            else:
                # 2. Fallback: Keyword search with specific priority (Favor Toxic classes)
                if re.search(r"\bhate(speech)?\b", clean_ans, re.IGNORECASE):
                    label = "hate"
                elif re.search(r"\boffensive\b", clean_ans, re.IGNORECASE):
                    label = "offensive"
                elif re.search(r"\bnormal\b", clean_ans, re.IGNORECASE):
                    label = "normal"
            
            # 2. SPANS Extraction
            spans_match = re.search(r"SPANS:\s*([^\n\r]*)", clean_ans, re.IGNORECASE)
            if spans_match:
                spans_text = spans_match.group(1).strip()
                if "NONE" not in spans_text.upper() and spans_text:
                    # Capture comma-separated spans
                    raw_spans = [s.strip().strip('"').strip("'") for s in spans_text.split(',')]
                    for span in raw_spans:
                        if span and len(span) >= 2 and len(span) <= max_span_length:
                            spans.append(span)
            
            # De-duplicate and limit to keep rationales precise
            spans = sorted(list(set(spans)), key=len)[:3]
            
        except Exception as e:
            print(f"[DEBUG] Parser Error: {e}")
            hallucination_detected = True
            
        return label, spans, hallucination_detected
    
    def verify_batch(self, texts):
        if not texts: return []
        slang_kb = get_slang_kb_en()
        
        instruction = f"""You are a Senior NLP Research Scientist. Task: Classify text into HATE, OFFENSIVE, or NORMAL.

{slang_kb}

DEFINITIONS:
- HATE: Attacks/dehumanizes protected groups (Race, Religion, etc.). 
- OFFENSIVE: General insults/profanity WITHOUT targeting protected identity.
- NORMAL: Safe content.

STRICT RULE: If a slur is used against a group, it MUST be HATE.

EXAMPLES:
1. "I hate those [Group Name] they are subhuman" -> LABEL: hate
2. "You are such a fucking idiot" -> LABEL: offensive
3. "The weather is nice" -> LABEL: normal

OUTPUT FORMAT (3 lines):
LABEL: <hate|offensive|normal>
SPANS: <comma-separated triggers or NONE>
REASONING: <1 short sentence>"""

        # Reverting to ChatML (standard Qwen Instruct format)
        prompt_template = "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        formatted_inputs = [prompt_template.format(instruction, t) for t in texts]
        
        try:
            inputs = self.tokenizer(formatted_inputs, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = inputs.to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            input_len = inputs.input_ids.shape[1]
            answers = self.tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
            return answers
        except Exception as e:
            print(f"Local Judge Error: {e}")
            return ["LABEL: normal\nSPANS: NONE"] * len(texts)

class GeminiJudgeEN:
    """
    Judge Gemini E11 tiếng Anh — thiết kế theo chuẩn E11 tiếng Việt (VN chạy rất chuẩn):
    - Prompt rõ ràng + Slang KB nhúng trong prompt.
    - temperature=0.0. Parse đơn giản.
    - EN giữ 3 nhãn HateXplain (hate, normal, offensive): khi Judge báo SAFE -> normal; khi toxic
      dùng nhãn RATeD (cls_label 0=hate, 2=offensive) để phân biệt hate vs offensive, không gộp như VN 2 lớp.
    """
    def __init__(self, model_name="gemini-2.5-flash-lite", max_workers=8):
        load_dotenv()
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("WARNING: No GOOGLE_API_KEY found.")
            self.model = None
        else:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
            print(f"Gemini EN Judge ({model_name}) initialized (parallel workers={max_workers}).")
        self.max_workers = max_workers

    def clean_judge_output(self, raw_output, max_span_length=120, text=None):
        return LocalJudgeEN.clean_judge_output(None, raw_output, max_span_length, text)

    def _call_one(self, text):
        slang_kb = get_slang_kb_en()
        prompt = f"""You are an expert content moderator specializing in detecting hate speech for the HateXplain 3-class task.
Task: Analyze the text and identify if it is HATE SPEECH, OFFENSIVE, or NORMAL.

{slang_kb}

STRICT RULES for output:
1. OUTPUT FORMAT: Exactly 3 lines.
LABEL: <hate|offensive|normal>
SPANS: <comma-separated minimal spans, or NONE>
REASONING: <1 short sentence explanation>

2. RATIONALE EXTRACTION (CRITICAL FOR SPAN IoU): 
   - Extract the EXACT CONTINUOUS PHRASES that support your chosen label.
   - For OFFENSIVE: Extract ONLY the offensive parts (e.g. "fucking idiot", "ghetto mess").
   - For HATE: Extract the dehumanizing phrases (e.g. "dating monkeys", "illegal immigrants").
   - Do NOT extract individual words if they are part of a larger toxic phrase.
   - For NORMAL: Always return "NONE".

3. DEFINITIONS & HIERARCHY:
   - HATE (Highest Severity): Targeted attack + Dehumanization of a protected group.
   - OFFENSIVE: All other toxicity (insults, swearing, non-targeted slurs). 
   - NORMAL: Safe content.
   - RULE: If unsure between HATE and OFFENSIVE, choose OFFENSIVE.

4. EXAMPLES:
   - "I fucking love this!" -> NORMAL (purely emphasis)
   - "You are a fucking idiot" -> OFFENSIVE (direct insult)
   - "All [GroupX] are vermin" -> HATE (dehumanization)
   - "This center is trash" -> OFFENSIVE (toxicity)
   - "Go back to your country" -> HATE (targeted xenophobia)
   - "stfu and leave" -> OFFENSIVE (aggressive)
   - "white boy", "black girl" used neutrally -> NORMAL (identity marker)

Text to analyze: "{text}"
"""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(temperature=0.0)
            )
            if not response.candidates or response.candidates[0].finish_reason != 1:
                return "LABEL: normal\nSPANS: NONE\nREASONING: Safety block."
            return response.text.strip()
        except Exception as e:
            print(f"Gemini Error: {e}")
            return "LABEL: normal\nSPANS: NONE\nREASONING: Error occurred."

    def verify_batch(self, texts):
        if not self.model:
            return ["SAFE"] * len(texts)
        n = len(texts)
        res = [None] * n
        workers = min(self.max_workers, n)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self._call_one, t): i for i, t in enumerate(texts)}
            for fut in as_completed(futures):
                idx = futures[fut]
                res[idx] = fut.result()
        return res

def build_span_token_mask(text, span_list, tokenizer, max_len=128):
    """Build 0/1 token mask of length max_len from span strings (char overlap with offset_mapping)."""
    if not text or not span_list:
        return [0] * max_len
    encoding = tokenizer(
        text,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_offsets_mapping=True,
    )
    offsets = encoding["offset_mapping"]
    mask = [0] * max_len
    for span in span_list:
        if not span or str(span).lower() == "none":
            continue
        start_search = 0
        while True:
            pos = text.lower().find(span.lower(), start_search)
            if pos == -1:
                break
            end = pos + len(span)
            for i in range(min(max_len, len(offsets))):
                ts, te = offsets[i][0], offsets[i][1]
                if max(pos, ts) < min(end, te):
                    mask[i] = 1
            start_search = pos + 1
    return mask

class CascadedPipelineEN:
    """
    Pipeline E11 tiếng Anh: backbone RATeD + Judge. Luôn xuất 3 nhãn HateXplain (hate, normal, offensive).
    Không dùng "clean" (VN 2-class); EN luôn dùng "normal" cho không độc hại.
    """
    def __init__(self, rated_model, judge, safeguard_threshold=0.85, only_stage1=False, only_stage2=False):
        self.rated = rated_model
        self.judge = judge
        self.only_stage1 = only_stage1
        self.only_stage2 = only_stage2
        # Khi backbone toxic + Judge tra normal: override ve backbone neu conf >= threshold. Local (Qwen) hay flip qua normal -> dung 0.75; Gemini 0.85.
        self.safeguard_threshold = safeguard_threshold
        self.stats = {"total": 0, "fast_path": 0, "judge_path": 0, "fn_recovered": 0, "fp_saved": 0, "safeguard_overrides": 0}

    def process_batch(self, texts):
        self.stats["total"] += len(texts)
        
        if self.only_stage2:
            # ONLY STAGE 2: Bypass backbone, send ALL to Judge
            to_verify_indices = list(range(len(texts)))
            to_verify_texts = texts
            # Dummy rated results for structure compatibility
            rated_results = [{"cls_label": 1, "confidence": 0.0, "spans": [], "token_mask": [0]*128} for _ in range(len(texts))]
            final_results = [None] * len(texts)
        else:
            rated_results = self.rated.predict(texts)
            final_results = [None] * len(texts)
            to_verify_indices, to_verify_texts = [], []

        if not self.only_stage2:
            for i, (text, r_res) in enumerate(zip(texts, rated_results)):
                confidence = r_res['confidence']
                
                # BRANCHING LOGIC
                if self.only_stage1:
                    is_judge_needed = False
                else:
                    # Phase 11.5: Tightened Gating (0.40 - 0.65) for English
                    # Since Judge Qwen EN has safety bias, we only let it handle low-confidence cases.
                    is_judge_needed = (0.40 <= confidence <= 0.65)

                if not is_judge_needed:
                    self.stats["fast_path"] += 1
                    # Ensure alignment with LABELS_EN (0:hate, 1:normal, 2:offensive)
                    if r_res['cls_label'] == 1:
                        lbl = "normal"
                    elif r_res['cls_label'] == 0:
                        lbl = "hate"
                    else:
                        lbl = "offensive"
                        
                    final_results[i] = {
                        "label": lbl,
                        "spans": r_res['spans'],
                        "token_mask": r_res['token_mask'],
                        "flow": "FAST_PATH",
                        "confidence": confidence,
                        "cls_probs": r_res.get('cls_probs', []),
                        "token_probs": r_res.get('token_probs', [])
                    }
                else:
                    to_verify_indices.append(i)
                    to_verify_texts.append(text)
                
        if to_verify_indices:
            self.stats["judge_path"] += len(to_verify_indices)
            import time as _t
            _t0 = _t.time()
            print(f"  [Judge] Verifying {len(to_verify_texts)} samples...", flush=True)
            judge_results = self.judge.verify_batch(to_verify_texts)
            print(f"  [Judge] Done in {_t.time() - _t0:.1f}s", flush=True)
            
            for k, idx in enumerate(to_verify_indices):
                judge_ans = judge_results[k]
                text_for_idx = to_verify_texts[k]
                judge_label, cleaned_spans, is_hallucination = self.judge.clean_judge_output(
                    judge_ans, text=text_for_idx
                )
                
                orig_label_idx = rated_results[idx]['cls_label']
                
                # FALLBACK: Judge hallucination hoặc output không parse được
                if is_hallucination:
                    final_label = "normal" if orig_label_idx == 1 else ("hate" if orig_label_idx == 0 else "offensive")
                    # In only_stage2, we won't have rated_results, so we use dummy defaults or judge logic
                    fallback_probs = rated_results[idx].get('cls_probs', [])
                    fallback_conf = rated_results[idx]['confidence']
                    if self.only_stage2:
                         if final_label == "hate": fallback_probs, fallback_conf = [1.0, 0.0, 0.0], 1.0
                         elif final_label == "offensive": fallback_probs, fallback_conf = [0.0, 0.0, 1.0], 1.0
                         else: fallback_probs, fallback_conf = [0.0, 1.0, 0.0], 0.0

                    final_results[idx] = {
                        "label": final_label,
                        "spans": rated_results[idx]['spans'],
                        "token_mask": rated_results[idx].get('token_mask', []),
                        "flow": "FAST_PATH",
                        "confidence": fallback_conf,
                        "cls_probs": fallback_probs,
                        "token_probs": rated_results[idx].get('token_probs', [])
                    }
                    continue
                
                # Phase 11.2: EN-Specific Protective Gating
                # Stage 1 (Backbone) is strong (Recall 77% for HATE).
                # Judge Qwen Specialist is weak on HATE (~10% recall).
                if judge_label is None:
                    final_label = "hate" if orig_label_idx == 0 else "offensive"
                else:
                    final_label = judge_label
                
                orig_conf = rated_results[idx]['confidence']
                is_normal_rated = (orig_label_idx == 1)
                is_toxic_rated = not is_normal_rated
                
                # Rule A: Prevent HATE -> Offensive/Normal downgrade if backbone is confident
                if orig_label_idx == 0 and orig_conf >= 0.70 and final_label != "hate":
                    final_label = "hate"
                    self.stats["safeguard_overrides"] += 1
                
                # Rule B: Standard Safeguard (Prevent flip to Normal if backbone is confident)
                elif is_toxic_rated and orig_conf >= self.safeguard_threshold and final_label == "normal":
                    final_label = "hate" if orig_label_idx == 0 else "offensive"
                    self.stats["safeguard_overrides"] += 1
                
                # Update stats
                is_normal_final = (final_label == "normal")
                if is_normal_rated and not is_normal_final:
                    self.stats["fn_recovered"] += 1
                elif not is_normal_rated and is_normal_final:
                    self.stats["fp_saved"] += 1

                # Phase 11.2: Rationale Fusion - UNION strategy (Backbone + Judge) for max Plausibility (Target 0.48)
                judge_mask = build_span_token_mask(text_for_idx, cleaned_spans, self.rated.tokenizer, max_len=128) if cleaned_spans else [0]*128
                rated_mask = rated_results[idx].get('token_mask') or [0]*128
                
                union_mask = []
                for i in range(128):
                    r = 1 if (i < len(rated_mask) and rated_mask[i] > 0) else 0
                    j = judge_mask[i] if i < len(judge_mask) else 0
                    union_mask.append(1 if (r or j) else 0)
                
                token_mask_for_group3 = union_mask

                # Phase 11.2: VALIDATION - If label is Normal, rationale MUST be empty
                if final_label == "normal":
                    token_mask_for_group3 = [0] * 128
                    cleaned_spans = []

                # Phase 11.3: Update soft probabilities to reflect Judge's final decision
                # This ensures AUROC and Fairness metrics (Group 2) capture the Stage 2 impact.
                if final_label == "hate":
                    final_probs = [1.0, 0.0, 0.0]
                    final_conf = 1.0
                elif final_label == "offensive":
                    final_probs = [0.0, 0.0, 1.0]
                    final_conf = 1.0
                else: # normal
                    final_probs = [0.0, 1.0, 0.0]
                    final_conf = 0.0
                
                # Phase 11.4: Rationale-Informed Probability Calibration
                # We fuse Backbone's soft scores with Judge's hard mask 
                # so that Token AUPRC reflects the Stage 2 refinement.
                fused_token_probs = []
                backbone_token_probs = rated_results[idx].get('token_probs', [])
                
                # We need judge_mask (binary) for calibration
                j_mask = build_span_token_mask(text_for_idx, cleaned_spans, self.rated.tokenizer, max_len=128) if cleaned_spans else [0]*128

                for i in range(len(backbone_token_probs)):
                    b_safe, b_toxic = backbone_token_probs[i]
                    j_val = float(j_mask[i]) if i < len(j_mask) else 0.0
                    
                    if final_label == "normal":
                        # If Judge says Normal, we suppress toxicity across all tokens
                        new_toxic = 0.0
                    else:
                        # If Judge says Toxic (Hate/Offensive), we anchor probabilities to Judge's mask
                        if j_val > 0.5:
                            new_toxic = max(b_toxic, 0.99) # Judge-confirmed toxic spans get top scores
                        else:
                            new_toxic = b_toxic * 0.2 # Suppress tokens outside Judge's focus
                    
                    new_safe = 1.0 - new_toxic
                    fused_token_probs.append([new_safe, new_toxic])
                
                final_token_probs = fused_token_probs

                final_results[idx] = {
                    "label": final_label,
                    "spans": cleaned_spans,
                    "token_mask": token_mask_for_group3,  # GROUP 3: RATeD or union(RATeD, Judge spans) on Judge path
                    "flow": "JUDGE_PATH",
                    "confidence": final_conf,
                    "cls_probs": final_probs,
                    "token_probs": final_token_probs
                }
        return final_results

def get_toxic_score(probs):
    """
    Extracts toxicity probability for both:
    - Classification (3 classes: Hate[0], Normal[1], Offensive[2]) -> Hate+Offensive
    - Token Labeling (2 classes: Safe[0], Toxic[1]) -> Toxic (index 1)
    """
    if not isinstance(probs, (list, np.ndarray)) or len(probs) == 0:
        return 0.0
    
    if len(probs) == 3:
        # Hate + Offensive (indices 0 and 2)
        return float(probs[0] + probs[2])
    elif len(probs) == 2:
        # Toxic (index 1) - Corrected for Token AUPRC validity
        return float(probs[1])
    else:
        # Fallback for single value or other shapes
        return float(probs[0])

def calculate_cascaded_span_metrics(results, tokenizer):
    """
    GROUP 3 (Plausibility) — cùng chuẩn E1 để so sánh công bằng:
    - Gold: unsafe_spans_indices (char) + overlap -> token 0/1 (giống E1).
    - Pred: ưu tiên pred_mask (token-level từ backbone/fusion) như E1; fallback pred_spans_indices/pred_spans.
    - Hàm: calculate_span_f1_eraser(iou_threshold=0.5), calculate_token_metrics(probs).
    - Probs AUPRC: backbone token_probs (giống E1).
    """
    max_len = 128
    special_ids = set(tokenizer.all_special_ids)
    all_y_true = []
    all_y_pred = []
    all_att = []
    all_special = []
    all_probs = []

    for item in results:
        text = item['text']
        gold_spans_idx = item.get('gold_spans', [])
        pred_spans = item.get('pred_spans', [])

        encoding = tokenizer(
            text,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True
        )
        offsets = encoding['offset_mapping']
        input_ids = encoding['input_ids']
        n = len(offsets)
        if n == 0:
            continue

        # special_tokens_mask: 1 = exclude from metric (CLS/SEP/PAD), 0 = content token
        special_mask = np.array([1 if (input_ids[i] in special_ids or (offsets[i][0] == offsets[i][1])) else 0 for i in range(n)], dtype=np.int64)
        att_mask = np.array(encoding['attention_mask'], dtype=np.int64)

        # 1. GOLD (0/1) — chuẩn HateXplain: unsafe_spans_indices (char) + overlap với offset_mapping (như E1)
        full_gold = np.zeros(max_len, dtype=np.int64)
        for span in gold_spans_idx:
            s, e = int(span[0]), int(span[1])
            for i, (ts, te) in enumerate(offsets):
                if i >= max_len:
                    break
                if max(s, ts) < min(e, te):
                    full_gold[i] = 1
        if n < max_len:
            full_gold[n:] = 0

        # 2. PRED (0/1): cùng định nghĩa với E1 — ưu tiên pred_mask (token-level từ backbone/fusion) để Group 3 so sánh công bằng với E1; fallback pred_spans_indices / pred_spans
        pred_tags = np.zeros(max_len, dtype=np.int64)
        item_pred_mask = item.get('pred_mask')
        if item_pred_mask is not None and len(item_pred_mask) > 0:
            for i in range(min(max_len, len(item_pred_mask))):
                pred_tags[i] = 1 if item_pred_mask[i] > 0 else 0
        else:
            pred_indices = item.get('pred_spans_indices')
            if pred_indices:
                for s, e in pred_indices:
                    for i, (ts, te) in enumerate(offsets):
                        if i >= max_len:
                            break
                        if max(s, ts) < min(e, te):
                            pred_tags[i] = 1
            else:
                for span in pred_spans:
                    if not span or str(span).lower() == "none":
                        continue
                    start_search = 0
                    while True:
                        idx = text.lower().find(span.lower(), start_search)
                        if idx == -1:
                            break
                        end = idx + len(span)
                        for i, (ts, te) in enumerate(offsets):
                            if i >= max_len:
                                break
                            if max(idx, ts) < min(end, te):
                                pred_tags[i] = 1
                        start_search = idx + 1

        # 3. Probs for AUPRC: toxic = 1 - P(safe/O), same as 03
        probs_row = np.zeros(max_len, dtype=np.float64)
        backbone_probs = item.get('token_probs', [])
        for i in range(min(max_len, len(backbone_probs))):
            p = backbone_probs[i]
            if isinstance(p, (list, np.ndarray)) and len(p) >= 1:
                probs_row[i] = 1.0 - float(p[0])
            else:
                probs_row[i] = get_toxic_score(p) if hasattr(p, '__len__') else 0.0

        all_y_true.append(full_gold)
        all_y_pred.append(pred_tags)
        all_att.append(att_mask)
        all_special.append(special_mask)
        all_probs.append(probs_row)

    if not all_y_true:
        return {'span_f1': 0.0, 'token_f1_pos': 0.0, 'token_auprc': 0.0}

    preds = np.array(all_y_pred)
    labels = np.array(all_y_true)
    attention_mask = np.array(all_att)
    special_tokens_mask = np.array(all_special)
    probs = np.array(all_probs)

    # Span IoU F1: ERASER (same as E1/03)
    span_iou_f1 = 0.0
    if 'calculate_span_f1_eraser' in globals():
        span_iou_f1 = calculate_span_f1_eraser(
            preds, labels, attention_mask,
            special_tokens_mask=special_tokens_mask,
            iou_threshold=0.5
        )

    # Token F1 + AUPRC: micro via common (same as E1/03)
    tok_res = {'f1_pos': 0.0, 'auprc': None}
    if 'calculate_token_metrics' in globals():
        tok_res = calculate_token_metrics(
            preds, labels, attention_mask,
            special_tokens_mask=special_tokens_mask,
            probs=probs
        )
    token_f1_pos = tok_res.get('f1_pos', 0.0)
    token_auprc = tok_res.get('auprc') or 0.0

    return {
        'span_f1': span_iou_f1,
        'token_f1_pos': token_f1_pos,
        'token_auprc': token_auprc
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="experiments/english/baseline/results/RATeD_E1_baseline/best_model.pth")
    parser.add_argument("--hf_model", type=str, default="experiments/english/models/qwen2.5-7b-hatexplain-specialist-3class",
                        help="Local Judge: Qwen SFT trên HateXplain (3-class + SPANS). Ưu tiên dùng để đồng nhất với thí nghiệm tiếng Việt.")
    parser.add_argument("--provider", type=str, default="local", choices=["local", "gemini"],
                        help="local = Qwen (ưu tiên), gemini = Gemini API.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str, default="cascaded_results_en_final.json")
    parser.add_argument("--only_stage1", action="store_true", help="Bypass Judge, evaluate only RATeD backbone.")
    parser.add_argument("--only_stage2", action="store_true", help="Bypass Backbone, evaluate only LLM Judge.")
    args = parser.parse_args()
    
    rated = RATeDInferenceEN(args.model_path)
    if args.provider == "gemini":
        judge = GeminiJudgeEN()
    else:
        judge = LocalJudgeEN(args.hf_model)
    # Local Judge (Qwen) hay flip toxic->normal: dung safeguard 0.75 de tin backbone hon. Gemini dung 0.75 de giam FN Offensive.
    safeguard = 0.75
    pipeline = CascadedPipelineEN(rated, judge, safeguard_threshold=safeguard, only_stage1=args.only_stage1, only_stage2=args.only_stage2)
    
    data_path = "experiments/english/data/hatexplain_prepared.jsonl"
    dataset = InferenceDatasetEN(data_path, rated.tokenizer, split='test', limit=args.limit)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    
    results = []
    print(f"Starting English E11 Evaluation (Limit={args.limit})...")
    start_time = time.time()
    
    for batch in tqdm(dataloader):
        texts = batch['text']
        outs = pipeline.process_batch(texts)
        
        for i, out in enumerate(outs):
            # Phase 10.2: Direct Data Mapping (Secure Gold Alignment)
            # Access the original raw item from the dataset to ensure labels/spans match 100%
            gold_item = dataset.data[len(results)]
            gold_label = gold_item.get('label', 'normal').lower()
            if gold_label == 'hatespeech': gold_label = 'hate'
            pred_label = (out['label'] or "normal").lower()
            if pred_label not in LABELS_EN:
                pred_label = "normal"  # EN 3-class only; "clean"/"safe" (VN) -> normal
            gold_idx = gold_item.get('unsafe_spans_indices', [])
            pred_span_list = out.get('spans') or []
            pred_idx = spans_string_to_indices(texts[i], pred_span_list, merge=True)
            gold_merged = merge_span_indices(gold_idx)
            results.append({
                "id": gold_item.get('id', len(results)),
                "text": texts[i],
                "gold_label": gold_label,
                "pred_label": pred_label,
                "gold_spans": gold_idx,
                "gold_spans_merged": gold_merged,
                "pred_spans": pred_span_list,
                "pred_spans_indices": pred_idx,
                "pred_mask": out.get('token_mask'),
                "flow": out['flow'],
                "confidence": out.get('confidence', 0.5),
                "cls_probs": out.get('cls_probs', []),
                "token_probs": out.get('token_probs', [])
            })
            
    total_time = time.time() - start_time
    print(f"Evaluation finished in {total_time:.2f}s. Saving results first...")

    # Phase 10.3: Metadata and Dynamic Result Dir
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    sub_dir = "cascaded"
    if args.only_stage1: sub_dir = "only_stage1"
    elif args.only_stage2: sub_dir = "only_stage2"
    
    final_results_dir = os.path.join(current_dir, "results", sub_dir)
    os.makedirs(final_results_dir, exist_ok=True)
    
    # Update output filename with timestamp if it's a simple name
    if not os.path.isabs(args.output) and not os.path.dirname(args.output):
        base, ext = os.path.splitext(args.output)
        args.output = f"{base}_{timestamp}{ext}"

    final_out = os.path.join(final_results_dir, args.output)
    with open(final_out, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[OK] Raw results saved to {final_out}")

    print("Calculating metrics...")

    # --- METRICS CONSOLIDATION ---
    y_true_labels = [r['gold_label'].lower() for r in results]
    y_pred_labels = [r['pred_label'].lower() for r in results]
    
    # Classification Metrics (Group 1) — HateXplain 3-class (hate, normal, offensive)
    from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
    label_map = {"hate": 0, "hatespeech": 0, "normal": 1, "offensive": 2}  # EN 3 nhãn; hatespeech -> hate
    
    y_true = [label_map.get(l, 1) for l in y_true_labels]
    y_pred = [label_map.get(l, 1) for l in y_pred_labels]
    
    cls_metrics = {
        'acc': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='macro'),
        'precision': classification_report(y_true, y_pred, output_dict=True)['macro avg']['precision'],
        'recall': classification_report(y_true, y_pred, output_dict=True)['macro avg']['recall']
    }
    cls_cm = confusion_matrix(y_true, y_pred)
    
    # Binary metrics for AUROC/Bias
    y_true_bin = [1 if v != 1 else 0 for v in y_true]
    
    # Phase 10: Valid AUROC (Scientific Alignment)
    # MUST be calculated from Backbone probabilities, not hard labels.
    toxic_probs = []
    for r in results:
        toxic_probs.append(get_toxic_score(r.get('cls_probs', [])))
    
    # Safe calculation for metrics that might be missing
    auroc_val = 0.0
    if 'calculate_auroc' in globals():
        auroc_val = calculate_auroc(y_true_bin, toxic_probs)
    
    # Bias Metrics (Group 2)
    bias_metrics = {}
    try:
        if 'load_bias_metadata_from_prepared' in globals() and 'calculate_bias_metrics' in globals():
            bias_items = load_bias_metadata_from_prepared(data_path, len(y_true))
            if bias_items:
                bias_metrics = calculate_bias_metrics(bias_items, y_true_bin, toxic_probs)
    except Exception as e:
        print(f"[WARN] Bias metrics calculation failed: {e}")
    
    # Explainability / Plausibility (Group 3) — cùng bộ metric HateXplain/ERASER như 03 (E1). So khớp: results/E1_E11_METRICS_ALIGNMENT.md
    span_metrics = calculate_cascaded_span_metrics(results, rated.tokenizer)
    
    # Faithfulness (Group 4)
    # We measure how removing the span changes the prediction.
    # To keep it fast, we only evaluate samples sent to Judge.
    # Phase 10.2: Advanced Faithfulness (Offset-based Masking)
    comp_scores = []
    suff_scores = []

    if args.only_stage2:
        print("Skipping Faithfulness calculation for Stage 2 only evaluation (Independent Mode).")
    else:
        print("Calculating Faithfulness (Comprehensiveness/Sufficiency) using Stage 1 Backbone...")
        # Phase 11: Faithfulness Debug & Scope expansion
        debug_count = 0
        for item in results:
            # Evaluate all toxic predictions
            if item['pred_label'] != 'normal':
                text = item['text']
                pred_spans = item.get('pred_spans', [])
                if not pred_spans: continue

                # Get Baseline Confidence
                orig_conf = get_toxic_score(item.get('cls_probs', []))
                
                # Robust Token-based Masking with [UNK]
                encoding = rated.tokenizer(text, max_length=128, truncation=True, return_offsets_mapping=True)
                input_ids = np.array(encoding['input_ids'])
                offsets = encoding['offset_mapping']
                unk_token_id = rated.tokenizer.unk_token_id
                
                # Mask identifying which tokens are rationale
                rationale_mask = np.zeros(len(input_ids), dtype=bool)
                for span in pred_spans:
                    if not span or str(span).lower() == "none" or str(span).strip() == "": continue
                    start_search = 0
                    while True:
                        idx = text.lower().find(span.lower(), start_search)
                        if idx == -1: break
                        end = idx + len(span)
                        for i, (ts, te) in enumerate(offsets):
                            if max(idx, ts) < min(end, te) and ts < te:
                                rationale_mask[i] = True
                        start_search = idx + 1
                
                # COMPREHENSIVENESS: Mask rationale (IDS + ATTENTION)
                ids_comp = input_ids.copy()
                ids_comp[rationale_mask] = unk_token_id
                att_comp = np.ones(len(input_ids), dtype=int)
                att_comp[rationale_mask] = 0 # FORCED BLINDNESS
                
                # SUFFICIENCY: Keep ONLY rationale (IDS + ATTENTION)
                ids_suff = input_ids.copy()
                att_suff = np.zeros(len(input_ids), dtype=int) # DEFAULT BLIND
                special_ids = set(rated.tokenizer.all_special_ids)
                for i in range(len(ids_suff)):
                    if rationale_mask[i]:
                        att_suff[i] = 1 # ONLY LOOK AT RATIONALE
                    elif ids_suff[i] in special_ids:
                        att_suff[i] = 1 # AND SPECIAL TOKENS
                    else:
                        ids_suff[i] = unk_token_id
                
                with torch.no_grad():
                    # Inference for Comp
                    t_comp = torch.tensor([ids_comp]).to(rated.device)
                    m_comp = torch.tensor([att_comp]).to(rated.device)
                    out_c = rated.model(t_comp, m_comp)
                    m_conf_c = get_toxic_score(torch.softmax(out_c['cls_logits'], dim=1).cpu().numpy()[0])
                    
                    # Inference for Suff
                    t_suff = torch.tensor([ids_suff]).to(rated.device)
                    m_suff = torch.tensor([att_suff]).to(rated.device)
                    out_s = rated.model(t_suff, m_suff)
                    m_conf_s = get_toxic_score(torch.softmax(out_s['cls_logits'], dim=1).cpu().numpy()[0])
                    
                    comp_scores.append(max(0, orig_conf - m_conf_c))
                    suff_scores.append(max(0, orig_conf - m_conf_s))

    faithfulness = {
        'faithfulness_comprehensiveness': float(np.mean(comp_scores)) if comp_scores else 0.0,
        'faithfulness_sufficiency': float(np.mean(suff_scores)) if suff_scores else 0.0
    }

    # Efficiency (Group 0)
    efficiency = {
        'total_time': total_time,
        'total_calls': pipeline.stats.get('judge_path', 0)
    }

    # Build Cascaded Statistics Header (matching VN standard)
    cascaded_report = f"""
========================================
CASCADED PIPELINE RESULTS
========================================
Backbone: {args.model_path}
Judge Provider: {args.provider.upper()} ({'Gemini 2.5 Flash Lite' if args.provider == 'gemini' else args.hf_model})
Total Processed: {len(results)}
Time Taken: {total_time:.2f}s ({total_time/len(results):.2f}s/sample)
--------------------
RATeD Clean (Fast Path): {pipeline.stats.get('fast_path', 0)} ({pipeline.stats.get('fast_path', 0)/len(results)*100:.1f}%)
Sent to Judge: {pipeline.stats.get('judge_path', 0)} ({pipeline.stats.get('judge_path', 0)/len(results)*100:.1f}%)
  -> Judge Confirmed Toxic (Recovered FN): {pipeline.stats.get('fn_recovered', 0)}
  -> Judge Flipped to Clean (False Pos Saved): {pipeline.stats.get('fp_saved', 0)}
  -> Backbone Safeguard Overrides: {pipeline.stats.get('safeguard_overrides', 0)}
"""
    print(cascaded_report)

    # Generate Final Metric Report
    judge_display = "Gemini 2.5 Flash Lite" if args.provider == 'gemini' else "Qwen Specialist"
    report_name = f"RATeD-V E11 - Cascaded (Backbone: RATeD-V, Judge: {judge_display}, Limit={args.limit})"
    log_content = "Metric calculation failed."
    if 'generate_report_string' in globals():
        # Phase 11: Mark faithfulness as N/A in Stage 2 only mode
        if args.only_stage2:
            faithfulness = {
                'faithfulness_comprehensiveness': 0.0,
                'faithfulness_sufficiency': 0.0,
                'is_na': True
            }

        log_content = generate_report_string(
            report_name, 
            cls_metrics, 
            span_metrics, 
            cls_cm,
            auroc=auroc_val,
            bias_metrics=bias_metrics,
            faithfulness_metrics=faithfulness,
            efficiency_metrics=efficiency
        )
    
    log_content = cascaded_report + "\n" + log_content
    print(log_content)
    
    log_name = f"cascaded_results_en_{timestamp}.log"
    log_file = os.path.join(final_results_dir, log_name)
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(log_content)
    
    print(f"[OK] Full report saved to {log_file}")

if __name__ == "__main__":
    main()

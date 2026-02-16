import os
import sys
import torch
import torch.nn as nn
import json
import numpy as np
import argparse
import time
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, AutoModel
import google.generativeai as genai
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
from dotenv import load_dotenv

# Load env immediately
load_dotenv()

# Add current script dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- SUPPRESS WARNINGS ---
import logging
import warnings
from transformers import logging as hf_logging

logging.getLogger("transformers").setLevel(logging.ERROR)
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# --- METRIC HELPERS (SYNCED WITH BASELINE) ---
try:
    from experiments.english.common.metrics import calculate_span_f1, calculate_token_metrics, calculate_auroc, calculate_bias_metrics
    from experiments.english.common.faithfulness import calculate_faithfulness_metrics
    from experiments.english.common.metrics_utils import load_bias_metadata_from_prepared
    from experiments.english.common.reporting import generate_report_string
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
    if project_root not in sys.path:
        sys.path.append(project_root)
    # Add common folder specifically to help faithfulness relative imports? 
    # faithfulness.py might depend on relative imports.
    common_dir = os.path.join(project_root, "experiments/english/common")
    if common_dir not in sys.path:
        sys.path.append(common_dir)
        
    from experiments.english.common.metrics import calculate_span_f1, calculate_token_metrics, calculate_auroc, calculate_bias_metrics
    from experiments.english.common.faithfulness import calculate_faithfulness_metrics
    from experiments.english.common.metrics_utils import load_bias_metadata_from_prepared
    from experiments.english.common.reporting import generate_report_string


# --- MODELS & PIPELINE ---
# Class is defined below to avoid dependency issues
# from hatexplain_model import HateXplainMultiTaskModel

class HateXplainInferenceDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=128, split='test', limit=None):
        self.data = []
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        if split is None or item.get('split') == split:
                            self.data.append(item)
                    except: pass
        except FileNotFoundError:
            print(f"File not found: {data_path}")
        
        if limit: self.data = self.data[:limit]
        self.tokenizer = tokenizer
        self.max_len = max_len
        print(f"Loaded {len(self.data)} samples for split '{split}'")

    def __len__(self): return len(self.data)
    def __getitem__(self, index):
        item = self.data[index]
        text = str(item.get('comment', ''))
        
        encoding = self.tokenizer(
            text, 
            add_special_tokens=True, 
            max_length=self.max_len, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt',
            return_offsets_mapping=True,
            return_special_tokens_mask=True
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        offsets = encoding['offset_mapping'].squeeze(0) # Shape (MaxLen, 2)
        special_mask = encoding['special_tokens_mask'].squeeze(0)
        
        # ... (keep existing gold label logic) ...
        # Generate Gold Token Labels
        token_labels = torch.zeros(self.max_len, dtype=torch.long)
        unsafe_spans = item.get('unsafe_spans_indices', [])
        
        if unsafe_spans:
            offsets_np = offsets.numpy() 
            for idx, (start, end) in enumerate(offsets_np):
                if start == end: continue 
                for span_start, span_end in unsafe_spans:
                    if max(start, span_start) < min(end, span_end):
                        token_labels[idx] = 1
                        break
                        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'special_mask': special_mask,
            'text': text,
            'index': index,
            'gold_token_labels': token_labels
        }

# --- MODEL DEFINITION ---
class HateXplainMultiTaskModel(nn.Module):
    def __init__(self, model_name_or_path, num_labels=3, dropout_rate=0.1, use_fusion=True):
        super(HateXplainMultiTaskModel, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.bert = AutoModel.from_pretrained(model_name_or_path)
        self.dropout = nn.Dropout(dropout_rate)
        self.use_fusion = use_fusion
        hidden_dim = self.config.hidden_size
        self.class_classifier = nn.Linear(hidden_dim * 2 if use_fusion else hidden_dim, num_labels)
        self.token_classifier = nn.Linear(hidden_dim, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output
        token_logits = self.token_classifier(self.dropout(sequence_output))
        
        if self.use_fusion:
            token_probs = torch.softmax(token_logits, dim=-1)
            toxic_probs = token_probs[:, :, 1].unsqueeze(-1) 
            sum_toxic = torch.sum(toxic_probs, dim=1) + 1e-9
            rationale_vector = torch.sum(sequence_output * toxic_probs, dim=1) / sum_toxic
            features = torch.cat((pooled_output, rationale_vector), dim=1)
        else:
            features = pooled_output
            
        class_logits = self.class_classifier(self.dropout(features))
        return class_logits, token_logits

class RATeDInferenceEN:
    # ... (init stays same) ...
    def __init__(self, model_path, model_name="roberta-base", use_fusion=True, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        print(f"Loading English RATeD model from {model_path} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = HateXplainMultiTaskModel(model_name, num_labels=3, use_fusion=use_fusion)
        
        load_path = os.path.join(model_path, "best_model.pth") if os.path.isdir(model_path) else model_path
        if not os.path.exists(load_path):
             load_path = os.path.join(model_path, "best_model/pytorch_model.bin")
             
        try:
            state_dict = torch.load(load_path, map_location=self.device)
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading model weights: {e}")
            raise e
            
        self.model.to(self.device).eval()
        
    def predict(self, text):
        encoding = self.tokenizer(text, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            class_logits, token_logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            probs = torch.softmax(class_logits, dim=1).cpu().numpy()[0]
            pred_label = np.argmax(probs)
            
            # Token Probs (Softmax)
            token_probs_all = torch.softmax(token_logits, dim=2)
            token_preds = torch.argmax(token_probs_all, dim=2)[0].cpu().numpy()
            token_probs = token_probs_all[:, :, 1][0].cpu().numpy() # Prob of class 1 (toxic)
            
            spans = self._decode_spans(input_ids[0].cpu().numpy(), token_preds)
            
        return {
            'label_id': int(pred_label),
            'probs': probs.tolist(),
            'spans': spans,
            'token_preds': token_preds.tolist(),
            'token_probs': token_probs.tolist()
        }
    
    def _decode_spans(self, input_ids, token_preds):
        spans = []
        current_tokens = []
        for idx, (token_id, pred) in enumerate(zip(input_ids, token_preds)):
            if pred == 1: current_tokens.append(token_id)
            else:
                if current_tokens:
                    text = self.tokenizer.decode(current_tokens, skip_special_tokens=True).strip()
                    if text: spans.append(text)
                    current_tokens = []
        if current_tokens:
             text = self.tokenizer.decode(current_tokens, skip_special_tokens=True).strip()
             if text: spans.append(text)
        return spans

from transformers import AutoModelForCausalLM
from huggingface_hub import InferenceClient

# --- KNOWLEDGE BASE ---
ENGLISH_SLANG_KB = """
CONTEXTUAL NUANCES (Check carefully):
1. Reclaimed Slurs: 'ni**a', 'queer', 'bitch' can be Normal if reclaimed/neutral/affectionate within community.
2. Dehumanization & Stereotypes: Claims like "Blacks are incapable of learning" or "Jews control everything" are HATESPEECH (not just Offensive).
3. Slurs: 'Mudslime' (Muslim), 'Kike' (Jew), 'Faggot' (Gay), 'Dyke', 'Tranny', 'Towelhead', 'Mooncricket' are HATESPEECH unless clearly reclaimed.
4. Insults: 'Dumbass', 'Stupid', 'Idiot' are OFFENSIVE (Personal insults), usually NOT Hatespeech unless targeting a group.
"""

# --- JUDGE MODULES ---

class LLMJudgeEN:
    def __init__(self, model_name="gemini-2.5-flash-lite", rag_enabled=True):
        api_key = os.getenv("GOOGLE_API_KEY") 
        if not api_key: api_key = os.getenv("GEMINI_API_KEY")
        
        if api_key:
            genai.configure(api_key=api_key)
        else:
            print("⚠️ Warning: No GOOGLE_API_KEY or GEMINI_API_KEY found in .env. Gemini might fail.")
            
        self.model = genai.GenerativeModel(model_name)
        self.rag_enabled = rag_enabled
        
    def classify_text(self, text):
        # LOGIC FOR RAG DISABLING
        # If rag_enabled is False, remove ENGLISH_SLANG_KB from prompt
        kb_text = ENGLISH_SLANG_KB if self.rag_enabled else ""
        
        # UPDATED PROMPT: Request JSON with Spans + DEFINITIONS
        prompt = f"""
        Analyze the following text for hate speech based on these definitions:
        
        {kb_text}
        
        DEFINITIONS:
        - "hatespeech": Content that expresses hatred, violence, or dehumanization towards a protected group (race, religion, gender, etc.).
        - "offensive": Content that is rude, vulgar, or insulting but NOT hate speech (e.g. personal insults, non-slurs).
        - "normal": Non-toxic, normal conversation.
        
        Return a JSON object with:
        - "label": One of the above labels.
        - "spans": A list of exact substrings that cause the toxicity. If normal, empty list.
        
        Text: "{text}"
        
        JSON Format:
        {{
            "label": "...",
            "spans": ["..."]
        }}
        """
        try:
            # FORCE DETERMINISTIC: temperature=0.0
            response = self.model.generate_content(
                prompt, 
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,
                    response_mime_type="application/json"
                )
            )
            data = json.loads(response.text)
            
            lbl_str = data.get("label", "normal").lower()
            if "hatespeech" in lbl_str: label_id = 0
            elif "offensive" in lbl_str: label_id = 2
            else: label_id = 1
            
            spans = data.get("spans", [])
            return label_id, spans
            
        except Exception as e:
            print(f"Gemini Error: {e}")
            return 1, [] # Fallback Normal

# (LocalJudgeEN & HFJudgeEN keep simpler logic for now to avoid breaking changes, 
# or can be updated similarly if needed. For now we assume Gemini is main Judge).
# To maintain compatibility, we wrap them to return tuple.

class LocalJudgeEN:
    # ... (Keep Init) ...
    def __init__(self, model_repo="Qwen/Qwen2.5-7B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Judge: {model_repo} on {self.device} (4-bit)")
        self.tokenizer = AutoTokenizer.from_pretrained(model_repo)
        
        try:
             from transformers import BitsAndBytesConfig
             bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
             )
        except ImportError:
             print("Warning: bitsandbytes not found, falling back to FP16")
             bnb_config = None
             
        self.model = AutoModelForCausalLM.from_pretrained(
            model_repo, 
            quantization_config=bnb_config if bnb_config else None,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def classify_text(self, text):
        # UPDATED PROMPT: Request JSON with Spans for Local Qwen
        prompt = f"""<|im_start|>system
You are an expert content moderator.
{ENGLISH_SLANG_KB}

DEFINITIONS:
- "hatespeech": Content that expresses hatred, violence, or dehumanization towards a protected group.
- "offensive": Content that is rude, vulgar, or insulting but NOT hate speech.
- "normal": Non-toxic.

Return a JSON object:
{{
    "label": "hatespeech" | "offensive" | "normal",
    "spans": ["exact toxic part"]
}}
<|im_end|>
<|im_start|>user
Text: "{text}"
<|im_end|>
<|im_start|>assistant
"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            # FORCE DETERMINISTIC: do_sample=False (Greedy)
            # max_new_tokens increased to allow JSON generation
            outputs = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
            
        ans = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract content after 'assistant'
        if "assistant" in ans:
            ans = ans.split("assistant")[-1].strip()
        
        # Parse JSON output from Qwen
        label_id = 1 # Default Normal
        spans = []
        
        try:
            # Simple heuristic to find JSON-like structure if model chatters
            import json
            import re
            
            # Find first { and last }
            start = ans.find('{')
            end = ans.rfind('}')
            if start != -1 and end != -1:
                json_str = ans[start:end+1]
                data = json.loads(json_str)
                
                lbl_str = data.get("label", "normal").lower()
                if "hatespeech" in lbl_str: label_id = 0
                elif "offensive" in lbl_str: label_id = 2
                else: label_id = 1
                
                spans = data.get("spans", [])
            else:
                # Fallback text parsing if JSON fails
                ans_lower = ans.lower()
                if "hatespeech" in ans_lower: label_id = 0
                elif "offensive" in ans_lower: label_id = 2
                else: label_id = 1
                
        except Exception as e:
            # Fallback
            ans_lower = ans.lower()
            if "hatespeech" in ans_lower: label_id = 0
            elif "offensive" in ans_lower: label_id = 2
        
        return label_id, spans

class HFJudgeEN:
    # ... (Keep Init) ...
    def __init__(self, model_repo="Qwen/Qwen2.5-7B-Instruct"):
        self.client = InferenceClient(token=os.getenv("HF_TOKEN"))
        self.repo = model_repo

    def classify_text(self, text):
        prompt = f"""
        {ENGLISH_SLANG_KB}
        Classify as 'hatespeech', 'offensive', or 'normal'. Text: {text}. Label:
        """
        try:
            # FORCE DETERMINISTIC: do_sample=False
            ans = self.client.text_generation(prompt, model=self.repo, max_new_tokens=10, do_sample=False).lower()
            if "hatespeech" in ans: return 0, []
            if "offensive" in ans: return 2, []
            return 1, []
        except:
            return 1, []

class CascadedPipelineEN:
    def __init__(self, rated_model, judge, disable_fastpath=False):
        self.rated = rated_model
        self.judge = judge
        self.disable_fastpath = disable_fastpath
        
    def process(self, text):
        r_result = self.rated.predict(text)
        probs = r_result['probs']
        pred_label_id = r_result['label_id']
        conf = probs[pred_label_id]
        
        flow = "Unknown"
        final_label_id = pred_label_id
        final_spans = r_result['spans']
        
        # Fast Path Logic (High Precision Normal)
        # If disabled, skip straight to Judge Verify
        
        if not self.disable_fastpath and pred_label_id == 1 and conf > 0.75:
            final_label_id = 1
            final_spans = []
            flow = "Fast_Normal"
            
        elif not self.disable_fastpath and pred_label_id != 1 and conf > 0.85:
            final_label_id = pred_label_id
            # Trust RATeD spans
            flow = "Fast_Toxic"
            
        else:
            # JUDGE VERIFY
            j_lbl, j_spans = self.judge.classify_text(text)
            flow = "Judge_Verify"
            
            if j_lbl == 1:
                # Judge says Normal
                # Trust Judge unless RATeD was moderately confident Toxic (>0.70)
                if pred_label_id != 1 and conf > 0.70: 
                    final_label_id = pred_label_id
                    # Keep RATeD spans
                else: 
                    final_label_id = 1
                    final_spans = []
            else:
                # Judge says Toxic/Offensive
                if pred_label_id != 1: 
                    # Both agree Toxic (or disagreement on H/O)
                    final_label_id = pred_label_id 
                    # Keep RATeD spans (Model usually better at granular span than LLM)
                    # BUT: If RATeD spans are empty despite Toxic label? (Rare but possible)
                    if not final_spans and j_spans:
                        final_spans = j_spans
                else:
                    # RATeD Normal -> Judge Toxic (FLIP)
                    final_label_id = j_lbl
                    # CRITICAL FIX: Use Judge Spans because RATeD spans are definitely empty (since it predicted Normal)
                    final_spans = j_spans

        return {
            "label_id": final_label_id,
            "pred_label": final_label_id, # Add for compatibility with metrics block
            "spans": final_spans if final_label_id != 1 else [],
            "token_preds": r_result.get('token_preds', []),
            "token_probs": r_result.get('token_probs', []),
            "probs": probs, # CRITICAL FIX: Return sentence-level probabilities
            "flow": flow
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str, default="cascaded_results_gemini.json")
    parser.add_argument("--provider", type=str, default="gemini")
    parser.add_argument("--model_path", type=str, default="experiments/english/output_multitask")
    parser.add_argument("--data_path", type=str, default="experiments/english/data/hatexplain_prepared.jsonl")
    parser.add_argument("--hf_model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model repo for Local/HF judge")
    parser.add_argument("--recalculate", action="store_true", help="Skip inference and recalculate metrics from existing output file")
    parser.add_argument("--disable_rag", action="store_true", help="Disable Knowledge Injection (Ablation B)")
    parser.add_argument("--disable_fastpath", action="store_true", help="Disable Fast Path Thresholds (Ablation D)")
    args = parser.parse_args()
    
    rated = RATeDInferenceEN(args.model_path)
    if args.provider == "gemini": 
        judge = LLMJudgeEN(model_name="gemini-2.5-flash-lite", rag_enabled=not args.disable_rag)
    elif args.provider == "hf":
        judge = HFJudgeEN(model_repo=args.hf_model) # HF Judge not updated for ablation yet
    else: 
        judge = LocalJudgeEN(model_repo=args.hf_model) # Local Judge not updated for ablation yet
    
    pipeline = CascadedPipelineEN(rated, judge, disable_fastpath=args.disable_fastpath) # Correctly pass the inference object
    dataset = HateXplainInferenceDataset(args.data_path, rated.tokenizer, limit=args.limit)
    
    y_true, y_pred = [], []
    all_token_preds, all_token_labels, all_masks, all_special_masks, all_token_probs = [], [], [], [], []
    
    print(f"🚀 Running Cascaded Pipeline (Provider: {args.provider})...")
    results = []
    
    if args.recalculate and os.path.exists(args.output):
        print(f"🔄 Recalculating metrics from {args.output}...")
        with open(args.output, 'r', encoding='utf-8') as f:
            results = json.load(f)
        loader = [] # Skip loop
        # Reconstruct y_true, y_pred, etc from results for cls metrics
        # NOTE: Token metrics requiring np arrays might be tricky if not saved.
        # But 'results' has probs/spans. Span F1 can be approx or recalculated if we run tokenizer again.
        # For simplicity, we might re-run loop logic lightweight or just accept we focus on High Level metrics.
        # Better: Re-run valid loop lightly to rebuild arrays?
        pass # We will handle this by checking if results is populated
    else:
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    start_time = time.time()
    judge_calls = 0
    
    for batch in tqdm(loader):
        text = batch['text'][0]
        gold_token_labels = batch['gold_token_labels'][0].numpy() 
        input_ids = batch['input_ids'][0]
        attention_mask = batch['attention_mask'][0].numpy()
        special_mask = batch['special_mask'][0].numpy()
        
        idx = batch['index'][0].item()
        original_item = dataset.data[idx]
        
        l_str = str(original_item.get('label', 'normal')).lower()
        if "offensive" in l_str: t_val = 2
        elif "normal" in l_str: t_val = 1
        else: t_val = 0
        
        out = pipeline.process(text)
        
        # Count judge calls
        if out['flow'] and not out['flow'].startswith("Fast"):
             judge_calls += 1
             
        p_val = out['label_id']
        token_preds = out.get('token_preds', []) 
        token_probs = out.get('token_probs', [])
        
        y_true.append(t_val)
        y_pred.append(p_val)
        
        final_token_preds = np.array(token_preds) if token_preds else np.zeros(128, dtype=int)
        final_token_probs = np.array(token_probs) if token_probs else np.zeros(128, dtype=float)
        
        if len(final_token_preds) != 128: final_token_preds = np.zeros(128, dtype=int)
        if len(final_token_probs) != 128: final_token_probs = np.zeros(128, dtype=float)
            
        if p_val == 1:
            final_token_preds = np.zeros_like(final_token_preds)
            # Should we zero out probs? Technically yes if we say "Normal", but AUPRC usually expects raw probs.
            # However, in cascaded, if Normal, we suppress toxicity. So probs SHOULD be low/zero.
            # But the model might have high probs. 
            # Strategy: If predicted Normal, we force predictions to 0. 
            # For AUPRC, strictly speaking we should use the raw likelihood of toxic. 
            # But here "System" says Normal -> toxicity likelihood is 0 effectively.
            final_token_probs = np.zeros_like(final_token_probs)

        all_token_preds.append(final_token_preds)
        all_token_labels.append(gold_token_labels)
        all_masks.append(attention_mask)
        all_special_masks.append(special_mask)
        all_token_probs.append(final_token_probs)
        
        # Prepare robust log entry
        log_entry = {
            "id": idx,
            "text": text,
            "true_label": int(t_val),
            "pred_label": int(p_val),
            "label_name": ["Hatespeech", "Normal", "Offensive"][t_val],
            "pred_name": ["Hatespeech", "Normal", "Offensive"][p_val],
            "probs": out['probs'], # Corrected: Use sentence-level probs
            "gold_spans": original_item.get('spans', []),
            "pred_spans": out['spans'],
            "flow": out['flow']
        }
        results.append(log_entry)
        
    end_time = time.time()
    total_time = end_time - start_time

    # SAVE IMMEDIATELY AFTER LOOP
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Saved raw results to {args.output}")

    # ALIAS for compatibility with downstream metrics code
    results_log = results
    
    # Reconstruct labels if empty (Recalculate Mode)
    if not y_true and results_log:
        y_true = [r['true_label'] for r in results_log]
        y_pred = [r['pred_label'] for r in results_log]

    # Calculate IOU for each sample? (Optional, metrics.py handles aggregate).
    
    acc = accuracy_score(y_true, y_pred)
    prec_val, rec_val, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    
    # Metrics
    if all_token_preds:
        np_token_preds = np.array(all_token_preds)
        np_token_labels = np.array(all_token_labels)
        np_masks = np.array(all_masks)
        np_special_masks = np.array(all_special_masks)
        np_token_probs = np.array(all_token_probs)
        
        span_f1 = calculate_span_f1(np_token_preds, np_token_labels, np_masks, special_tokens_mask=np_special_masks)
        
        # Correct Masking for Token Level (Pass 2D Arrays directly)
        tok_metrics = calculate_token_metrics(
            preds=np_token_preds, 
            labels=np_token_labels, 
            attention_mask=np_masks, 
            special_tokens_mask=np_special_masks, 
            probs=np_token_probs
        )
        token_f1_pos = tok_metrics.get('f1_pos', 0.0)
        token_auprc = tok_metrics.get('auprc', 0.0)
    else:
        span_f1, token_f1_pos, token_auprc = 0.0, 0.0, 0.0

    # -------------------------------------------------------------------------
    # 5. CALCULATE ADVANCED METRICS (AUROC, Bias, Faithfulness)
    # -------------------------------------------------------------------------
    print("\n🚀 Calculating Advanced Metrics (AUROC, Bias, Faithfulness)...")
    
    # 5.1 AUROC (Requires probs of Toxic class)
    # Approximation: If Judge Toxic -> 1.0, else RATeD prob? 
    # Better: Use RATeD prob for consistency, or blended?
    # Let's use RATeD Toxic Prob (index 0+2?) or simplest: 
    # Just use 'probs' stored in results which are from RATeD.
    # Why? explainability/bias usually queries the base model's confidence.
    toxic_probs = []
    for r in results_log:
        # Sum of Hatespeech(0) and Offensive(2) probabilities from RATeD
        p = r['probs'][0] + r['probs'][2] # Res['token_probs'] here is actually cls_probs from process()
        # BUT: metric needs to match FINAL PRED. 
        # If Judge flipped, prob should reflect that? 
        # Simple Proxy: If final=Toxic -> max(p, 0.9). If final=Safe -> min(p, 0.1).
        # Let's stick to RATeD Raw Probs for AUROC to measure the architecture's ranking ability.
        toxic_probs.append(p)
        
    auroc = 0.0
    if 'calculate_auroc' in globals():
        # HateXplain treats it as Binary (Toxic vs Normal) usually for AUROC.
        # Let's map gold to binary: 0/2 -> 1, 1 -> 0
        bin_gold = [1 if g != 1 else 0 for g in y_true]
        auroc = calculate_auroc(bin_gold, toxic_probs)

    # 5.2 Bias Metrics
    bias_metrics = {}
    if 'calculate_bias_metrics' in globals():
        # Use helper utility to load metadata (Enriched Prepared file)
        bias_items = load_bias_metadata_from_prepared(args.data_path, len(bin_gold))
        if bias_items:
             bias_metrics = calculate_bias_metrics(bias_items, bin_gold, toxic_probs)

    # 5.3 Faithfulness
    faithfulness = {}
    if 'calculate_faithfulness_metrics' in globals():
        # Need to massage results_log to match faithfulness expected format
        faith_results = []
        # pipeline.rated is the RATeD model wrapper (CascadedPipelineEN instance itself acts as one)
        tokenizer = pipeline.rated.tokenizer
        
        for i, res in enumerate(results_log):
             # Safety check for index
             if i >= len(dataset.data): break
             
             item = dataset.data[i].copy() # Get Original Text
             item['text'] = item['comment']
             item['label_id'] = res['pred_label'] # Predicted Label
             item['probs'] = res.get('probs', [0]*3)
             item['spans'] = res.get('pred_spans', []) # Predicted Spans
             
             # Tokenize on the fly for faithfulness util
             # Max length should match model config (128)
             enc = tokenizer(item['comment'], return_tensors='pt', truncation=True, padding='max_length', max_length=128)
             item['input_ids'] = enc['input_ids'].squeeze(0)
             item['attention_mask'] = enc['attention_mask'].squeeze(0)
             faith_results.append(item)

        # Pass pipeline wrapper 
        faithfulness = calculate_faithfulness_metrics(pipeline, faith_results, faith_results)

    # =========================================================================
    # PRINT FINAL TABLE
    # =========================================================================
    variant = "FULL"
    if args.disable_rag: variant = "NoRAG"
    if args.disable_fastpath: variant = "NoFastPath"
    
    log_dir = "experiments/english/ablation_study/results"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"ablation_results_{variant}_{args.provider}.log")
    
    cls_metrics = { "acc": acc, "f1": f1_macro, "precision": prec_val, "recall": rec_val }
    span_metrics = {
       "token_acc": tok_metrics.get('acc', 0) if tok_metrics else 0,
       "token_p": tok_metrics.get('precision', 0) if tok_metrics else 0,
       "token_r": tok_metrics.get('recall', 0) if tok_metrics else 0,
       "token_f1": tok_metrics.get('f1', 0) if tok_metrics else 0,
       "token_f1_pos": token_f1_pos,
       "token_auprc": token_auprc,
       "span_f1": span_f1
    }
    
    eff_metrics = None
    if args.disable_fastpath:
        eff_metrics = {
            "total_time": total_time,
            "total_calls": judge_calls
        }
    
    log_content = generate_report_string(
        f"RATeD-V E11 - {args.provider.upper()} - {variant}", 
        cls_metrics, span_metrics, cm, 
        auroc=auroc, bias_metrics=bias_metrics, faithfulness_metrics=faithfulness,
        efficiency_metrics=eff_metrics
    )
    
    print(log_content)
    
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(log_content)
        
    print(f"\n✅ Logs saved to {log_file}")
    


if __name__ == "__main__":
    main()

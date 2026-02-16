
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import json
import os
import sys
import numpy as np

# Setup path logic
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'experiments', 'vietnamese', 'scripts')))
from model_multitask import PhoBERTMultiTask

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'experiments', 'english')))
# Try importing HateXplain model
try:
    from hatexplain_model import HateXplainMultiTaskModel
except ImportError:
    pass

class ReportDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        text = str(item.get('comment', ''))
        # Generalize label handling if needed, but for now specific to current dataset structure
        # ViHOS: label='unsafe' -> 1
        # HateXplain: label='hatespeech' -> 0, etc. (Handled in loading)
        label = item.get('label_id', 0) 
        
        return {
            'input_ids': self.tokenizer(text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')['input_ids'].flatten(),
            'attention_mask': self.tokenizer(text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class Inferencer:
    def __init__(self, device):
        self.device = device

    def load_vihos_data(self, data_path):
        data = []
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        if item.get('split') == 'test':
                            item['label_id'] = 1 if item.get('label') == 'unsafe' else 0
                            data.append(item)
                    except: pass
        except FileNotFoundError:
            print(f"File not found: {data_path}")
        return data

    def load_hatexplain_data(self, data_path):
        data = []
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        if item.get('split') == 'test':
                            # HateXplain Mapping
                            # 0: Hatespeech, 1: Normal, 2: Offensive
                            lbl = item.get('label')
                            if lbl == 'hatespeech': item['label_id'] = 0
                            elif lbl == 'normal': item['label_id'] = 1
                            else: item['label_id'] = 2
                            data.append(item)
                    except: pass
        except FileNotFoundError:
            print(f"File not found: {data_path}")
        return data

    def infer_vihos(self, model_path, data_path, model_name="vinai/phobert-base-v2"):
        print(f"\n--- Running ViHOS Inference on {self.device} ---")
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            return None

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = PhoBERTMultiTask.from_pretrained(model_name, num_labels=2)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device).eval()

        raw_data = self.load_vihos_data(data_path)
        dataset = ReportDataset(raw_data, tokenizer)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        results = {
            'preds': [], 'labels': [], 'probs': [], 'texts': [],
            'pred_spans': [], 'wc_tokens': []
        }

        with torch.no_grad():
            for batch in tqdm(loader, desc="ViHOS"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Classification
                probs = torch.softmax(outputs['cls_logits'], dim=1)
                preds = torch.argmax(outputs['cls_logits'], dim=1).cpu().numpy()
                
                results['preds'].extend(preds)
                results['labels'].extend(labels.cpu().numpy())
                results['probs'].extend(probs.cpu().numpy())
                
                # Spans
                token_preds = torch.argmax(outputs['token_logits'], dim=2).cpu().numpy()
                input_ids_cpu = input_ids.cpu().numpy()
                batch_text_strs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                results['texts'].extend(batch_text_strs)

                # span decoding logic
                sep_id = tokenizer.sep_token_id
                cls_id = tokenizer.cls_token_id
                pad_id = tokenizer.pad_token_id

                for i in range(len(input_ids)):
                    tok_ids = input_ids_cpu[i]
                    tok_p = token_preds[i]
                    
                    sample_spans = []
                    current_span = []
                    
                    for idx, tid in enumerate(tok_ids):
                        if tid in [sep_id, cls_id, pad_id]: continue
                        word = tokenizer.decode([tid]).strip()
                        if tok_p[idx] == 1:
                            results['wc_tokens'].append(word)
                            current_span.append(word)
                        else:
                            if current_span:
                                sample_spans.append(" ".join(current_span))
                                current_span = []
                    if current_span: sample_spans.append(" ".join(current_span))
                    results['pred_spans'].append(sample_spans)
        
        results['raw_data'] = raw_data
        return results

    def infer_hatexplain(self, model_path, data_path, model_name="roberta-base"):
        print(f"\n--- Running HateXplain Inference on {self.device} ---")
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            return None

        try:
             # Ensure import works, might be defined above
             pass
        except:
             return None

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = HateXplainMultiTaskModel(model_name, num_labels=3)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device).eval()

        raw_data = self.load_hatexplain_data(data_path)
        # Handle custom dataset logic inside loop or wrapping? 
        # For simplicity, reuse simpler loop structure since HX needs custom handling often
        
        # We can reuse ReportDataset if we passed 'label_id' correctly
        dataset = ReportDataset(raw_data, tokenizer, max_len=128)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        results = {
            'preds': [], 'labels': [], 'probs': [], 'texts': [],
            'pred_spans': [], 'wc_tokens': [], 'raw_data': raw_data
        }
        
        # Helper lists for WC
        for item in raw_data:
            # GT spans for WC? No, we want predicted spans for WC usually.
            pass

        with torch.no_grad():
            for batch in tqdm(loader, desc="HateXplain"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # HX Model returns tuple
                logits_tuple = model(input_ids, attention_mask)
                cls_logits = logits_tuple[0]
                token_logits = logits_tuple[1]
                
                probs = torch.softmax(cls_logits, dim=1)
                preds = torch.argmax(cls_logits, dim=1).cpu().numpy()
                
                results['preds'].extend(preds)
                results['labels'].extend(labels.cpu().numpy())
                results['probs'].extend(probs.cpu().numpy())
                
                token_preds = torch.argmax(token_logits, dim=2).cpu().numpy()
                input_ids_cpu = input_ids.cpu().numpy()
                
                # Since HX Raw texts might be long, we might just use the raw_data texts
                # But for alignment, let's trust the loaded order (shuffle=False)
                
                # But we should populate results['texts'] properly aligned
                # The Dataset __getitem__ gets text from 'comment'.
                
                sep_id = tokenizer.sep_token_id
                cls_id = tokenizer.cls_token_id
                pad_id = tokenizer.pad_token_id

                for i in range(len(input_ids)):
                    tok_ids = input_ids_cpu[i]
                    tok_p = token_preds[i]
                    
                    sample_spans = []
                    current_span = []
                    
                    for idx, tid in enumerate(tok_ids):
                        if tid in [sep_id, cls_id, pad_id]: continue
                        word = tokenizer.decode([tid], clean_up_tokenization_spaces=False).strip()
                        if tok_p[idx] == 1:
                            results['wc_tokens'].append(word)
                            current_span.append(word)
                        else:
                            if current_span:
                                sample_spans.append(" ".join(current_span))
                                current_span = []
                    if current_span: sample_spans.append(" ".join(current_span))
                    results['pred_spans'].append(sample_spans)
                    
        # Populate texts from raw_data to ensure quality (decode might produce artifacts)
        results['texts'] = [item['comment'] for item in raw_data]
        
        return results

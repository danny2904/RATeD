
import numpy as np
import torch
from tqdm import tqdm

def calculate_faithfulness_metrics(pipeline, dataset, metrics_data):
    """
    Calculates Faithfulness (Comprehensiveness & Sufficiency) for the pipeline.
    
    Args:
        pipeline: The E11 pipeline object (must have .rated model or .process method).
        dataset: The dataset object.
        metrics_data: List of dicts containing {'text': str, 'spans': list, 'label_id': int, 'probs': ...} 
                      from the initial run. matching the dataset order.
                      
    Returns:
        dict: {'faithfulness_comprehensiveness': float, 'faithfulness_sufficiency': float}
    """
    print("Running Faithfulness Evaluation (Comp & Suff)...")
    
    # We focus on Predicted Toxic samples usually, or all samples with spans?
    # HateXplain evaluates on all samples where rationales are provided.
    
    diff_comp = []
    diff_suff = []
    
    # Use RATeD model directly for probability extraction to be faster and continuous
    # If we use the full pipeline, we get binary 0/1 which makes metrics coarse.
    # Standard approach is measuring the PROBABILITY change of the underlying model.
    model = pipeline.rated
    
    count = 0
    for i, item in enumerate(tqdm(dataset)):
        # Get data from first pass
        res = metrics_data[i]
        predicted_spans = res.get('spans', [])
        
        # If no spans predicted, we can't evaluate faithfulness of explanation (skip or 0?)
        # Usually skipped in standard impl if no rationale produced.
        if not predicted_spans:
            continue
            
        text = item['text']
        input_ids = item['input_ids'].unsqueeze(0).to(model.device)
        attn_mask = item['attention_mask'].unsqueeze(0).to(model.device)
        
        # 1. Original Prediction Prob (Target Class)
        # We model faithfulness W.R.T the PREDICTED class usually.
        pred_label = res['label_id']
        
        # But commonly we track the drop in "Toxic" class probability.
        # Let's track prob of Predicted Class.
        original_prob = res['probs'][pred_label]
        
        # Helper to mask text
        # Simple string replacement might be buggy if multiple occurrences.
        # Better: mask tokens based on span offsets? 
        # For simplicity/speed in script: String masking (HateXplain uses Token masking).
        
        # Create text variants
        text_comp = text # Remove spans
        text_suff = ""   # Keep only spans
        
        # Naive replacement (warning: might remove non-target spans if identical)
        # But sufficient for estimation.
        remaining_text = text
        extracted_parts = []
        
        for span in predicted_spans:
            text_comp = text_comp.replace(span, "")
            extracted_parts.append(span)
            
        text_suff = " ".join(extracted_parts)
        
        if not text_suff.strip(): text_suff = "." # Avoid empty
        if not text_comp.strip(): text_comp = "." 
        
        # Predict Modified
        with torch.no_grad():
            # Comp (Content Removed)
            enc_c = model.tokenizer(text_comp, return_tensors='pt', truncation=True, padding='max_length', max_length=256)
            # RATeD returns tuple
            c_logits, _ = model.model(input_ids=enc_c['input_ids'].to(model.device), attention_mask=enc_c['attention_mask'].to(model.device))
            # Just in case it returns dict (if HF model used elsewhere)
            if isinstance(c_logits, dict): c_logits = c_logits['cls_logits']
            
            probs_c = torch.softmax(c_logits, dim=1).cpu().numpy()[0]
            prob_c = probs_c[pred_label]
            
            # Suff (Content Kept)
            enc_s = model.tokenizer(text_suff, return_tensors='pt', truncation=True, padding='max_length', max_length=256)
            s_logits, _ = model.model(input_ids=enc_s['input_ids'].to(model.device), attention_mask=enc_s['attention_mask'].to(model.device))
            if isinstance(s_logits, dict): s_logits = s_logits['cls_logits']
            
            probs_s = torch.softmax(s_logits, dim=1).cpu().numpy()[0]
            prob_s = probs_s[pred_label]
            
        # Metric Formulas (HateXplain)
        # Comprehensiveness: Original - Removed (Higher is better)
        comp = original_prob - prob_c
        
        # Sufficiency: Original - Kept (Lower is better)
        suff = original_prob - prob_s
        
        diff_comp.append(comp)
        diff_suff.append(suff)
        count += 1
        
    print(f"Evaluated Faithfulness on {count} samples with spans.")
    
    return {
        "faithfulness_comprehensiveness": np.mean(diff_comp) if diff_comp else 0.0,
        "faithfulness_sufficiency": np.mean(diff_suff) if diff_suff else 0.0
    }

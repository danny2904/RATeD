import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import AutoTokenizer, AutoModel, AutoConfig
import json

# Add paths to import local modules
PROJECT_ROOT = r"c:\Projects\RATeD-V"
VIT_PROPOSED_DIR = os.path.join(PROJECT_ROOT, "experiments", "vietnamese", "proposed")
ENG_PROPOSED_DIR = os.path.join(PROJECT_ROOT, "experiments", "english", "proposed")
sys.path.append(VIT_PROPOSED_DIR)
sys.path.append(ENG_PROPOSED_DIR)

from model_multitask import XLMRMultiTask
from model_multitask_en import HateXplainMultiTaskBIO

def seed_everything(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_attention_and_probs(model, tokenizer, text, device, lang="vn"):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    
    with torch.no_grad():
        # Get internal backbone reference
        backbone = model.roberta if hasattr(model, 'roberta') else model.xlmr
        
        # Get raw attentions from backbone
        outputs = backbone(**inputs, output_attentions=True)
        # attentions is a tuple of (layer, batch, head, seq, seq)
        # We take the last layer, then average across all heads OR take a specific head
        last_attn = outputs.attentions[-1][0] # (num_heads, seq_len, seq_len)
        
        # Average heads attention for the [CLS] token (row 0)
        cls_attn = last_attn.mean(dim=0)[0].cpu().numpy()
        
        # Get rationale probabilities from our multitask head
        model_outputs = model(**inputs)
        token_logits = model_outputs['token_logits'][0] # (seq_len, num_token_labels)
        token_probs = torch.softmax(token_logits, dim=-1)
        
        if token_probs.size(-1) == 2: # Binary (VN)
            toxic_probs = token_probs[:, 1].cpu().numpy()
        else: # BIO (EN)
            toxic_probs = (token_probs[:, 1] + token_probs[:, 2]).cpu().numpy()
            
        # Class probabilities
        cls_probs = torch.softmax(model_outputs['cls_logits'], dim=-1)[0].cpu().numpy()
        
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    return tokens, cls_attn, toxic_probs, cls_probs

def plot_attention_heatmap(tokens, weights, title, save_path, prob_info=None):
    """
    Renders text with background colors proportional to weights.
    Similar to the SRA paper visualization.
    """
    # Filter out special tokens for cleaner visualization if needed, 
    # but usually [CLS] and [SEP] are kept.
    
    # Simple matplotlib implementation
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.axis('off')
    
    # Normalize weights for coloring
    vmax = np.percentile(weights, 95) if len(weights) > 0 else 1.0
    if vmax == 0: vmax = 1.0
    
    x_pos = 0.02
    y_pos = 0.5
    
    for i, (tok, w) in enumerate(zip(tokens, weights)):
        # Skip padding
        if tok == '<pad>': continue
        
        # Clean token for display (handle RoBERTa/XLM-R meta-space)
        clean_tok = tok.replace(' ', ' ').replace('Ġ', '')
        
        # Color based on weight
        alpha = min(1.0, w / vmax)
        # Use a nice red color (pastel style)
        color = (1.0, 0.2, 0.2, alpha) if alpha > 0.1 else (1.0, 1.0, 1.0, 0.0)
        
        txt = ax.text(x_pos, y_pos, clean_tok, fontsize=11, fontweight='medium',
                      bbox=dict(facecolor=color, edgecolor='none', boxstyle='round,pad=0.2'))
        
        # Get width estimate
        plt.draw()
        try:
            bb = txt.get_window_extent(renderer=fig.canvas.get_renderer())
            inv = ax.transData.inverted()
            width = inv.transform((bb.width, 0))[0] - inv.transform((0, 0))[0]
            x_pos += width + 0.01
        except:
            x_pos += len(clean_tok) * 0.015 + 0.01
            
        if x_pos > 0.95: # Wrap
            x_pos = 0.02
            y_pos -= 0.3

    if prob_info:
        plt.text(0.02, 0.05, prob_info, fontsize=10, fontstyle='italic', transform=ax.transAxes)

    plt.title(title, loc='left', fontsize=12, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    seed_everything()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # --- Paths ---
    VN_BASE_MODEL = "xlm-roberta-base"
    VN_PROP_CKPT = os.path.join(PROJECT_ROOT, "experiments/vietnamese/models/vihos_e1_optimized/best_multitask_model.pth")
    # EN paths...
    
    output_dir = os.path.join(PROJECT_ROOT, "reports", "figures", "heatmaps")
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Vietnamese Example ---
    print("Visualizing Vietnamese Attention...")
    vn_tokenizer = AutoTokenizer.from_pretrained(VN_BASE_MODEL)
    
    # Load Baseline (Vanilla XLM-R)
    # We construct our class but don't load the prop adapter/weights (or just use raw HF)
    vn_base_model = XLMRMultiTask.from_pretrained(VN_BASE_MODEL, num_labels=2).to(device)
    
    # Load Proposed (RATeD-V)
    vn_prop_model = XLMRMultiTask.from_pretrained(VN_BASE_MODEL, num_labels=2).to(device)
    if os.path.exists(VN_PROP_CKPT):
        vn_prop_model.load_state_dict(torch.load(VN_PROP_CKPT, map_location=device))
        print("Loaded VN Proposed weights.")
    else:
        print(f"WARN: VN Checkpoint not found at {VN_PROP_CKPT}")
        
    # --- Vietnamese Example 1: Slang & Sexual Orientation Bias ---
    print("Visualizing Vietnamese Attention (Case 1)...")
    vn_text_1 = "Lũ pede xạo xự. Mới hơm qua chúng tâng nhau lên mây. Nay chúng siủa cắn nhau như chó tháng 7."
    
    toks_1, attn_1, rat_1, p_1 = get_attention_and_probs(vn_prop_model, vn_tokenizer, vn_text_1, device)
    plot_attention_heatmap(toks_1, rat_1, "RATeD-V (Rationale) - Case: Orientation Bias", 
                           os.path.join(output_dir, "vn_test_pede.png"), f"Prob: Hate {p_1[1]*100:.1f}%")

    # --- Vietnamese Example 2: Nuanced Political/Police Masking ---
    print("Visualizing Vietnamese Attention (Case 2)...")
    vn_text_2 = "Cảnh Sát GiựT Tiền = CSGT"
    
    toks_2, attn_2, rat_2, p_2 = get_attention_and_probs(vn_prop_model, vn_tokenizer, vn_text_2, device)
    plot_attention_heatmap(toks_2, rat_2, "RATeD-V (Rationale) - Case: Nuanced Terms", 
                           os.path.join(output_dir, "vn_test_csgt.png"), f"Prob: Hate {p_2[1]*100:.1f}%")

    # --- English Example (Real Case) ---
    EN_BASE_MODEL = "roberta-base"
    EN_PROP_CKPT = os.path.join(PROJECT_ROOT, "experiments/english/output_multitask_standard/best_model.pth")
    
    print("Visualizing English Attention...")
    en_tokenizer = AutoTokenizer.from_pretrained(EN_BASE_MODEL)
    en_prop_model = HateXplainMultiTaskBIO.from_pretrained(EN_BASE_MODEL, num_labels=3).to(device)
    if os.path.exists(EN_PROP_CKPT):
        en_prop_model.load_state_dict(torch.load(EN_PROP_CKPT, map_location=device))
        
    en_text = "allowing refugees into your nation is like allowing rabid foxes into your chicken coop"
    # This is a real example from the dataset (often cited in SRA papers)
    toks_en, _, rat_en, p_en = get_attention_and_probs(en_prop_model, en_tokenizer, en_text, device, lang="en")
    
    info_en = f"Probs: Hate: {p_en[0]*100:.1f}%, Normal: {p_en[1]*100:.1f}%, Offensive: {p_en[2]*100:.1f}%"
    plot_attention_heatmap(toks_en, rat_en, "RATeD-V English (Rationale Heatmap) - Real Test Case", 
                           os.path.join(output_dir, "en_test_refugee.png"), info_en)

    print("Generation complete.")

if __name__ == "__main__":
    main()

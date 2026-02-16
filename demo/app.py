import gradio as gr
import torch
import torch.nn as nn
import numpy as np
import os
import sys
from transformers import AutoTokenizer, AutoConfig, BitsAndBytesConfig
from models import RATeDMultiTask
from qwen_logic import QwenJudge, parse_judge_response

# --- Configuration ---
PROJECT_ROOT = "."
VN_MODEL_PATH = os.path.join(PROJECT_ROOT, "experiments/vietnamese/models/vihos_e1_optimized/best_multitask_model.pth")
EN_MODEL_PATH = os.path.join(PROJECT_ROOT, "experiments/english/output_multitask_standard/best_model.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Loading ---
def load_models():
    print(f"Loading Models & LLM on {DEVICE}...")
    
    # 1. Stage 1 Models
    vn_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    vn_config = AutoConfig.from_pretrained("xlm-roberta-base", num_labels=2)
    vn_model = RATeDMultiTask(config=vn_config).to(DEVICE)
    if os.path.exists(VN_MODEL_PATH):
        print(f"Loading VN weights from {VN_MODEL_PATH}...")
        vn_model.load_state_dict(torch.load(VN_MODEL_PATH, map_location=DEVICE))
    vn_model.eval()

    en_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    en_config = AutoConfig.from_pretrained("roberta-base", num_labels=3)
    en_model = RATeDMultiTask(config=en_config).to(DEVICE)
    if os.path.exists(EN_MODEL_PATH):
        print(f"Loading EN weights from {EN_MODEL_PATH}...")
        en_model.load_state_dict(torch.load(EN_MODEL_PATH, map_location=DEVICE))
    en_model.eval()

    # 2. Stage 2 Judge (Qwen 7B)
    try:
        qwen_judge = QwenJudge(base_model="Qwen/Qwen2.5-7B-Instruct")
    except Exception as e:
        print(f"Error loading Qwen: {e}. Falling back to Stage 1 only.")
        qwen_judge = None

    return {
        "vn": {"model": vn_model, "tokenizer": vn_tokenizer, "labels": ["Safe (Clean)", "Hate Speech"]},
        "en": {"model": en_model, "tokenizer": en_tokenizer, "labels": ["Hate Speech", "Normal", "Offensive"]},
        "judge": qwen_judge
    }

MODELS = load_models()

def generate_heatmap_html(tokens, weights):
    """Generate HTML string with colored tokens based on weights."""
    html = '<div style="font-family: sans-serif; line-height: 2.2; font-size: 1.1em;">'
    vmax = 0.8 
    
    for tok, w in zip(tokens, weights):
        if tok in ["<s>", "</s>", "<pad>", "<mask>", "<unk>"]: continue
        clean_tok = tok.replace(' ', ' ').replace('Ġ', ' ')
        alpha = 0
        if w > 0.3:
            alpha = min(0.9, (w - 0.3) / (vmax - 0.3))
        bg_color = f"rgba(255, 60, 60, {alpha})" if alpha > 0 else "transparent"
        font_weight = "bold" if alpha > 0.4 else "normal"
        html += f'<span style="background-color: {bg_color}; padding: 2px 5px; border-radius: 4px; margin: 0 3px; font-weight: {font_weight};">{clean_tok}</span>'
    html += '</div>'
    return html

def predict_e11(text, lang):
    m_info = MODELS[lang]
    model = m_info["model"]
    tokenizer = m_info["tokenizer"]
    labels = m_info["labels"]
    judge = MODELS["judge"]
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(**inputs)
        cls_logits = outputs["cls_logits"]
        cls_probs = torch.softmax(cls_logits, dim=-1)[0].cpu().numpy()
        s1_idx = np.argmax(cls_probs)
        s1_conf = cls_probs[s1_idx]
        
        # Gating Thresholds (E11 Logic)
        # Gating Thresholds (Align EXACTLY with research scripts)
        if lang == "vn":
            is_toxic_s1 = (s1_idx == 1) # 0:Clean, 1:Hate
            # 04_run_proposed_cascaded_verify.py Logic:
            # is_judge_needed = not ((is_toxic_rated and confidence > 0.90) or (not is_toxic_rated and confidence >= 0.98))
            needs_judge = not (
                (is_toxic_s1 and s1_conf > 0.90) or 
                (not is_toxic_s1 and s1_conf >= 0.98)
            )
            toxic_prob = cls_probs[1]
            token_logits = outputs["token_logits"][0]
            token_probs = torch.softmax(token_logits, dim=-1)
            toxic_weights = (token_probs[:, 1] + token_probs[:, 2]).cpu().numpy()
            s1_debug_info = f"Stage 1: {labels[s1_idx]} | Probs: Safe={cls_probs[0]:.4f}, Hate={cls_probs[1]:.4f}"
        else:
            is_toxic_s1 = (s1_idx != 1) # 0:Hate, 1:Normal, 2:Offensive
            # 04_cascaded_verify_en.py Logic:
            # is_judge_needed = (0.45 <= confidence <= 0.98)
            # Note: English confidence is 1.0 - prob(Normal)
            toxic_prob = 1.0 - cls_probs[1]
            needs_judge = (0.45 <= toxic_prob <= 0.98)
            token_logits = outputs["token_logits"][0]
            token_probs = torch.softmax(token_logits, dim=-1)
            toxic_weights = (token_probs[:, 0] + token_probs[:, 2]).cpu().numpy()
            s1_debug_info = f"Stage 1: {labels[s1_idx]} | Probs: Hate={cls_probs[0]:.4f}, Normal={cls_probs[1]:.4f}, Off={cls_probs[2]:.4f}"

        print(f"\n[DEMO DEBUG - {lang.upper()}]")
        print(f"Input: {text}")
        print(s1_debug_info)
        print(f"Needs Judge Stage 2? {needs_judge}")

        status = f"✅ Fast Path (Stage 1: {labels[s1_idx]})"
        final_idx = s1_idx
        
        if needs_judge and judge:
            status = "⚖️ Verified by Qwen LLM (Stage 2)"
            judge_res = judge.verify(text, lang)
            print(f"-> Raw Judge Response: {repr(judge_res)}")
            final_idx = parse_judge_response(judge_res, lang)
            
            # --- SAFEGUARD LOGIC (Align with research scripts) ---
            # If Judge says Normal (idx 1) but Stage 1 was Very Confident in Toxicity
            if lang == "en":
                # safeguard_threshold = 0.85 (from CascadedPipelineEN)
                if final_idx == 1 and is_toxic_s1 and toxic_prob >= 0.85:
                    print(f"⚠️ SAFEGUARD TRIGGERED: Overriding Judge's 'Normal' with Stage 1's {labels[s1_idx]}")
                    final_idx = s1_idx
                    status = f"🛡️ Safeguard (Overridden to {labels[final_idx]})"
                else:
                    status = f"⚖️ Verified (Stage 2: {labels[final_idx]})"
            else:
                # Vietnamese currently has no explicit safeguard in snippet 04
                status = f"⚖️ Verified (Stage 2: {labels[final_idx]})"

            # Override probabilities for UI
            new_probs = np.zeros_like(cls_probs)
            new_probs[final_idx] = 0.99
            cls_probs = new_probs

        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        heatmap_html = generate_heatmap_html(tokens, toxic_weights)

    res_dict = {labels[i]: float(cls_probs[i]) for i in range(len(labels))}
    # Return 3 items: Probability Dict (for Label component), Heatmap HTML, Status Text
    return res_dict, heatmap_html, status

# --- Gradio UI Design ---
with gr.Blocks(title="RATeD-V: E11 Cascaded Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🛡️ RATeD-V: E11 Cascaded Pipeline (Qwen Edition)
    **Deep Feature Fusion (Stage 1) + Qwen LLM Reasoning (Stage 2)**
    """)
    
    with gr.Tabs() as tabs:
        with gr.TabItem("Vietnamese (ViHOS)"):
            with gr.Row():
                with gr.Column(scale=2):
                    input_vn = gr.Textbox(label="Nhập nội dung cần kiểm tra", lines=3)
                    btn_vn = gr.Button("Analyze (E11)", variant="primary")
                with gr.Column(scale=1):
                    status_vn = gr.Textbox(label="Execution Flow", interactive=False)
                    label_vn = gr.Label(label="Final Prediction")
            
            gr.Markdown("### 🔍 Explainability Heatmap (from Stage 1 Feature Fusion)")
            html_vn = gr.HTML()

            btn_vn.click(fn=lambda x: predict_e11(x, "vn"), inputs=input_vn, outputs=[label_vn, html_vn, status_vn])

        with gr.TabItem("English (HateXplain)"):
            with gr.Row():
                with gr.Column(scale=2):
                    input_en = gr.Textbox(label="Enter comment here", lines=3)
                    btn_en = gr.Button("Analyze (E11)", variant="primary")
                with gr.Column(scale=1):
                    status_en = gr.Textbox(label="Execution Flow", interactive=False)
                    label_en = gr.Label(label="Final Prediction")
            
            gr.Markdown("### 🔍 Explainability Heatmap (from Stage 1 Feature Fusion)")
            html_en = gr.HTML()

            btn_en.click(fn=lambda x: predict_e11(x, "en"), inputs=input_en, outputs=[label_en, html_en, status_en])


    gr.Markdown("""
    ---
    *Scientific Insight: RATeD-V uses an Attention-based Guided Fusion mechanism to align classification with human-labeled rationales (α=10).*
    """)

if __name__ == "__main__":
    demo.launch(share=False)

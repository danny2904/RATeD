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
PROJECT_ROOT = ".." # Point to parent from demo/
VN_MODEL_PATH = os.path.join(PROJECT_ROOT, "experiments/vietnamese/models/vihos_e1_optimized/best_multitask_model.pth")
EN_MODEL_PATH = os.path.join(PROJECT_ROOT, "experiments/english/output_multitask_standard/best_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Loading ---
def load_models():
    print(f"🚀 Loading RATeD-V Core on {DEVICE}...")
    
    # 1. Stage 1 Models
    vn_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    vn_config = AutoConfig.from_pretrained("xlm-roberta-base", num_labels=2)
    vn_model = RATeDMultiTask(config=vn_config).to(DEVICE)
    if os.path.exists(VN_MODEL_PATH):
        vn_model.load_state_dict(torch.load(VN_MODEL_PATH, map_location=DEVICE))
    vn_model.eval()

    en_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    en_config = AutoConfig.from_pretrained("roberta-base", num_labels=3)
    en_model = RATeDMultiTask(config=en_config).to(DEVICE)
    if os.path.exists(EN_MODEL_PATH):
        en_model.load_state_dict(torch.load(EN_MODEL_PATH, map_location=DEVICE))
    en_model.eval()

    # 2. Stage 2 Judge (Qwen 7B)
    print(f"🚀 Loading Qwen-2.5-7B Judge (4-bit)...")
    try:
        qwen_judge = QwenJudge(base_model="Qwen/Qwen2.5-7B-Instruct", device=DEVICE)
    except Exception as e:
        print(f"Error loading Qwen: {e}")
        qwen_judge = None

    return {
        "vn": {"model": vn_model, "tokenizer": vn_tokenizer, "labels": ["Safe", "Unsafe"]},
        "en": {"model": en_model, "tokenizer": en_tokenizer, "labels": ["Hate Speech", "Normal", "Offensive"]},
        "judge": qwen_judge
    }

MODELS = load_models()

def generate_heatmap_html(tokens, weights):
    """Generate HTML string with colored tokens based on weights."""
    html = '<div style="display: flex; flex-wrap: wrap; gap: 4px; margin-top: 10px;">'
    vmax = 0.8 
    for tok, w in zip(tokens, weights):
        if tok in ["<s>", "</s>", "<pad>", "<mask>", "<unk>"]: continue
        clean_tok = tok.replace(' ', ' ').replace('Ġ', ' ')
        alpha = 0
        if w > 0.3:
            alpha = min(0.9, (w - 0.3) / (vmax - 0.3))
        bg_color = f"rgba(255, 60, 60, {alpha})" if alpha > 0 else "rgba(200, 200, 200, 0.1)"
        color = "white" if alpha > 0.4 else "inherit"
        html += f'<span style="background-color: {bg_color}; padding: 2px 6px; border-radius: 4px; font-size: 0.9em; color: {color}; border: 1px solid rgba(0,0,0,0.1);">{clean_tok}</span>'
    html += '</div>'
    return html

def chat_interface(message, history, lang):
    m_info = MODELS[lang]
    model = m_info["model"]
    tokenizer = m_info["tokenizer"]
    labels = m_info["labels"]
    judge = MODELS["judge"]
    
    inputs = tokenizer(message, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(**inputs)
        cls_probs = torch.softmax(outputs["cls_logits"], dim=-1)[0].cpu().numpy()
        s1_idx = np.argmax(cls_probs)
        s1_conf = cls_probs[s1_idx]
        
        # Gating Logic
        if lang == "vn":
            is_toxic_s1 = (s1_idx == 1)
            needs_judge = not ((is_toxic_s1 and s1_conf > 0.90) or (not is_toxic_s1 and s1_conf >= 0.98))
            toxic_weights = (torch.softmax(outputs["token_logits"][0], dim=-1)[:, 1] + torch.softmax(outputs["token_logits"][0], dim=-1)[:, 2]).cpu().numpy()
            s1_info = f"Stage 1 (RoBERTa): {labels[s1_idx]} ({s1_conf:.2%})"
        else:
            toxic_prob = 1.0 - cls_probs[1]
            danger_keywords = ["asshole", "bitch", "cunt", "nigger", "idiot", "stupid", "fuck", "retard", "hoe"]
            has_danger_word = any(w in message.lower().replace(" ", "") for w in danger_keywords)
            needs_judge = (0.15 <= toxic_prob <= 0.98) or has_danger_word
            toxic_weights = (torch.softmax(outputs["token_logits"][0], dim=-1)[:, 0] + torch.softmax(outputs["token_logits"][0], dim=-1)[:, 2]).cpu().numpy()
            s1_info = f"Stage 1 (RoBERTa): {labels[s1_idx]} ({s1_conf:.2%})"

        final_label = labels[s1_idx]
        judge_reasoning = ""
        flow_status = "⚡ Fast Path (Stage 1)"

        if needs_judge and judge:
            flow_status = "⚖️ Verified by Qwen (Stage 2)"
            raw_res = judge.verify(message, lang)
            # Simple extraction for chat
            lines = raw_res.split("\n")
            for line in lines:
                if "LABEL:" in line.upper(): 
                    j_idx = parse_judge_response(raw_res, lang)
                    final_label = labels[j_idx]
                if "REASONING:" in line.upper():
                    judge_reasoning = line.split(":", 1)[1].strip()
            
            # Safeguard logic for EN
            if lang == "en" and final_label == "Normal" and (s1_idx != 1) and toxic_prob >= 0.85:
                final_label = labels[s1_idx]
                flow_status = "🛡️ Safeguard Overridden"
                judge_reasoning = "LLM tried to downgrade, but S1 confidence was too high."

        # Visual Formatting
        emoji = "✅" if "Normal" in final_label or final_label == "Safe" else "⚠️"
        if "Hate" in final_label or "Unsafe" in final_label or "Offensive" in final_label:
            emoji = "🚫"
        
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        heatmap = generate_heatmap_html(tokens, toxic_weights)
        
        response = f"### {emoji} Prediction: **{final_label.upper()}**\n\n"
        response += f"**Decision Flow:** `{flow_status}`\n\n"
        if judge_reasoning:
            response += f"**💡 AI Analysis:** *{judge_reasoning}*\n\n"
        
        response += f"---\n#### 🔍 Token-level Sensitivity (Heatmap)\n{heatmap}\n\n"
        response += f"<details style='margin-top:10px; border: 1px solid #ddd; padding: 10px; border-radius: 8px;'><summary>🛠️ Technical Deep Dive</summary>\n\n"
        response += f"- **Backbone:** {s1_info}\n"
        response += f"- **Logits:** `{cls_probs.tolist()}`\n"
        response += f"- **Hardware:** {DEVICE}\n"
        response += f"</details>"
        
        return response

# --- Custom CSS for UI-UX Pro Max (DARK MODE) ---
custom_css = """
.gradio-container { background-color: #0f172a !important; color: #f8fafc !important; font-family: 'Inter', sans-serif !important; }
.main-header { text-align: center; margin-bottom: 2rem; color: #6366f1; }
.chatbot-container { border: 1px solid #1e293b !important; border-radius: 12px !important; background: #1e293b !important; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5) !important; }
footer { display: none !important; }
.gr-button-primary { background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%) !important; border: none !important; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2) !important; }
.gr-tabs { border-bottom: 1px solid #334155 !important; }
.gr-tabitem { color: #94a3b8 !important; }
.gr-tabitem.selected { color: #f8fafc !important; border-bottom: 2px solid #6366f1 !important; }
details { background: #1e293b !important; border: 1px solid #334155 !important; color: #cbd5e1 !important; }
/* Heatmap adjustment for dark mode */
.heatmap-token { border: 1px solid rgba(255,255,255,0.1) !important; color: #f8fafc !important; }
"""

# --- GUI Layout ---
with gr.Blocks(title="RATeD: A Rationalized Multitask Learning Framework for Explainable Toxic Expression Detection", theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="purple", neutral_hue="slate"), css=custom_css) as demo:
    # Force dark mode via javascript or theme setting
    demo.load(None, None, None, js="() => document.body.classList.add('dark')")
    
    with gr.Group():
        gr.Markdown("#RATeD: A Rationalized Multitask Learning Framework for Explainable Toxic Expression Detection", elem_classes="main-header")
        gr.Markdown("### *Intelligence Meets Aesthetics* ", elem_classes="main-header")
    
    with gr.Tabs():
        with gr.TabItem("🇬🇧 English"):
            gr.ChatInterface(
                fn=lambda msg, hist: chat_interface(msg, hist, "en"),
                chatbot=gr.Chatbot(height=550, label="RATeD-V EN", show_label=False, elem_classes="chatbot-container"),
                examples=["fuck you", "Those immigrants are cockroaches", "I love this scientific project!"],
                title=None
            )
            
        with gr.TabItem("🇻🇳 Tiếng Việt"):
            gr.ChatInterface(
                fn=lambda msg, hist: chat_interface(msg, hist, "vn"),
                chatbot=gr.Chatbot(height=550, label="RATeD-V VN", show_label=False, elem_classes="chatbot-container"),
                examples=["đồ ngu học", "mẹ con kia đi khách", "hôm nay trời đẹp quá"],
                title=None
            )

    gr.Markdown("""
    <div style="text-align: center; color: #475569; font-size: 0.8em; margin-top: 20px;">
    &copy; 2026 RATeD-V Research Team | Dark Mode Activated
    </div>
    """)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)

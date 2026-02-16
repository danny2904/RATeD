import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import re
import os

class QwenJudge:
    def __init__(self, base_model="Qwen/Qwen2.5-7B-Instruct", device="cuda"):
        self.device = device
        self.base_model_name = base_model
        
        # Paths to your SFT checkpoints (LoRA adapters)
        self.adapters = {
            "en": r"c:\Projects\RATeD-V\experiments\english\models\qwen2.5-7b-hatexplain-sft-3class\checkpoint-8350",
            "vn": r"c:\Projects\RATeD-V\experiments\vietnamese\models\qwen2.5-7b-vihos-sft\checkpoint-1000"
        }
        
        print(f"Loading Base Qwen in 4-bit...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        base_model_obj = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Initialize PeftModel with EN adapter first
        print(f"🚀 Initializing PEFT with EN adapter...")
        self.model = PeftModel.from_pretrained(base_model_obj, self.adapters["en"], adapter_name="en")
        
        # Load VN adapter
        if os.path.exists(self.adapters["vn"]):
            print(f"🚀 Loading VN adapter...")
            self.model.load_adapter(self.adapters["vn"], adapter_name="vn")
        
        self.current_lang = "en"
        self.model.set_adapter("en")
        self.model.eval()

    def _switch_adapter(self, lang):
        if self.current_lang == lang:
            return
        
        if lang in ["en", "vn"]:
            print(f"🔄 Switching to adapter: {lang.upper()}")
            self.model.set_adapter(lang)
            self.current_lang = lang
        else:
            print(f"⚠️ Unknown lang {lang}, keeping current.")

    def verify(self, text, lang="vn"):
        self._switch_adapter(lang)
        self.model.eval()
        
        if lang == "vn":
            # --- VIETNAMESE SPECIALIST PROMPT (Align with 04_run_proposed_cascaded_verify.py) ---
            slang_kb = """LƯU Ý ĐẶC BIỆT VỀ TỪ LÓNG VIỆT NAM (TEENCODE):
1. 'má', 'mẹ', 'bà': 
   - Nếu dùng để gọi người thân hoặc cảm thán nhẹ ("má ơi", "mẹ nó", "bà già"): KHÔNG PHẢI HATE (SAFE).
   - Nếu chửi thề trực tiếp ("đ* má", "con mẹ mày", "mẹ kiếp"): LÀ HATE (TOXIC).
2. 'đồ', 'hàng':
   - Nếu chỉ vật dụng ("đồ chơi", "hàng hóa"): SAFE.
   - Nếu miệt thị ("đồ ngu", "con hàng", "đồ chó"): TOXIC.
3. 'đồng bóng', 'pđ', 'bede', 'bóng cái': Từ lóng miệt thị giới tính (LGBT): TOXIC (HATE).
4. 'bug', 'lag', 'hack': Thuật ngữ game/công nghệ: SAFE.
5. 'phò', 'cave', 'đi khách': Miệt thị phụ nữ hoặc nghề nghiệp nhạy cảm: TOXIC.
6. 'bắc kỳ', 'nam kỳ', 'ba que', 'bò đỏ': Từ phân biệt vùng miền/chính trị cực đoan: TOXIC."""
            
            instruction = f"""Bạn là chuyên gia phân tích ngôn từ thù ghét trên mạng xã hội Việt Nam. Hãy trích xuất chính xác tất cả các từ hoặc cụm từ độc hại, xúc phạm hoặc thù ghét trong văn bản. Nếu văn bản an toàn, chỉ trả về 'SAFE'.

{slang_kb}"""
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": text}
            ]
        else:
            # --- ENGLISH SPECIALIST PROMPT (Align with 04_cascaded_verify_en.py) ---
            slang_kb = """GROUP 1: HATE (EXTREME TARGETED DEHUMANIZATION)
- STRICT CRITERIA: Must target a PROTECTED GROUP (Race, Religion, Sexual Orientation, Disability) AND use dehumanizing metaphors or violent threats.
- SLURS: Only "nigger", "kike", "paki", "coon", "spic", "chink" when used to attack.
- METAPHORS: "vermin", "cockroaches", "animals", "parasites", "virus", "shithole" (targeting groups).

GROUP 2: OFFENSIVE (GENERAL TOXICITY & INSULTS)
- ALL OTHER TOXICITY: Any insult, profanity, or aggressive language that does NOT target a protected identity group specifically.
- INSULTS: "retard", "idiot", "stupid", "pos", "asshole", "trash", "clown", "dumbass", "bitch", "hoe", "cunt".
- AGGRESSIVE: "stfu", "fucking hate you", "get out".
- THEME: "illegal immigrants", "moslem terrorist" (label as OFFENSIVE unless extreme dehumanization is present).

GROUP 3: NORMAL (SAFE & DESCRIPTIVE)
- RECLAIMED: "nigga" (casual AAE).
- NEUTRAL: Descriptive identity words used without malice.
- EMPHASIS: "fucking love this", "holy shit"."""
            
            instruction = f"""You are a content moderator specializing in detecting hate speech for the HateXplain 3-class task.
You must follow the MINIMAL RATIONALE philosophy: Only extract the smallest set of tokens that make the text toxic. Topic words (e.g., 'Immigrants', 'Refugees', 'Jews') are NOT toxic unless directly used as part of a slur.

{slang_kb}

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
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": text}
            ]

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, 
                max_new_tokens=128, 
                temperature=0.1, 
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

def parse_judge_response(response, lang="en"):
    """Strictly extract label from LLM output. Mapping: EN (0:Hate, 1:Normal, 2:Offensive), VN (0:Safe, 1:Hate)"""
    res_upper = response.upper()
    if lang == "en":
        # Search for LABEL: first
        import re
        label_match = re.search(r"LABEL:\s*(HATE|OFFENSIVE|NORMAL)", res_upper)
        if label_match:
            label = label_match.group(1)
            if label == "HATE": return 0
            if label == "NORMAL": return 1
            if label == "OFFENSIVE": return 2
            
        # Fallback to general keyword search
        if "HATE" in res_upper: return 0
        if "OFFENSIVE" in res_upper: return 2
        return 1 # Fallback to normal
    else:
        # Vietnamese logic
        if "SAFE" in res_upper: return 0
        return 1 # Toxic (Hate Speech)

"""
English Specialist Judge SFT Training (Qwen-7B)
Environment: mistral310 (trl 0.15.2, transformers 4.51.1, PyTorch 2.6.0)
"""
import os
# CRITICAL: Disable multiprocessing to prevent Windows subprocess loop
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DATASETS_OFFLINE"] = "0"

# Isolate to GPU 2 (Index 1) for VRAM balancing
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from datasets import load_dataset
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel

def train_specialist_en():
    """
    SFT Training for English Specialist Judge (Qwen-7B).
    Aligned with Vietnamese gold standard (06_sft_qwen_vihos.py) hyperparameters.
    """
    # 0. Clear Cache
    torch.cuda.empty_cache()

    # 1. Load Model & Tokenizer
    max_seq_length = 512
    dtype = None # Auto detection
    load_in_4bit = True

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    # 2. Add LoRA Adapters (r=16, alpha=32 as per VN standard)
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 32,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    # 3. Prompt Template (ChatML)
    prompt_template = """<|im_start|>system
{}<|im_end|>
<|im_start|>user
{}<|im_end|>
<|im_start|>assistant
{}<|im_end|>"""

    def tokenize_func(example):
        text = prompt_template.format(
            example["instruction"],
            example["input"],
            example["output"]
        )
        # padding="max_length" ensures all samples have the same length (512)
        outputs = tokenizer(
            text, 
            truncation=True, 
            max_length=max_seq_length, 
            padding="max_length",
            return_tensors=None # Return flat lists
        )
        outputs["labels"] = outputs["input_ids"][:] # Shallow copy of the list
        return outputs

    # Load English SFT Data (OPTIMIZED 3-CLASS)
    data_path = "experiments/english/data/hatexplain_sft_train_3class_optimized.json"
    dataset = load_dataset("json", data_files={"train": data_path}, split="train")
    
    print("Pre-tokenizing dataset (sequential)...")
    dataset = dataset.map(tokenize_func, num_proc=1, remove_columns=dataset.column_names)

    # 4. Standard Trainer (Bypassing SFTTrainer/trl bugs on Windows)
    from transformers import Trainer, DataCollatorForLanguageModeling
    
    trainer = Trainer(
        model = model,
        train_dataset = dataset,
        tokenizer = tokenizer,
        data_collator = DataCollatorForLanguageModeling(tokenizer = tokenizer, mlm = False),
        args = TrainingArguments(
            num_train_epochs = 5,  # 5 epochs for better learning
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_ratio = 0.1,  # 10% warmup
 
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 10,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "experiments/english/models/qwen2.5-7b-hatexplain-sft-3class",
            report_to = "none",
        ),
    )

    # 5. Train Specialist
    print("Starting English Specialist Judge (Qwen) SFT Training...")
    trainer_stats = trainer.train()

    # 6. Save Specialist Checkpoint
    output_dir = "experiments/english/models/qwen2.5-7b-hatexplain-specialist-3class"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"English Specialist Judge (3-Class) saved to {output_dir}")

if __name__ == "__main__":
    train_specialist_en()

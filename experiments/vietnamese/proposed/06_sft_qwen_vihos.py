from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import os

def train():
    # 0. Clear Cache
    torch.cuda.empty_cache()

    # 1. Load Model & Tokenizer
    max_seq_length = 512
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    # 2. Add LoRA Adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Rank
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 32,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth", # 2x faster
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    # 3. Data Formatting
    prompt_template = """<|im_start|>system
{}<|im_end|>
<|im_start|>user
{}<|im_end|>
<|im_start|>assistant
{}<|im_end|>"""

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs       = examples["input"]
        outputs      = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = prompt_template.format(instruction, input, output)
            texts.append(text)
        return { "text" : texts, }

    dataset = load_dataset("json", data_files={"train": "experiments/vietnamese/data/vihos_sft_train.json"}, split="train")
    # Omitting num_proc defaults to single-process, which is safer on Windows
    dataset = dataset.map(formatting_prompts_func, batched = True)

    # 4. Trainer
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = None,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 100,
            max_steps = 1000, # Approx 1 epoch for 8k samples with BS=8
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 10,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "experiments/vietnamese/models/qwen2.5-7b-vihos-sft",
        ),
    )

    # 5. Train
    print("Starting SFT Training...")
    trainer_stats = trainer.train()

    # 6. Save
    model.save_pretrained("experiments/vietnamese/models/qwen2.5-7b-vihos-specialist")
    tokenizer.save_pretrained("experiments/vietnamese/models/qwen2.5-7b-vihos-specialist")
    print("Model saved to experiments/vietnamese/models/qwen2.5-7b-vihos-specialist")

if __name__ == "__main__":
    train()

# ============================================================================
# AI Security Agent - Stage 3: Reasoning Distillation
# Fine-tuning Llama-3 8B to mimic Expert Teacher Reasoning
# ============================================================================

import torch
import json
import os
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

# ====================
# Configuration
# ====================
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B"
MAX_SEQ_LENGTH = 2048

def load_cot_data(role: str) -> Dataset:
    """Load the specialized CoT data for distillation."""
    file_path = f"d:/Global Talent/AI Security Agent/datasets/distillation/{role}_team_cot.json"
    with open(file_path, "r") as f:
        data = json.load(f)
    
    formatted_data = []
    for item in data:
        # Template designed to separate reasoning from output for better distillation
        text = f"""### Instruction:
{item['instruction']}

### Security Scenario:
{item['input']}

### Expert Reasoning:
{item['output']}"""
        formatted_data.append({"text": text})
    
    return Dataset.from_list(formatted_data)

def run_distillation(role: str):
    """Execution of Stage 3: Reasoning Distillation."""
    print(f"🚀 Starting Stage 3 Distillation for {role.upper()} Team...")
    
    # 1. Load Model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        load_in_4bit = True,
    )

    # 2. Add LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r = 128, # Higher rank for complex reasoning capture
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 32,
        lora_dropout = 0,
        use_gradient_checkpointing = "unsloth",
    )

    # 3. Load Data
    dataset = load_cot_data(role)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    # 4. Trainer Setup
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset["train"],
        eval_dataset = dataset["test"],
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 10,
            num_train_epochs = 2,
            learning_rate = 1e-4,
            fp16 = True,
            logging_steps = 1,
            output_dir = f"outputs/distilled_{role}_team",
            report_to = "none",
        ),
    )

    # 5. Execute
    trainer.train()
    
    model.save_pretrained(f"outputs/distilled_{role}_team")
    tokenizer.save_pretrained(f"outputs/distilled_{role}_team")
    print(f"✅ Stage 3 Distillation Complete for {role.upper()} Team.")

if __name__ == "__main__":
    # Example: Run for Red Team
    if os.path.exists("d:/Global Talent/AI Security Agent/datasets/distillation/red_team_cot.json"):
        run_distillation("red")
    else:
        print("❌ CoT data not found. Run datasets/distillation/generate_cot_data.py first.")

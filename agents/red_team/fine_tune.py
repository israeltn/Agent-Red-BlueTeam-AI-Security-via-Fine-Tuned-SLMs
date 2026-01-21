# ============================================================================
# AI Security Agent - Red Team SLM Fine-tuning Pipeline
# Unsloth + Llama-3 8B for Adversarial Risk Simulation
# ============================================================================

import torch
import json
import os
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel
from huggingface_hub import notebook_login
import pandas as pd
from typing import Dict, List
import random

# ====================
# Configuration
# ====================
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B"
MAX_SEQ_LENGTH = 2048
OUTPUT_DIR = "outputs/red_team_slm"
HF_MODEL_REPO = "israeltn/red-team-risk-slm" # Updated for user repo

def load_red_team_data(file_path: str) -> Dataset:
    """Load and format the Red Team dataset."""
    with open(file_path, "r") as f:
        data = json.load(f)
    
    formatted_data = []
    for item in data:
        text = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{item['instruction']}

### Input:
{item['input']}

### Response:
{item['output']}"""
        formatted_data.append({"text": text})
    
    return Dataset.from_list(formatted_data)

def train_red_team_slm():
    """Main training pipeline for Red Team SLM."""
    
    # 1. Load Data
    print("Loading Red Team dataset...")
    dataset = load_red_team_data("d:/Global Talent/AI Security Agent/datasets/red_team_data.json")
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    # 2. Load Model & Tokenizer
    print(f"Loading base model: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None,
        load_in_4bit = True,
    )

    # 3. Add LoRA Adapters
    print("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = 64,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0, # Optimized for Unsloth
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 42,
    )

    # 4. Configure Trainer
    print("Configuring SFTTrainer...")
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
            warmup_steps = 5,
            num_train_epochs = 1, # Demo epoch
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 42,
            output_dir = OUTPUT_DIR,
        ),
    )

    # 5. Train
    print("🚀 Starting fine-tuning for Red Team SLM...")
    trainer.train()
    
    # 6. Save
    print(f"Saving Red Team SLM to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Push to Hugging Face
    print(f"Pushing model to Hugging Face: {HF_MODEL_REPO}")
    model.push_to_hub(
        HF_MODEL_REPO,
        token=os.getenv("HF_TOKEN"),
        commit_message="Fine-tuned Red Team SLM for AI Security"
    )
    tokenizer.push_to_hub(
        HF_MODEL_REPO,
        token=os.getenv("HF_TOKEN"),
        commit_message="Upload tokenizer"
    )
    
    print("✅ Red Team SLM Training & Export Complete!")

if __name__ == "__main__":
    if os.path.exists("d:/Global Talent/AI Security Agent/datasets/red_team_data.json"):
        train_red_team_slm()
    else:
        print("❌ Dataset not found. Please run datasets/red_team/generate_data.py first.")

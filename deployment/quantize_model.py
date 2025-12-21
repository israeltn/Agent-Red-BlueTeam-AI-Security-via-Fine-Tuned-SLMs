# ============================================================================
# AI Security Agent - Stage 6: Multi-Stage Quantization
# Optimizing SLMs for Sub-100ms Inference in Private Environments
# ============================================================================

import torch
from unsloth import FastLanguageModel

def export_to_quantized_format(model_path: str, output_path: str, method: str = "4bit"):
    """
    Exports the fine-tuned SLM to a quantized format (GGUF or 4bit/8bit QLoRA).
    """
    print(f"📦 Starting Stage 6 Quantization ({method}) for {model_path}...")
    
    # 1. Load the fine-tuned adapter
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = 2048,
        load_in_4bit = True if "4bit" in method else False,
    )

    # 2. Save/Export based on method
    if method == "gguf":
        # Exporting to GGUF for local execution (e.g., llama.cpp)
        model.save_pretrained_gguf(output_path, tokenizer, quantization_method = "q4_k_m")
    elif method == "4bit":
        # Save as 4-bit LoRA adapter
        model.save_pretrained_merged(output_path, tokenizer, save_method = "lora")
    elif method == "merged":
        # Merge adapter into base and save (Full model)
        model.save_pretrained_merged(output_path, tokenizer, save_method = "merged_4bit")

    print(f"✅ Quantization Complete. Model saved to: {output_path}")

if __name__ == "__main__":
    # Example deployment path
    import os
    os.makedirs("d:/Global Talent/AI Security Agent/deployment/models", exist_ok=True)
    
    # Mocking a saved model path for demonstration
    mock_path = "outputs/distilled_red_team"
    if os.path.exists(mock_path):
        export_to_quantized_format(mock_path, "d:/Global Talent/AI Security Agent/deployment/models/red_team_slm_4bit", method="4bit")
    else:
        print("💡 Note: Run the distillation script (Stage 3) first to generate a model for quantization.")

import torch
import sys

def verify_environment():
    print("--- Environment Verification ---")
    
    # Check Python version
    print(f"Python Version: {sys.version}")
    
    # Check PyTorch and CUDA
    print(f"PyTorch Version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Total VRAM: {vram:.2f} GB")
    else:
        print("⚠️ WARNING: CUDA is not available. Fine-tuning Llama-3 with Unsloth requires a GPU.")

    # Check for Unsloth
    try:
        import unsloth
        print(f"Unsloth version: {unsloth.__version__}")
    except ImportError:
        print("❌ ERROR: Unsloth is not installed. Please install it using: pip install unsloth")

    # Check for Transformers and PEFT
    try:
        import transformers
        import peft
        print(f"Transformers version: {transformers.__version__}")
        print(f"PEFT version: {peft.__version__}")
    except ImportError:
        print("❌ ERROR: Transformers or PEFT not installed.")

    print("\nEnvironment check complete.")

if __name__ == "__main__":
    verify_environment()

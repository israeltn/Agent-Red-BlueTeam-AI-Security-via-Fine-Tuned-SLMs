# AI Security SLM - Fine-Tuning Guide

This guide provides step-by-step instructions for fine-tuning Llama-3 8B Specialized Security Models (Red Team and Blue Team) using **Unsloth** and **LoRA**.

---

## 🚀 Option 1: Fine-tuning on Personal PC (Local)

### Prerequisites
*   **GPU**: NVIDIA GPU (RTX 3060+, A-series, or better) with at least 8GB VRAM (16GB+ recommended).
*   **Operating System**: Linux or Windows (via WSL2 recommended for better compatibility).
*   **Driver**: Latest NVIDIA Drivers and CUDA Toolkit (12.1+).

### Step 1: Environment Setup
1.  **Install Dependencies**:
    ```powershell
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    pip install torch transformers datasets trl peft accelerate bitsandbytes xformers
    ```
2.  **Verify Environment**:
    Run the verification script I provided:
    ```powershell
    python verify_setup.py
    ```

### Step 2: Prepare Datasets
Ensure you have generated the latest datasets:
```powershell
python datasets/red_team/generate_data.py
python datasets/blue_team/generate_data.py
```

### Step 3: Run Fine-tuning
Choose which model you want to train:
*   **Blue Team (Defense)**:
    ```powershell
    python agents/blue_team/fine_tune.py
    ```
*   **Red Team (Attacks)**:
    ```powershell
    python agents/red_team/fine_tune.py
    ```

### Step 4: Save & Export
*   The models will be saved automatically to the `outputs/` or script-specified local directory.
*   To push to Hugging Face, ensure your `HF_TOKEN` is set as an environment variable:
    ```powershell
    $env:HF_TOKEN = "your_write_token"
    ```

---

## ☁️ Option 2: Fine-tuning on Google Colab (Recommended)

### Step 1: Open the Notebook
1.  Navigate to the `experiments/` folder in your project.
2.  Upload `fine_tune_llama3.ipynb` to [Google Colab](https://colab.research.google.com/).
3.  Ensure your runtime is set to **GPU** (Runtime -> Change runtime type -> GPU -> T4/A100).

### Step 2: Upload Datasets
1.  Click the **Folder Icon** in the left sidebar of Colab.
2.  Upload `blue_team_data.json` and `red_team_data.json` from your local `datasets/` directory.

### Step 3: Configure Secrets
1.  Click the **Key Icon** (Secrets) in the left sidebar.
2.  Add a new secret named `HF_TOKEN`.
3.  Paste your Hugging Face **Write Token** and enable "Notebook access".

### Step 4: Execute Cells
Run the cells sequentially. The notebook is pre-configured to:
1.  Install high-speed Unsloth kernels.
2.  Load both Red and Blue team datasets.
3.  Perform the training.
4.  Optionally push the final model to your Hugging Face Hub.

---

## 💡 Pro Tips
*   **Out of Memory (OOM)**: If you get a memory error, reduce `per_device_train_batch_size` to 1 in the `fine_tune.py` or script.
*   **Hugging Face Repo**: Update the `HF_MODEL_REPO` variable in the scripts to match your username (e.g., `israeltn/security-slm`).
*   **Monitoring**: You can enable **Weights & Biases** (wandb) for real-time loss tracking by uncommenting the `report_to` lines in the training arguments.

---

## ✅ Summary of Success
Once finished, you should see:
1.  A local folder containing your LoRA adapters.
2.  A new repository on your Hugging Face profile with the fine-tuned model.
3.  The ability to load the model via:
    ```python
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained("your-hf-repo/model-name")
    ```

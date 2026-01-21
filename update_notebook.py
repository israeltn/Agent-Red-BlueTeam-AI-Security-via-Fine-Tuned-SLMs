import json

notebook_path = 'd:/Global Talent/AI Security Agent/experiments/fine_tune_llama3.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 3 (Index 4: Markdown, Index 5: Code)
# Actually let's find the cells by markers or content since indices might shift.

new_cells = []
for cell in nb['cells']:
    if cell['cell_type'] == 'markdown' and '### CELL 3: Stage 2' in ''.join(cell['source']):
        new_cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### CELL 3: Load Final Datasets\n",
                "Upload `blue_team_data.json` and `red_team_data.json` to the Colab environment (click the folder icon on the left -> Upload)."
            ]
        })
        continue
    
    if cell['cell_type'] == 'code' and 'generate_cot_security_data' in ''.join(cell['source']):
        new_cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import json\n",
                "import os\n",
                "from datasets import Dataset\n",
                "\n",
                "def load_local_dataset(file_path):\n",
                "    if not os.path.exists(file_path):\n",
                "        print(f\"⚠️ Warning: {file_path} not found. Please upload it.\")\n",
                "        return []\n",
                "    with open(file_path, 'r') as f:\n",
                "        return json.load(f)\n",
                "\n",
                "# Load both datasets\n",
                "blue_data = load_local_dataset('blue_team_data.json')\n",
                "red_data = load_local_dataset('red_team_data.json')\n",
                "\n",
                "combined_data = blue_data + red_data\n",
                "print(f\"✅ Loaded {len(blue_data)} Blue Team samples\")\n",
                "print(f\"✅ Loaded {len(red_data)} Red Team samples\")\n",
                "print(f\"✅ Total samples: {len(combined_data)}\")"
            ]
        })
        continue

    if cell['cell_type'] == 'code' and 'dataset = Dataset.from_list(formatted_samples)' in ''.join(cell['source']):
        new_cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def format_alpaca_prompt(sample: Dict) -> str:\n",
                "    return f\"\"\"### Instruction:\\n{sample['instruction']}\\n\\n### Input:\\n{sample['input']}\\n\\n### Response:\\n{sample['output']}\"\"\"\n",
                "\n",
                "formatted_samples = [{\"text\": format_alpaca_prompt(s)} for s in combined_data]\n",
                "dataset = Dataset.from_list(formatted_samples).train_test_split(test_size=0.1, seed=42)\n",
                "print(f\"✅ Training samples: {len(dataset['train'])}\")"
            ]
        })
        continue

    if cell['cell_type'] == 'code' and 'HUB_MODEL_ID = \"your-username/slm-security-distilled\"' in ''.join(cell['source']):
        new_cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from google.colab import userdata\n",
                "import os\n",
                "\n",
                "# REPLACE with your Hub username and model name\n",
                "HUB_MODEL_ID = \"your-username/ai-security-slm-llama3\"\n",
                "\n",
                "print(f\"🚀 Saving model locally...\")\n",
                "model.save_pretrained(\"ai-security-slm-local\")\n",
                "tokenizer.save_pretrained(\"ai-security-slm-local\")\n",
                "\n",
                "print(f\"🚀 Pushing model to Hugging Face Hub: {HUB_MODEL_ID}\")\n",
                "# Use Colab Secrets for your token or notebook_login()\n",
                "try:\n",
                "    hf_token = userdata.get('HF_TOKEN')\n",
                "except:\n",
                "    hf_token = None\n",
                "\n",
                "model.push_to_hub(HUB_MODEL_ID, token=hf_token)\n",
                "tokenizer.push_to_hub(HUB_MODEL_ID, token=hf_token)\n",
                "print(\"✅ Save and Push complete!\")"
            ]
        })
        continue

    new_cells.append(cell)

nb['cells'] = new_cells

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")

# Autonomous Red/Blue Team SLM Agent - Project Overview

This project implements an **autonomous dual-agent system** for AI security, powered by fine-tuned Llama-3 8B models.

## 🚀 Key Features

- ⚔️ **Red Team Agent**: Autonomously generates security attacks (Prompt Injection, Jailbreaking).
- 🛡️ **Blue Team Agent**: A fine-tuned classifier for real-time threat detection.
- 🧠 **Agentic Reasoning**: Uses **LangGraph** for stateful multi-agent decision making.
- 🛠️ **Tool Integration**: Agents can invoke security scanners and firewall rules.
- ⚡ **High-Performance API**: FastAPI delivery with sub-100ms latency.

## 📂 Project Structure

- `colab_notebook.py`: The core fine-tuning pipeline using Unsloth and PEFT/LoRA.
- `security_orchestrator.py`: Stateful agent orchestration using **LangGraph**.
- `security_tools.py`: Collection of executable security tools.
- `red_team_agent.py`: Logic for the autonomous attacker.
- `adversarial_controller.py`: Simple orchestrator for simulation loops.
- `api.py`: Production-ready FastAPI deployment script.

## 🛠️ Getting Started

### 1. Run the Adversarial Simulation
Observe the agents interacting and generating training data:
```bash
python adversarial_controller.py
```

### 2. Deploy the Production API
Start the FastAPI server for real-time inference:
```bash
uvicorn api:app --reload
```

### 3. Test the Security Analysis
Run the benchmark test against the active API:
```bash
python test_api.py
```

## 🔄 Continuous Learning Path

1. **Simulate**: Run the `adversarial_controller.py` to generate `adversarial_data.json`.
2. **Train**: Use the modular functions in `colab_notebook.py` to fine-tune the Blue Team model on the newly discovered attack patterns.
3. **Deploy**: Update the model in `api.py` to include the latest security weights.

### 6. 🐳 Docker Deployment
For production scale, use the provided Docker configuration:
- `deployment/docker/Dockerfile`: Multi-stage build for the FastAPI service.
- `deployment/docker/docker-compose.yml`: Orchestrates the service and sets up the environment.

Run with:
```bash
cd deployment/docker
docker-compose up --build
```

---
**Project Status**: Production-Ready Base | **Model**: Llama-3.1-8B-4bit

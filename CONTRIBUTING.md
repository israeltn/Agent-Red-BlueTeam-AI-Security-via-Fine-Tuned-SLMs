# Contributing to the Autonomous AI Security Agent

We welcome contributions from the global AI and Security communities! To maintain the high technical standard and innovation required for this project, please adhere to the following guidelines.

## 🎯 Our Vision
To build the world's most efficient and autonomous defense layer for AI agents. We value code that is **performant**, **secure**, and **scalable**.

## 🛠️ Technical Standards

To ensure "Innovation and Leadership" in our codebase, we follow these strict standards:

### 1. Python Excellence
- **PEP 8**: All code must adhere to PEP 8 formatting.
- **Type Hinting**: All function signatures MUST include type hints (Python 3.10+ syntax).
- **Asynchronous Programming**: Use `async`/`await` for I/O bound operations (FastAPI, Network Requests).
- **Documentation**: Every class and public method must have a detailed Docstring (Google Style).

### 2. Model Integrity
- Any changes to fine-tuning logic in `agents/blue_team/` must be benchmarked for both accuracy AND latency.
- We prioritize Small Language Models (SLMs) over larger models to maintain real-time performance.

### 3. Security First
- Never commit API keys or hardcoded tokens.
- All new features must be tested against the Red Team's attack generation loop to ensure no new regressions are introduced.

## 🔄 Development Process

1. **Fork & Branch**: Create a feature branch from `main`.
2. **Commit Messages**: Use conventional commits (e.g., `feat: add RAG poisoning detector`, `fix: resolve OOM during fine-tuning`).
3. **Tests**: Ensure all existing tests in `tests/` pass. Add new tests for new features.
4. **Pull Request**: Provide a detailed description of your changes, including any performance benchmarks or metrics.

## 🤝 Code of Conduct
We are committed to a collaborative, inclusive environment. Leadership means mentoring others—be helpful and professional in code reviews.

---
*By contributing, you agree that your code will be licensed under the project's MIT License.*

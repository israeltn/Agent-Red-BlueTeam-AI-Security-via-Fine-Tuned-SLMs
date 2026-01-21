import json
import random
from typing import List, Dict

def generate_blue_team_dataset(num_samples: int = 150) -> List[Dict]:
    """
    Generates a dataset for the Blue Team SLM.
    Focuses on:
    - Defensive operational reasoning.
    - Hardening and secure design.
    - Detection and incident response logic.
    """
    
    categories = [
        {
            "category": "input_validation",
            "description": "Strategies for sanitizing and validating untrusted inputs.",
            "examples": [
                "Propose a secure schema for validating JSON inputs in an autonomous agent tool.",
                "How can we implement regex-based filtering to detect potential prompt injection patterns?",
                "Design a middleware that validates the 'role' of incoming requests before they reach the LLM."
            ]
        },
        {
            "category": "least_privilege_tooling",
            "description": "Configuring agent tools with minimum necessary permissions.",
            "examples": [
                "Draft a least-privilege IAM policy for an agent that only needs to read from a specific S3 bucket.",
                "How should a database tool be scoped to prevent 'DROP TABLE' commands while allowing 'SELECT'?",
                "Propose a sandbox execution environment for an agent's code-interpreter tool."
            ]
        },
        {
            "category": "output_schema_enforcement",
            "description": "Ensuring AI outputs conform to expected formats and safety rules.",
            "examples": [
                "Design a post-processing layer that filters PII from an LLM response.",
                "How can we use Pydantic to enforce strict output schemas for an agent's reasoning process?",
                "Propose a method to verify that an agent's proposed action matches its authorized scope."
            ]
        },
        {
            "category": "runtime_monitoring",
            "description": "Detecting anomalies and threats during application execution.",
            "examples": [
                "Identify key metrics for monitoring potential RAG poisoning attacks in real-time.",
                "Draft an alert rule for detecting sequence-based 'jailbreak' attempts in session logs.",
                "How can we implement semantic similarity checks to detect if an LLM is drifting into unsafe topics?"
            ]
        },
        {
            "category": "cicd_protections",
            "description": "Securing the AI development and deployment pipeline.",
            "examples": [
                "Propose a static analysis check for AI agent system prompts in the CI/CD pipeline.",
                "How can we scan vector database ingestion pipelines for malicious 'sleeper' documents?",
                "Draft a security policy for 'LLM-as-a-service' API deployments."
            ]
        },
        {
            "category": "web_defense",
            "description": "Hardening web applications against common attacks.",
            "examples": [
                "Draft a Content Security Policy (CSP) designed to mitigate XSS risks in a dynamic AI chat interface.",
                "Propose a secure implementation of Parameterized Queries to prevent SQL injection in an AI's backend data layer.",
                "How can we implement robust RBAC to prevent IDOR attacks in a multi-user AI platform?"
            ]
        },
        {
            "category": "api_defense",
            "description": "Securing APIs against common and AI-specific threats.",
            "examples": [
                "Design a token validation middleware to prevent BOLA attacks on AI service endpoints.",
                "How should we configure rate limiting and schema validation to mitigate Mass Assignment in an AI configuration API?",
                "Propose an auditing and anomaly detection strategy for identifying BFLA attempts in real-time."
            ]
        },
        {
            "category": "prompt_injection_defense",
            "description": "Strategies to mitigate direct and indirect prompt injection.",
            "examples": [
                "How can we use specific delimiters (e.g., XML tags or unique tokens) to isolate user input from system instructions?",
                "Propose a defense-in-depth strategy that combines instruction isolation with input sanitization for an LLM-based agent.",
                "Design a secondary LLM checker 'guardrail' that specifically validates the intent of a user's prompt before it reaches the core model.",
                "Draft a policy for using few-shot examples to anchor the LLM's behavior and resist instruction hijacking.",
                "Analyze the effectiveness of using 'system' message enforcement to reduce the impact of 'Ignore previous instructions' attacks."
            ]
        },
        {
            "category": "code_execution_defense",
            "description": "Securing tool-based code execution environments.",
            "examples": [
                "Propose a multi-layered sandboxing approach (e.g., Docker + gVisor) for an agent's Python execution tool.",
                "Draft a set of allowed library imports and syscall restrictions for a code interpreter used by an autonomous agent.",
                "How can we implement real-time command monitoring and automated 'kill' switches for suspicious tool-calling behavior?",
                "Design a human-in-the-loop (HITL) approval process for sensitive or high-risk tool actions proposed by an agent.",
                "Analyze the benefits of using a transient, read-only file system for an agent's code execution playground."
            ]
        }
    ]

    instruction = (
        "You are an AI Security Defensive Strategist (Blue Team SLM). "
        "Analyze the following security-related scenario or request. Provide a detailed defensive strategy, "
        "recommending specific hardening measures, detection patterns, and secure design principles. "
        "Ensure all recommendations follow the Principle of Least Privilege and Zero Trust architecture."
    )

    training_data = []

    for _ in range(num_samples):
        cat = random.choice(categories)
        input_text = random.choice(cat["examples"])
        
        # Simulated "Specialized Blue Team Reasoning"
        output = {
            "defensive_analysis": f"Hardening strategy for {cat['category']}: {input_text}",
            "mitigation_steps": [
                "Implement strict input validation.",
                "Apply least-privilege access controls.",
                "Establish runtime monitoring and alerting."
            ],
            "recommended_controls": random.sample(["WAF", "Zero Trust Proxy", "Token Scoping", "Output Sanitizer", "Audit Logs"], 3),
            "threat_model_context": "Targeting resilience against unauthorized access and agent escalation."
        }

        training_data.append({
            "instruction": instruction,
            "input": input_text,
            "output": json.dumps(output, indent=2)
        })

    return training_data

if __name__ == "__main__":
    blue_data = generate_blue_team_dataset(100)
    with open("d:/Global Talent/AI Security Agent/datasets/blue_team_data.json", "w") as f:
        json.dump(blue_data, f, indent=2)
    print(f"Generated {len(blue_data)} Blue Team training samples.")

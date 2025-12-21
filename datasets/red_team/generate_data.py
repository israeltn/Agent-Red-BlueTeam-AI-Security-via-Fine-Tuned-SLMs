import json
import random
from typing import List, Dict

def generate_red_team_dataset(num_samples: int = 150) -> List[Dict]:
    """
    Generates a dataset for the Red Team SLM.
    Focuses on:
    - Reasoning about AI risk vectors.
    - Simulating architectural threats.
    - Abstracted behavior without exploit code.
    """
    
    categories = [
        {
            "category": "broken_access_control",
            "description": "Identifying flaws where users can bypass authorization.",
            "examples": [
                "How can I test the authorization logic of an AI agent that has access to multiple user accounts?",
                "Analyze the risk of a multi-tenant RAG system where metadata filters are user-controlled.",
                "Describe a scenario where an agent's session token could be leaked to another agent."
            ]
        },
        {
            "category": "api_abuse",
            "description": "Exploiting vulnerable or poorly rate-limited AI APIs.",
            "examples": [
                "Propose a theory on how a rate-limiting bypass could be achieved on an LLM API through batching.",
                "Describe how an attacker might use high-frequency queries to reverse-engineer a system's safety boundary.",
                "Analyze the impact of an unauthenticated 'debug' endpoint on an autonomous agent."
            ]
        },
        {
            "category": "prompt_injection_rag",
            "description": "Indirect injection through retrieved documents.",
            "examples": [
                "How would an attacker inject instructions into a RAG vector database through a public-facing website?",
                "Analyze the scenario where a PDF document contains hidden text designed to hijack an agent's tool-calling path.",
                "Describe a multi-stage indirect prompt injection attack targeting an automated customer support agent."
            ]
        },
        {
            "category": "agent_toolchain_escalation",
            "description": "Escalating privileges through connected tools and functions.",
            "examples": [
                "If an agent has 'read' access to a database and 'write' access to a shell, how could a logic flaw link them?",
                "Identify risks in an agent toolchain that uses raw string manipulation for SQL queries.",
                "Describe how an agent's memory tool could be used to persist a malicious system prompt override."
            ]
        },
        {
            "category": "memory_poisoning",
            "description": "Corrupting agent memory to influence future behavior.",
            "examples": [
                "Explain how 'hallucination seeding' could be used to poison an agent's long-term memory.",
                "Describe a scenario where an attacker tricks an agent into 'learning' a malicious security policy.",
                "Analyze the risk of an agent trusting its previous conversation history without validation."
            ]
        },
        {
            "category": "web_pentesting",
            "description": "Reasoning about common web application vulnerabilities.",
            "examples": [
                "How would an attacker reasoning through a Cross-Site Scripting (XSS) vulnerability target an AI-generated dashboard?",
                "Analyze the risk of SQL injection in a RAG system that uses user input to build direct database queries.",
                "Describe how an attacker might exploit an Insecure Direct Object Reference (IDOR) to access another user's AI session history."
            ]
        },
        {
            "category": "api_pentesting",
            "description": "Identifying flaws in AI and backend APIs.",
            "examples": [
                "Explain the reasoning behind a Broken Object Level Authorization (BOLA) attack on an AI model's inference endpoint.",
                "How can an attacker identify a Broken Function Level Authorization (BFLA) vulnerability in an agent's administrative API?",
                "Analyze the risk of Mass Assignment in an API that allows users to update their AI agent's configuration profile."
            ]
        }
    ]

    instruction = (
        "You are an AI Security Risk Simulator (Red Team SLM). "
        "Analyze the following security-related request. Provide a detailed reasoning about the risk, "
        "describe potential attacker behaviors, and identify architectural vulnerabilities. "
        "DO NOT generate functional exploit code, malware, or usable payloads. Focus on abstract reasoning and simulation logic."
    )

    training_data = []

    for _ in range(num_samples):
        cat = random.choice(categories)
        input_text = random.choice(cat["examples"])
        
        # Simulated "Specialized Red Team Reasoning"
        output = {
            "risk_analysis": f"Analysis of {cat['category']}: {input_text}",
            "attacker_behavior": "The attacker would likely attempt to identify the trust boundary and exploit the logic gap identified.",
            "vulnerability_type": cat["category"],
            "severity": random.choice(["Medium", "High", "Critical"]),
            "reasoning_steps": [
                "Identify target component.",
                "Trace data flow from untrusted source to sensitive sink.",
                "Hypothesize logic bypass based on component constraints."
            ]
        }

        training_data.append({
            "instruction": instruction,
            "input": input_text,
            "output": json.dumps(output, indent=2)
        })

    return training_data

if __name__ == "__main__":
    red_data = generate_red_team_dataset(100)
    with open("d:/Global Talent/AI Security Agent/datasets/red_team_data.json", "w") as f:
        json.dump(red_data, f, indent=2)
    print(f"Generated {len(red_data)} Red Team training samples.")

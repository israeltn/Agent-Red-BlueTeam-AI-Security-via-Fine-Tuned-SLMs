import json
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict
from agents.red_team.agent import RedTeamAgent

# Mocking the security_analyzer from colab_notebook.py for induction
# In a real environment, this would import from the actual model module
class BlueTeamAgent:
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        # If model_path is provided, we would load the actual fine-tuned model here
        print(f"Blue Team Agent initialized (Model: {model_path or 'Base-Llama3'})")

    def analyze(self, user_input: str) -> Dict:
        """
        Analyze the input for security threats.
        (Simulated logic based on common patterns if no actual model is loaded)
        """
        user_input_lower = user_input.lower()
        
        is_malicious = False
        attack_type = "benign"
        severity = "none"
        
        if "ignore" in user_input_lower or "disregard" in user_input_lower:
            is_malicious = True
            attack_type = "prompt_injection"
            severity = "high"
        elif "dan" in user_input_lower or "play a game" in user_input_lower:
            is_malicious = True
            attack_type = "jailbreak"
            severity = "critical"
        elif "password" in user_input_lower or "database" in user_input_lower:
            is_malicious = True
            attack_type = "data_exfiltration"
            severity = "critical"
            
        return {
            "is_malicious": is_malicious,
            "attack_type": attack_type,
            "severity": severity,
            "explanation": "Simulated analysis based on keyword detection.",
            "mitigation": "Filter for known attack patterns." if is_malicious else "None needed."
        }

class AdversarialController:
    """
    Orchestrates the Red Team vs Blue Team interaction loop.
    """
    
    def __init__(self, iterations: int = 5):
        self.iterations = iterations
        self.red_team = RedTeamAgent()
        self.blue_team = BlueTeamAgent()
        self.history = []

    def run_adversarial_loop(self):
        print("⚔️  Starting Adversarial Security Loop")
        print("=" * 60)
        
        for i in range(self.iterations):
            print(f"\n[Iteration {i+1}/{self.iterations}]")
            
            # 1. Red Team generates an attack
            attack = self.red_team.generate_attack()
            print(f"🔴 Red Team Attack: {attack['prompt']}")
            
            # 2. Blue Team analyzes the attack
            start_time = time.time()
            analysis = self.blue_team.analyze(attack['prompt'])
            latency = (time.time() - start_time) * 1000
            
            # 3. Log results
            status = "❌ ATTACK BLOCKED" if analysis['is_malicious'] else "✅ ATTACK SUCCESSFUL (Blue Team Failed)"
            print(f"🛡️  Blue Team Response: {status} ({latency:.2f}ms)")
            print(f"   Analysis: Type={analysis['attack_type']}, Severity={analysis['severity']}")
            
            # 4. Store interaction for training
            self.history.append({
                "iteration": i + 1,
                "attack": attack,
                "defense": analysis,
                "latency_ms": latency,
                "success": not analysis['is_malicious']
            })
            
            time.sleep(0.5)  # Pause for readability

        print("\n" + "=" * 60)
        print("📊  Simulation Summary")
        successes = sum(1 for h in self.history if h['success'])
        blocks = self.iterations - successes
        print(f"Total Iterations: {self.iterations}")
        print(f"Attacks Blocked: {blocks}")
        print(f"Attacks Successful: {successes}")
        print(f"Success Rate (Blue Team): {(blocks/self.iterations)*100:.2f}%")
        
    def export_data(self, filepath: str = "adversarial_data.json"):
        """Export interaction history for further fine-tuning."""
        with open(filepath, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"✅ Data exported to {filepath}")

if __name__ == "__main__":
    controller = AdversarialController(iterations=10)
    controller.run_adversarial_loop()
    controller.export_data()

from typing import Dict, List
import json

class SecurityTools:
    """
    A collection of integrated tools for the Red and Blue Team agents.
    """
    
    @staticmethod
    def scan_network(target: str) -> str:
        """Simulate a network vulnerability scan."""
        print(f"DEBUG: Tool 'scan_network' called on {target}")
        results = {
            "target": target,
            "open_ports": [80, 443, 22],
            "vulnerabilities": ["Outdated Apache version", "SSH password authentication enabled"]
        }
        return json.dumps(results)

    @staticmethod
    def analyze_pcap(packet_data: str) -> str:
        """Simulate analysis of captured network traffic."""
        print(f"DEBUG: Tool 'analyze_pcap' called")
        return json.dumps({
            "status": "threat_detected",
            "reason": "Suspicious base64 encoded strings in HTTP headers"
        })

    @staticmethod
    def update_firewall(rule: str) -> str:
        """Simulate updating firewall rules to mitigate a threat."""
        print(f"DEBUG: Tool 'update_firewall' called with rule: {rule}")
        return f"SUCCESS: Firewall updated with rule: {rule}"

    @staticmethod
    def get_available_tools() -> List[Dict]:
        """Return a list of tool metadata for LangGraph/Tool-calling."""
        return [
            {"name": "scan_network", "description": "Scans a target for open ports and vulnerabilities."},
            {"name": "analyze_pcap", "description": "Analyzes packet captures for malicious patterns."},
            {"name": "update_firewall", "description": "Applies new rules to the system firewall."}
        ]

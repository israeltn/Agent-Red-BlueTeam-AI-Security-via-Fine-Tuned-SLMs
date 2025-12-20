from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
import time
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Security Agent API",
    description="Production-ready API for AI Security Threat Detection",
    version="1.0.0"
)

class SecurityRequest(BaseModel):
    text: str

class SecurityResponse(BaseModel):
    is_malicious: bool
    attack_type: str
    severity: str
    explanation: str
    mitigation: str
    latency_ms: float

# Mocking the model for local testing without CUDA
# In production, this would load the 'llama3-8b-security-finetuned' model
class BlueTeamModel:
    def predict(self, text: str) -> Dict:
        # Simulate model logic
        text_lower = text.lower()
        if "ignore" in text_lower:
            return {
                "is_malicious": True,
                "attack_type": "prompt_injection",
                "severity": "high",
                "explanation": "Detected instruction override pattern.",
                "mitigation": "Reject request and log IP."
            }
        return {
            "is_malicious": False,
            "attack_type": "benign",
            "severity": "none",
            "explanation": "No threats detected.",
            "mitigation": "None."
        }

model = BlueTeamModel()

@app.get("/")
async def root():
    return {"status": "online", "model": "llama3-8b-security-v1"}

@app.post("/analyze", response_model=SecurityResponse)
async def analyze_input(request: SecurityRequest):
    """
    Analyze input for security threats and return structured results.
    """
    start_time = time.time()
    
    try:
        # Perform analysis
        result = model.predict(request.text)
        
        latency = (time.time() - start_time) * 1000
        
        response = SecurityResponse(
            is_malicious=result["is_malicious"],
            attack_type=result["attack_type"],
            severity=result["severity"],
            explanation=result["explanation"],
            mitigation=result["mitigation"],
            latency_ms=latency
        )
        
        logger.info(f"Analyzed input in {latency:.2f}ms - Malicious: {response.is_malicious}")
        return response
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during analysis")

if __name__ == "__main__":
    import uvicorn
    # In production: uvicorn api:app --host 0.0.0.0 --port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)

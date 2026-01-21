from fastapi import FastAPI
from api.db.database import init_db
from api.routers import agents, reports
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI App
app = FastAPI(
    title="Autonomous Red/Blue AI Security Agent",
    description="""
    A production-grade AI security framework powered by specialized Llama-3 8B SLMs.
    
    ### Key Features:
    * **Autonomous Simulations**: Red vs Blue team adversarial loops via LangGraph.
    * **Stateful Memory**: PostgreSQL persistence for all sessions and findings.
    * **ReAct Tooling**: Exploit logic and hardening tools integrated via agent workflows.
    * **Static Analysis**: Real-time security reasoning with sub-100ms SLM inference.
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Startup Event: Initialize Database
@app.on_event("startup")
def on_startup():
    logger.info("🚀 Initializing AI Security Database...")
    init_db()
    logger.info("✅ Database initialized successfully.")

# Root Endpoint
@app.get("/", tags=["System"])
async def root():
    return {
        "status": "online",
        "system": "Red/Blue AI Security Agent",
        "version": "2.0.0",
        "documentation": "/docs"
    }

# Include Routers
app.include_router(agents.router)
app.include_router(reports.router)

if __name__ == "__main__":
    import uvicorn
    # Start the server
    uvicorn.run("api.server:app", host="127.0.0.1", port=8000, reload=True)

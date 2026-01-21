from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select
from typing import List, Dict
import uuid
from api.db.database import get_session
from api.db.models import SecuritySession, AgentMemory, SecurityReport
from api.agents.graph import create_security_graph
from langchain_core.messages import HumanMessage
import json

router = APIRouter(prefix="/agents", tags=["Agentic Workflow"])
graph = create_security_graph()

@router.post("/simulate", response_model=Dict)
async def start_simulation(scenario: str, session: Session = Depends(get_session)):
    """
    Start an autonomous Red/Blue team simulation for a given security scenario.
    """
    # 1. Create DB Session record
    session_uuid = str(uuid.uuid4())
    db_session = SecuritySession(session_id=session_uuid)
    session.add(db_session)
    session.commit()
    session.refresh(db_session)
    
    # 2. Add initial human message to memory
    human_memory = AgentMemory(
        session_id=db_session.id,
        role="human",
        content=scenario
    )
    session.add(human_memory)
    
    # 3. Run LangGraph Workflow
    inputs = {
        "messages": [HumanMessage(content=scenario)],
        "session_id": session_uuid
    }
    
    try:
        result = graph.invoke(inputs)
        
        # 4. Save Agentic outputs to DB
        red_report = result.get('red_report', {})
        blue_mitigation = result.get('blue_mitigation', {})
        
        # Save memories from Graph
        for msg in result.get('messages', []):
            if isinstance(msg, HumanMessage): continue
            mem = AgentMemory(
                session_id=db_session.id,
                role="agent",
                content=msg.content
            )
            session.add(mem)
            
        # 5. Generate and save final report
        report = SecurityReport(
            session_id=db_session.id,
            title=f"Simulation: {scenario[:30]}...",
            red_findings=json.dumps(red_report),
            blue_mitigation=json.dumps(blue_mitigation),
            risk_level=result.get('risk_level', 'medium')
        )
        session.add(report)
        session.commit()
        
        return {
            "session_id": session_uuid,
            "status": "completed",
            "findings": red_report,
            "mitigation": blue_mitigation,
            "risk_level": report.risk_level
        }
        
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")

@router.get("/history/{session_id}", response_model=List[Dict])
async def get_session_history(session_id: str, session: Session = Depends(get_session)):
    """Retrieve the conversation history and reasoning for a specific session."""
    statement = select(SecuritySession).where(SecuritySession.session_id == session_id)
    db_session = session.exec(statement).first()
    
    if not db_session:
        raise HTTPException(status_code=404, detail="Session not found")
        
    return [
        {"role": m.role, "content": m.content, "timestamp": m.timestamp}
        for m in db_session.memories
    ]

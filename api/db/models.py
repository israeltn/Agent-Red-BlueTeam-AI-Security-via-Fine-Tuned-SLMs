from typing import Optional, List
from sqlmodel import SQLModel, Field, Relationship
from datetime import datetime
import json

class SecuritySession(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(index=True, unique=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    memories: List["AgentMemory"] = Relationship(back_populates="session")
    reports: List["SecurityReport"] = Relationship(back_populates="session")

class AgentMemory(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: int = Field(foreign_key="securitysession.id")
    role: str # 'red', 'blue', 'system'
    content: str
    thought_process: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    session: SecuritySession = Relationship(back_populates="memories")

class SecurityReport(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: int = Field(foreign_key="securitysession.id")
    title: str
    red_findings: str # JSON string of list
    blue_mitigation: str # JSON string of list
    risk_level: str # 'low', 'medium', 'high', 'critical'
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    session: SecuritySession = Relationship(back_populates="reports")

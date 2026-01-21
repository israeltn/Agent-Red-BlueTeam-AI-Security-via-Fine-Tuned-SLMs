from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select
from typing import List, Dict
from api.db.database import get_session
from api.db.models import SecurityReport, SecuritySession

router = APIRouter(prefix="/reports", tags=["Security Reports"])

@router.get("/", response_model=List[Dict])
async def list_reports(session: Session = Depends(get_session)):
    """List all generated security reports."""
    reports = session.exec(select(SecurityReport)).all()
    return [
        {
            "id": r.id,
            "title": r.title,
            "risk_level": r.risk_level,
            "created_at": r.created_at
        }
        for r in reports
    ]

@router.get("/{report_id}", response_model=Dict)
async def get_report_details(report_id: int, session: Session = Depends(get_session)):
    """Get detailed findings and mitigations for a specific report."""
    report = session.get(SecurityReport, report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
        
    return {
        "id": report.id,
        "title": report.title,
        "risk_level": report.risk_level,
        "findings": report.red_findings,
        "mitigation": report.blue_mitigation,
        "created_at": report.created_at
    }

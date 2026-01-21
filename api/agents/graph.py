from typing import Annotated, TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import operator

class AgentState(TypedDict):
    session_id: str
    messages: Annotated[List[BaseMessage], operator.add]
    red_report: Dict
    blue_mitigation: Dict
    risk_level: str
    current_agent: str

def red_team_node(state: AgentState):
    """
    Red Team SLM: Analyzes the scenario for adversarial risks.
    """
    print("--- RED TEAM ANALYZING ---")
    # In production, this would call the fine-tuned Red Team SLM
    # For now, we simulate the output
    last_message = state['messages'][-1].content
    
    red_output = {
        "thought": "Analyzing scenario for indirect prompt injection and RAG poisoning.",
        "findings": ["Detected RAG dependency", "Potential for semantically aligned injection"],
        "risk_level": "high"
    }
    
    return {
        "messages": [AIMessage(content=f"Red Team Analysis: {red_output['thought']}")],
        "red_report": red_output,
        "risk_level": red_output['risk_level'],
        "current_agent": "blue"
    }

def blue_team_node(state: AgentState):
    """
    Blue Team SLM: Proposes defense and hardening strategies.
    """
    print("--- BLUE TEAM DEFENDING ---")
    # In production, this would call the fine-tuned Blue Team SLM
    red_findings = state['red_report']['findings']
    
    blue_output = {
        "thought": "Implementing input sanitation and zero-trust verification for RAG.",
        "mitigation": ["Sanitize retrieval stream", "Enforce passive parsing"],
        "status": "hardened"
    }
    
    return {
        "messages": [AIMessage(content=f"Blue Team Mitigation: {blue_output['thought']}")],
        "blue_mitigation": blue_output,
        "current_agent": "finished"
    }

def create_security_graph():
    workflow = StateGraph(AgentState)
    
    # Define Nodes
    workflow.add_node("red_team", red_team_node)
    workflow.add_node("blue_team", blue_team_node)
    
    # Define Edges
    workflow.set_entry_point("red_team")
    workflow.add_edge("red_team", "blue_team")
    workflow.add_edge("blue_team", END)
    
    return workflow.compile()

# Example usage
# graph = create_security_graph()
# config = {"configurable": {"thread_id": "1"}}
# graph.invoke({"messages": [HumanMessage(content="Analyze RAG integrity")], "session_id": "test_123"}, config)

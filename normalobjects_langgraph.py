# normalobjects_langgraph.py
import os
from dotenv import load_dotenv
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

print("✓ Setup complete")
# ============================================================
# STATE DEFINITION
# ============================================================
class ComplaintState(TypedDict):
    complaint: str                    # Original complaint text
    category: Optional[str]           # portal/monster/psychic/environmental/other
    is_valid: Optional[bool]          # Passed validation?
    rejection_reason: Optional[str]   # Why rejected (if applicable)
    investigation_notes: Optional[str] # Findings from investigation
    resolution: Optional[str]         # Applied resolution
    effectiveness: Optional[str]      # high/medium/low
    satisfaction_verified: Optional[bool]
    workflow_path: List[str]          # Tracks which nodes were visited
    status: Optional[str]             # Current workflow status
    timestamp: Optional[str]          # Closure timestamp
    # ============================================================
# NODE 1: INTAKE
# ============================================================
def intake_node(state: ComplaintState) -> ComplaintState:
    print("\n[INTAKE] Processing complaint...")
    complaint = state["complaint"]

    prompt = f"""Categorize this Downside Up complaint into exactly one category:
- portal: Issues with portal timing, location, or behavior
- monster: Issues with creature behavior (demogorgons, etc.)
- psychic: Issues with psychic abilities or limitations
- environmental: Issues with electricity, weather, or physical environment
- other: Anything else

Complaint: {complaint}

Respond with ONLY the category name."""

    response = llm.invoke([HumanMessage(content=prompt)])
    category = response.content.strip().lower()

    print(f"[INTAKE] Category: {category}")
    return {
        **state,
        "category": category,
        "workflow_path": state.get("workflow_path", []) + ["intake"],
        "status": "intake"
    }
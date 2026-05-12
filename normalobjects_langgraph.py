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
# ============================================================
# NODE 2: VALIDATE
# ============================================================
def validate_node(state: ComplaintState) -> ComplaintState:
    print("\n[VALIDATE] Validating complaint...")
    complaint = state["complaint"]
    category = state["category"]

    prompt = f"""You are a complaint validator for the Downside Up Bureau.

Complaint: "{complaint}"
Assigned category: {category}

Check ONLY the rule for this specific category:
- portal → Is there any mention of timing, schedule, or location of a portal? If yes: VALID
- monster → Is there any mention of creature behavior, actions, or interactions? If yes: VALID
- psychic → Is there any mention of a psychic ability, its limits, or how it works? If yes: VALID
- environmental → Is there any mention of electricity, power lines, weather, or physical phenomena? If yes: VALID
- other → Always: INVALID

Be generous: if the complaint loosely matches the category rule, mark it VALID.

Respond with exactly this format:
VERDICT: VALID or INVALID
REASON: one sentence explanation"""

    response = llm.invoke([HumanMessage(content=prompt)])
    text = response.content.strip()

    verdict = "INVALID"
    reason = "No reason provided"
    for line in text.split("\n"):
        if line.startswith("VERDICT:"):
            verdict = line.replace("VERDICT:", "").strip().upper()
        if line.startswith("REASON:"):
            reason = line.replace("REASON:", "").strip()

    is_valid = verdict == "VALID"
    print(f"[VALIDATE] Result: {verdict} — {reason}")

    return {
        **state,
        "is_valid": is_valid,
        "rejection_reason": None if is_valid else reason,
        "workflow_path": state.get("workflow_path", []) + ["validate"],
        "status": "valid" if is_valid else "rejected"
    }
# ============================================================
# ROUTING: after validate
# ============================================================
def route_after_validation(state: ComplaintState) -> str:
    if state["is_valid"]:
        return "investigate"
    else:
        print(f"\n[REJECTED] {state['rejection_reason']}")
        return END
    # ============================================================
# NODE 3: INVESTIGATE
# ============================================================
def investigate_node(state: ComplaintState) -> ComplaintState:
    print("\n[INVESTIGATE] Gathering evidence...")
    complaint = state["complaint"]
    category = state["category"]

    prompt = f"""You are an investigator at the Downside Up Bureau.

Category: {category}
Complaint: {complaint}

Investigation focus by category:
- portal: Temporal patterns, location consistency, environmental factors
- monster: Behavioral data, interaction patterns, environmental triggers
- psychic: Ability specifications, tested limitations, contextual factors
- environmental: Power line activity, atmospheric conditions, anomaly correlation

Write a short investigation report (3-4 sentences) with documented findings.
Be specific and reference the complaint details."""

    response = llm.invoke([HumanMessage(content=prompt)])
    notes = response.content.strip()

    print(f"[INVESTIGATE] Notes: {notes[:80]}...")
    return {
        **state,
        "investigation_notes": notes,
        "workflow_path": state.get("workflow_path", []) + ["investigate"],
        "status": "investigated"
    }
# ============================================================
# NODE 4: RESOLVE
# ============================================================
def resolve_node(state: ComplaintState) -> ComplaintState:
    print("\n[RESOLVE] Applying resolution...")
    category = state["category"]
    investigation_notes = state["investigation_notes"]

    prompt = f"""You are a resolution specialist at the Downside Up Bureau.

Category: {category}
Investigation findings: {investigation_notes}

Rules:
- Resolution must be specific to the category
- Must reference established Downside Up procedures
- Environmental or monster cases may require specialist team escalation
- End your response with: EFFECTIVENESS: high/medium/low

Write a resolution (3-4 sentences) then rate effectiveness."""

    response = llm.invoke([HumanMessage(content=prompt)])
    full_response = response.content.strip()

    # Extract effectiveness rating
    effectiveness = "medium"  # default
    for line in full_response.split("\n"):
        if "EFFECTIVENESS:" in line.upper():
            rating = line.split(":")[-1].strip().lower()
            if rating in ["high", "medium", "low"]:
                effectiveness = rating

    print(f"[RESOLVE] Effectiveness: {effectiveness}")
    return {
        **state,
        "resolution": full_response,
        "effectiveness": effectiveness,
        "workflow_path": state.get("workflow_path", []) + ["resolve"],
        "status": "resolved"
    }
# ============================================================
# NODE 5: CLOSE
# ============================================================
def close_node(state: ComplaintState) -> ComplaintState:
    print("\n[CLOSE] Closing complaint...")
    from datetime import datetime

    effectiveness = state["effectiveness"]
    followup = effectiveness == "low"

    print(f"[CLOSE] Category: {state['category']}")
    print(f"[CLOSE] Resolution applied ✓")
    print(f"[CLOSE] Satisfaction verified ✓")
    if followup:
        print(f"[CLOSE] ⚠ Low effectiveness → 30-day follow-up scheduled")

    return {
        **state,
        "satisfaction_verified": True,
        "timestamp": datetime.now().isoformat(),
        "workflow_path": state.get("workflow_path", []) + ["close"],
        "status": "closed"
    }
# ============================================================
# BUILD GRAPH
# ============================================================
workflow = StateGraph(ComplaintState)

# Add nodes
workflow.add_node("intake", intake_node)
workflow.add_node("validate", validate_node)
workflow.add_node("investigate", investigate_node)
workflow.add_node("resolve", resolve_node)
workflow.add_node("close", close_node)

# Define edges
workflow.set_entry_point("intake")
workflow.add_edge("intake", "validate")
workflow.add_conditional_edges("validate", route_after_validation, {
    "investigate": "investigate",
    END: END
})
workflow.add_edge("investigate", "resolve")
workflow.add_edge("resolve", "close")
workflow.add_edge("close", END)

# Compile
app = workflow.compile()
print("✓ Graph compiled")

# ============================================================
# TEST COMPLAINTS
# ============================================================
test_complaints = [
    "The Downside Up portal opens at different times each day. How do I predict when?",
    "Demogorgons sometimes work together and sometimes fight. What's their deal?",
    "El can move things with her mind but can't lift heavy rocks. Why?",
    "Why do creatures and power lines react so strangely together?",
    "This is not a valid complaint about something random"
]

print("\n" + "="*60)
print("BLOYCE'S PROTOCOL — COMPLAINT PROCESSOR")
print("="*60)

for i, complaint in enumerate(test_complaints, 1):
    print(f"\n{'='*60}")
    print(f"COMPLAINT {i}: {complaint[:60]}...")
    print("="*60)

    initial_state: ComplaintState = {
        "complaint": complaint,
        "category": None,
        "is_valid": None,
        "rejection_reason": None,
        "investigation_notes": None,
        "resolution": None,
        "effectiveness": None,
        "satisfaction_verified": None,
        "workflow_path": [],
        "status": None,
        "timestamp": None
    }

    result = app.invoke(initial_state)

    print(f"\n── SUMMARY ──")
    print(f"Path:        {' → '.join(result['workflow_path'])}")
    print(f"Category:    {result['category']}")
    print(f"Status:      {result['status']}")
    print(f"Effective:   {result.get('effectiveness', 'N/A')}")
    print(f"Timestamp:   {result.get('timestamp', 'N/A')}")
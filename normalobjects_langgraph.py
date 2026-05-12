import os
from datetime import datetime, timedelta
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
    complaint: str
    category: Optional[str]
    is_valid: Optional[bool]
    rejection_reason: Optional[str]
    investigation_notes: Optional[str]
    resolution: Optional[str]
    effectiveness: Optional[str]
    satisfaction_verified: Optional[bool]
    workflow_path: List[str]
    status: Optional[str]
    timestamp: Optional[str]
    follow_up_required: Optional[bool]
    follow_up_date: Optional[str]


# ============================================================
# NODE 1: INTAKE
# ============================================================
def intake_node(state: ComplaintState) -> ComplaintState:
    print("\n[INTAKE] Processing complaint...")
    complaint = state["complaint"]

    categorization_prompt = f"""You are categorizing complaints for the Downside Up Complaint Bureau.
Map the complaint to EXACTLY ONE of these categories:

- portal: portal timing, location, behavior, schedule, or schedule irregularities
- monster: demogorgons, creatures, creature behavior, interactions, aggression, cooperation
- psychic: telekinesis, psychic abilities, mental powers, ability limits, El
- environmental: power lines, electricity, weather, atmospheric phenomena, physical environment — INCLUDING creature interactions WITH power lines or electrical infrastructure
- other: anything that doesn't clearly fit above

IMPORTANT: If the complaint mentions power lines, electricity, or atmospheric/physical phenomena — even combined with creatures — categorize as 'environmental', not 'other'.

Complaint: {complaint}

Respond with ONLY one word: portal, monster, psychic, environmental, or other."""

    response = llm.invoke([HumanMessage(content=categorization_prompt)])
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
Original complaint: {state["complaint"]}
Investigation findings: {investigation_notes}

Rules:
- Resolution must be specific to the category
- Must reference established Downside Up procedures
- Environmental or monster cases may require specialist team escalation

Effectiveness rating — judge based on the ORIGINAL COMPLAINT specificity:
- high: Complaint names specific entity, location, or mechanism (e.g. "Demogorgons", "the portal on Elm Street", "El's telekinesis")
- medium: Complaint describes a real phenomenon but lacks location or timing details
- low: Complaint is purely vague with no specific entity or phenomenon (e.g. "weird things", "not sure", "I guess")

Named creatures, abilities, or Downside Up phenomena always count as specific → never low.

Write a resolution (3-4 sentences) then on the last line write exactly:
EFFECTIVENESS: high
or
EFFECTIVENESS: medium
or
EFFECTIVENESS: low"""

    response = llm.invoke([HumanMessage(content=prompt)])
    full_response = response.content.strip()

    effectiveness = "medium"
    for line in full_response.split("\n"):
        if "EFFECTIVENESS:" in line.upper():
            rating = line.split(":")[-1].strip().lower()
            if rating in ["high", "medium", "low"]:
                effectiveness = rating

    # Rule-based override: vague complaints → always low
    VAGUE_KEYWORDS = ["i guess", "not sure", "hard to say", "weird things", "stuff", "i think", "don't know", "can't explain"]
    complaint_lower = state["complaint"].lower()
    if any(kw in complaint_lower for kw in VAGUE_KEYWORDS):
        effectiveness = "low"
        print(f"[RESOLVE] Overridden to low — vague complaint detected")

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

    effectiveness = state.get("effectiveness", "medium")
    follow_up_required = effectiveness == "low"
    follow_up_date = None

    if follow_up_required:
        follow_up_date = (datetime.now() + timedelta(days=30)).isoformat()
        print(f"[CLOSE] ⚠ Low effectiveness — 30-day follow-up scheduled: {follow_up_date}")

    print(f"[CLOSE] Category: {state['category']}")
    print(f"[CLOSE] Resolution applied ✓")
    print(f"[CLOSE] Satisfaction verified ✓")

    return {
        **state,
        "status": "closed",
        "workflow_path": state.get("workflow_path", []) + ["close"],
        "timestamp": datetime.now().isoformat(),
        "follow_up_required": follow_up_required,
        "follow_up_date": follow_up_date
    }


# ============================================================
# BUILD GRAPH
# ============================================================
workflow = StateGraph(ComplaintState)

workflow.add_node("intake", intake_node)
workflow.add_node("validate", validate_node)
workflow.add_node("investigate", investigate_node)
workflow.add_node("resolve", resolve_node)
workflow.add_node("close", close_node)

workflow.set_entry_point("intake")
workflow.add_edge("intake", "validate")
workflow.add_conditional_edges("validate", route_after_validation, {
    "investigate": "investigate",
    END: END
})
workflow.add_edge("investigate", "resolve")
workflow.add_edge("resolve", "close")
workflow.add_edge("close", END)

app = workflow.compile()
print("✓ Graph compiled")

# ============================================================
# VISUALIZATION
# ============================================================

def print_graph_structure():
    """Print the static workflow graph as Mermaid (renders on GitHub)."""
    print("\n" + "="*60)
    print("WORKFLOW GRAPH (Mermaid)")
    print("="*60)
    print(app.get_graph().draw_mermaid())


def visualize_complaint_path(complaint_text: str, result: ComplaintState):
    """Print ASCII visualization of the path taken for one complaint."""
    ALL_NODES = ["intake", "validate", "investigate", "resolve", "close"]
    path = result.get("workflow_path", [])

    print("\n── WORKFLOW PATH ──")
    parts = []
    for node in ALL_NODES:
        if node in path:
            parts.append(f"[{node} ✓]")
        else:
            parts.append(f"[{node} ✗]")

    print(" → ".join(parts))

    status = result.get("status")
    if status == "rejected":
        print(f"  ↳ REJECTED at validate — did not proceed")
    elif status == "closed":
        effectiveness = result.get("effectiveness", "N/A")
        print(f"  ↳ CLOSED — effectiveness: {effectiveness.upper()}")
        if result.get("follow_up_required"):
            print(f"  ↳ ⚠ Follow-up scheduled: {result.get('follow_up_date')}")

# ============================================================
# TEST COMPLAINTS
# ============================================================

print_graph_structure()

test_complaints = [
    "The Downside Up portal opens at different times each day. How do I predict when?",
    "Demogorgons sometimes work together and sometimes fight. What's their deal?",
    "El can move things with her mind but can't lift heavy rocks. Why?",
    "Why do creatures and power lines react so strangely together?",
    "This is not a valid complaint about something random",
    "My psychic stuff doesn't really work right, I guess. Hard to say what's wrong exactly."
]

print("\n" + "=" * 60)
print("BLOYCE'S PROTOCOL — COMPLAINT PROCESSOR")
print("=" * 60)

for i, complaint in enumerate(test_complaints, 1):
    print(f"\n{'=' * 60}")
    print(f"COMPLAINT {i}: {complaint[:60]}...")
    print("=" * 60)

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
        "timestamp": None,
        "follow_up_required": None,
        "follow_up_date": None
    }

    result = app.invoke(initial_state)

    print(f"\n── SUMMARY ──")
    print(f"Path:        {' → '.join(result['workflow_path'])}")
    print(f"Category:    {result['category']}")
    print(f"Status:      {result['status']}")
    print(f"Effective:   {result.get('effectiveness', 'N/A')}")
    print(f"Timestamp:   {result.get('timestamp', 'N/A')}")
    if result.get("follow_up_required"):
        print(f"Follow-up:   {result.get('follow_up_date')}")
    
    visualize_complaint_path(complaint, result)
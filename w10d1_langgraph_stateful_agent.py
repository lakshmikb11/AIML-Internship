
"""W10D1: LangGraph stateful agent with conditional routing and HITL."""

from pathlib import Path
from typing import Literal
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class AgentState(TypedDict):
    user_input: str
    classification: str
    needs_review: bool
    human_decision: str
    response: str


def classify(state: AgentState) -> dict:
    """Classify the input using simple, deterministic rules."""
    text = state["user_input"].lower()

    review_keywords = (
        "delete",
        "send an email",
        "make a payment",
        "purchase",
        "approve",
        "share externally",
    )

    needs_review = any(word in text for word in review_keywords)

    if needs_review:
        category = "action"
    elif "?" in text or text.startswith(
        ("what", "why", "how", "when", "where", "who", "explain")
    ):
        category = "question"
    else:
        category = "general"

    print(f"[classify] category={category}, needs_review={needs_review}")

    return {
        "classification": category,
        "needs_review": needs_review,
        "human_decision": "",
        "response": "",
    }


def route(state: AgentState) -> dict:
    """Pause execution and collect human approval for sensitive actions."""
    print("[route] Human review required.")

    decision = interrupt(
        {
            "message": "Review this requested action before proceeding.",
            "user_input": state["user_input"],
            "classification": state["classification"],
            "question": "Type 'approve' or 'reject' to continue.",
        }
    )

    decision = str(decision).strip().lower()
    print(f"[route] Human decision: {decision}")

    return {"human_decision": decision}


def respond(state: AgentState) -> dict:
    """Respond based on the classification and review decision."""
    if state["needs_review"]:
        if state["human_decision"] != "approve":
            response = (
                "Action not approved. No action was performed."
            )
        else:
            response = (
                "Human approval received. The request is approved "
                "for demonstration; no external action was performed."
            )
    elif state["classification"] == "question":
        response = (
            f"I received your question: {state['user_input']}. "
            "This demo graph classified it as a question."
        )
    else:
        response = (
            f"I received your message: {state['user_input']}. "
            "This demo graph classified it as general input."
        )

    print(f"[respond] {response}")
    return {"response": response}


def choose_next(
    state: AgentState,
) -> Literal["route", "respond"]:
    """Route action requests through human review."""
    if state["needs_review"]:
        return "route"
    return "respond"


def build_graph():
    """Build the three-node stateful graph with checkpointing."""
    builder = StateGraph(AgentState)

    builder.add_node("classify", classify)
    builder.add_node("route", route)
    builder.add_node("respond", respond)

    builder.add_edge(START, "classify")

    builder.add_conditional_edges(
        "classify",
        choose_next,
        {
            "route": "route",
            "respond": "respond",
        },
    )

    builder.add_edge("route", "respond")
    builder.add_edge("respond", END)

    return builder.compile(checkpointer=MemorySaver())


def initial_state(user_input: str) -> AgentState:
    return {
        "user_input": user_input,
        "classification": "",
        "needs_review": False,
        "human_decision": "",
        "response": "",
    }


def run_demo() -> None:
    """Run five inputs and demonstrate interrupt/resume."""
    output_dir = Path("output_evidence") / "w10d1"
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_file = output_dir / "run_results.txt"

    lines = [
        "W10D1 LangGraph Stateful Agent — Execution Evidence",
        "=" * 55,
    ]

    graph = build_graph()

    test_inputs = [
        "What is LangGraph?",
        "Explain conditional routing",
        "Hello, agent!",
        "Delete the old project files",
        "Send an email to the team",
    ]

    for index, user_input in enumerate(test_inputs, start=1):
        lines.append(f"\nTest {index}: {user_input}")
        config = {"configurable": {"thread_id": f"w10d1-test-{index}"}}

        if index <= 3:
            result = graph.invoke(initial_state(user_input), config)
            lines.append(f"Classification: {result['classification']}")
            lines.append(f"Review required: {result['needs_review']}")
            lines.append(f"Response: {result['response']}")
        else:
            paused = graph.invoke(initial_state(user_input), config)
            interrupts = paused.get("__interrupt__", ())

            if not interrupts:
                raise AssertionError("Expected a human-review interrupt.")

            lines.append("Status: PAUSED for human review")
            lines.append(f"Interrupt: {interrupts[0].value}")

            decision = "approve" if index == 4 else "reject"
            result = graph.invoke(Command(resume=decision), config)

            lines.append(f"Human decision: {decision}")
            lines.append(f"Response: {result['response']}")

    evidence_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nEvidence saved to: {evidence_file}")


if __name__ == "__main__":
    run_demo()

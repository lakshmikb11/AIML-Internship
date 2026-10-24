
from pathlib import Path
from typing import Literal
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt


# ---------------------------------------------------------
# W10D2: LangGraph State Machines & Conditional Edges
# ---------------------------------------------------------

EVIDENCE_DIR = Path("output_evidence/w10d2")
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
EVIDENCE_FILE = EVIDENCE_DIR / "run_results.txt"


class AgentState(TypedDict, total=False):
    user_input: str
    category: str
    route: str
    response: str
    human_feedback: str



def classify_node(state: AgentState) -> dict:
    """Classify an incoming message using whole-word matching."""
    import re

    message = state["user_input"].lower()
    words = set(re.findall(r"\b\w+\b", message))

    if words.intersection({"urgent", "emergency", "critical"}):
        category = "urgent"
    elif words.intersection({"error", "bug", "python", "code"}):
        category = "technical"
    elif words.intersection({"hello", "hi", "hey"}):
        category = "greeting"
    else:
        category = "general"

    return {"category": category}



def choose_after_classify(
    state: AgentState,
) -> Literal["route", "respond"]:
    """Use a conditional edge to choose the next node."""
    if state["category"] == "greeting":
        return "respond"
    return "route"


def route_node(state: AgentState) -> dict:
    """Select the handling route from the classification."""
    category = state["category"]

    routes = {
        "urgent": "priority_support",
        "technical": "technical_support",
        "general": "general_support",
        "greeting": "friendly_response",
    }

    return {"route": routes.get(category, "general_support")}


def respond_node(state: AgentState) -> dict:
    """Pause for human review, then produce the response."""
    feedback = interrupt(
        {
            "message": "Human review required before responding.",
            "user_input": state["user_input"],
            "category": state["category"],
            "route": state.get("route", "friendly_response"),
            "question": "Approve this route or provide feedback.",
        }
    )

    route = state.get("route", "friendly_response")
    category = state["category"]

    response = (
        f"Handled as '{category}' via '{route}'. "
        f"Human review: {feedback}"
    )

    return {
        "human_feedback": str(feedback),
        "response": response,
    }


def build_graph():
    """Build a three-node graph with conditional routing."""
    builder = StateGraph(AgentState)

    builder.add_node("classify", classify_node)
    builder.add_node("route", route_node)
    builder.add_node("respond", respond_node)

    builder.add_edge(START, "classify")

    builder.add_conditional_edges(
        "classify",
        choose_after_classify,
        {
            "route": "route",
            "respond": "respond",
        },
    )

    builder.add_edge("route", "respond")
    builder.add_edge("respond", END)

    # A checkpointer is required to pause and resume execution.
    return builder.compile(checkpointer=MemorySaver())


def run_tests(graph, log: list[str]) -> None:
    """Run five inputs and check classification and routing."""
    cases = [
        ("Hello there!", "greeting", "friendly_response"),
        ("I have a Python error", "technical", "technical_support"),
        ("This is an urgent emergency", "urgent", "priority_support"),
        ("Can you explain machine learning?", "general", "general_support"),
        ("There is a bug in my code", "technical", "technical_support"),
    ]

    log.append("=== FIVE ROUTING TESTS ===")
    passed = 0

    for index, (message, expected_category, expected_route) in enumerate(
        cases, start=1
    ):
        state = {"user_input": message}
        classified = classify_node(state)
        category = classified["category"]
        route = (
            "friendly_response"
            if category == "greeting"
            else route_node({**state, **classified})["route"]
        )

        ok = category == expected_category and route == expected_route
        passed += int(ok)

        line = (
            f"Test {index}: {message!r} | "
            f"category={category} | route={route} | "
            f"{'PASS' if ok else 'FAIL'}"
        )
        log.append(line)
        print(line)

    log.append(f"Routing tests: {passed}/5 passed")
    assert passed == 5, "One or more routing tests failed."


def run_human_review_demo(graph, log: list[str]) -> None:
    """Pause on interrupt, then resume with human feedback."""
    config = {"configurable": {"thread_id": "w10d2-human-review"}}
    initial_state = {"user_input": "Please help with a Python bug."}

    log.append("\n=== HUMAN-IN-THE-LOOP TEST ===")

    paused_result = graph.invoke(initial_state, config=config)
    interrupt_items = paused_result.get("__interrupt__", ())

    assert interrupt_items, "Expected graph to pause for human review."

    interrupt_value = interrupt_items[0].value
    log.append(f"PAUSED: {interrupt_value}")
    print("\nGraph paused. Human review is required.")
    print(f"Review details: {interrupt_value}")

    # Simulate a human providing approval/feedback.
    human_input = "Approved. Explain the likely cause and suggest a fix."
    log.append(f"Human feedback: {human_input}")

    resumed_result = graph.invoke(
        Command(resume=human_input),
        config=config,
    )

    response = resumed_result.get("response", "")
    assert response, "Graph did not produce a response after resuming."

    log.append(f"RESUMED RESPONSE: {response}")
    log.append("Human-in-the-loop test: PASS")
    print(f"Resumed response: {response}")
    print("Human-in-the-loop test: PASS")


def main():
    log = [
        "W10D2: LangGraph State Machines & Conditional Edges",
        "Three nodes: classify -> route -> respond",
    ]

    graph = build_graph()
    run_tests(graph, log)
    run_human_review_demo(graph, log)

    EVIDENCE_FILE.write_text("\n".join(log) + "\n", encoding="utf-8")

    print(f"\nEvidence saved to: {EVIDENCE_FILE}")
    print("W10D2 checks completed successfully.")


if __name__ == "__main__":
    main()

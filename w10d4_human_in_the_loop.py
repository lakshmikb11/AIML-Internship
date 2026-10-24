
"""
W10D4: Human-in-the-Loop with LangGraph

Three nodes:
    classify -> route -> respond

Features:
- Conditional routing
- Five test inputs
- Human-in-the-loop interrupt
- Approval and rejection testing
- Pause and resume using Command
- Saved output evidence
"""

from pathlib import Path
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class GraphState(TypedDict, total=False):
    user_input: str
    category: str
    requires_approval: bool
    human_decision: str
    response: str


def classify(state: GraphState) -> dict:
    """Classify requests that require human approval."""
    text = state["user_input"].lower()

    approval_keywords = (
        "send an email",
        "send this email",
        "delete my account",
        "delete the account",
        "make a purchase",
        "issue a refund",
        "publish this",
        "deploy this",
        "submit this",
    )

    needs_approval = any(
        keyword in text for keyword in approval_keywords
    )

    category = (
        "approval_required"
        if needs_approval
        else "general"
    )

    return {"category": category}


def choose_route(state: GraphState) -> str:
    """Select the next node using the classification."""
    if state.get("category") == "approval_required":
        return "route"

    return "respond"


def route(state: GraphState) -> dict:
    """Mark the request as requiring human approval."""
    return {"requires_approval": True}


def respond(state: GraphState) -> dict:
    """Respond directly or pause for human approval."""
    user_input = state["user_input"]

    if state.get("requires_approval", False):
        decision = interrupt(
            {
                "message": (
                    "Human approval required before proceeding."
                ),
                "request": user_input,
                "options": ["approved", "rejected"],
            }
        )

        if decision == "approved":
            return {
                "human_decision": "approved",
                "response": (
                    "Request approved by a human. "
                    "The simulated action is authorized."
                ),
            }

        if decision == "rejected":
            return {
                "human_decision": "rejected",
                "response": (
                    "Request rejected by a human. "
                    "The simulated action was not authorized."
                ),
            }

        return {
            "human_decision": "invalid",
            "response": (
                "Invalid decision. The simulated action "
                "was not authorized."
            ),
        }

    return {
        "response": (
            "General request processed successfully: "
            f"{user_input}"
        )
    }


def build_graph():
    """Build and compile the three-node LangGraph."""
    builder = StateGraph(GraphState)

    builder.add_node("classify", classify)
    builder.add_node("route", route)
    builder.add_node("respond", respond)

    builder.add_edge(START, "classify")

    builder.add_conditional_edges(
        "classify",
        choose_route,
        {
            "route": "route",
            "respond": "respond",
        },
    )

    builder.add_edge("route", "respond")
    builder.add_edge("respond", END)

    return builder.compile(checkpointer=MemorySaver())


def main():
    graph = build_graph()
    evidence = []

    def log(message=""):
        print(message)
        evidence.append(str(message))

    log("=" * 65)
    log("W10D4: HUMAN-IN-THE-LOOP WITH LANGGRAPH")
    log("=" * 65)
    log("Graph nodes: classify -> route -> respond")
    log("Conditional routing: approval_required / general")
    log()

    test_inputs = [
        "Explain what LangGraph is.",
        "How does conversation memory work?",
        "Please send this email to my manager.",
        "Delete my account.",
        "Explain conditional edges in LangGraph.",
    ]

    passed_tests = 0
    approval_count = 0
    rejection_count = 0

    log("PART 1: FIVE INPUT TESTS")
    log("-" * 65)

    for index, user_input in enumerate(test_inputs, start=1):
        config = {
            "configurable": {
                "thread_id": f"w10d4-test-{index}"
            }
        }

        result = graph.invoke(
            {"user_input": user_input},
            config,
        )

        expected_category = (
            "approval_required"
            if index in (3, 4)
            else "general"
        )

        assert result.get("category") == expected_category, (
            f"Test {index}: incorrect classification"
        )

        log(f"\nTest {index}: {user_input}")
        log(f"Classification: {result.get('category')}")

        if expected_category == "approval_required":
            log("Route: classify -> route -> respond")
        else:
            log("Route: classify -> respond")

        assert result.get("category") == expected_category

        if "__interrupt__" in result:
            log("Interrupt detected: PASS")

            # Test 4 is rejected; Test 3 is approved.
            decision = "rejected" if index == 4 else "approved"

            log(f"Simulated human decision: {decision}")

            result = graph.invoke(
                Command(resume=decision),
                config,
            )

            assert result.get("human_decision") == decision
            assert result.get("response")

            log(f"Human decision: {result['human_decision']}")
            log(f"Response: {result['response']}")
            log("Decision validation: PASS")

            if decision == "approved":
                approval_count += 1
            else:
                rejection_count += 1

        else:
            assert result.get("response")
            log(f"Response: {result.get('response')}")

        passed_tests += 1
        log(f"Test {index}: PASS")

    log("\nPART 2: EXPLICIT PAUSE AND RESUME")
    log("-" * 65)

    config = {
        "configurable": {
            "thread_id": "w10d4-manual-demo"
        }
    }

    request = {
        "user_input": (
            "Please send this email to my manager."
        )
    }

    paused = graph.invoke(request, config)

    assert "__interrupt__" in paused, (
        "Expected the graph to pause for human input."
    )

    log("Status: PAUSED — human input required")
    log(f"Interrupt details: {paused['__interrupt__']}")
    log("Submitting simulated human decision: approved")

    resumed = graph.invoke(
        Command(resume="approved"),
        config,
    )

    assert resumed.get("human_decision") == "approved"
    assert resumed.get("response")

    log(f"Human decision: {resumed['human_decision']}")
    log(f"Final response: {resumed['response']}")
    log("Pause and resume validation: PASS")

    log("\nPART 3: FINAL VALIDATION")
    log("-" * 65)

    assert passed_tests == 5
    assert approval_count == 1
    assert rejection_count == 1

    log(f"Five input tests: {passed_tests}/5 PASS")
    log("Conditional routing: PASS")
    log("Human interrupt: PASS")
    log(f"Human approvals: {approval_count}")
    log(f"Human rejections: {rejection_count}")
    log("Graph resumed after human input: PASS")
    log("All assertions passed.")
    log("\nW10D4 DEMONSTRATION COMPLETED SUCCESSFULLY.")

    evidence_dir = Path("output_evidence") / "w10d4"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    evidence_file = (
        evidence_dir / "human_in_the_loop_results.txt"
    )

    evidence_file.write_text(
        "\n".join(evidence) + "\n",
        encoding="utf-8",
    )

    print(f"\nEvidence saved to: {evidence_file}")


if __name__ == "__main__":
    main()

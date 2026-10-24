
from langgraph.types import Command

from w10d1_langgraph_stateful_agent import (
    build_graph,
    initial_state,
)


def test_five_inputs_are_classified_and_routed():
    graph = build_graph()

    cases = [
        ("What is LangGraph?", "question", False),
        ("Explain conditional routing", "question", False),
        ("Hello, agent!", "general", False),
        ("Delete the old project files", "action", True),
        ("Send an email to the team", "action", True),
    ]

    for index, (user_input, expected_category, needs_review) in enumerate(
        cases, start=1
    ):
        config = {
            "configurable": {"thread_id": f"pytest-w10d1-{index}"}
        }
        result = graph.invoke(initial_state(user_input), config)

        assert result["classification"] == expected_category
        assert result["needs_review"] is needs_review

        if needs_review:
            assert result.get("__interrupt__")
        else:
            assert result["response"]


def test_human_interrupt_and_resume():
    graph = build_graph()
    config = {"configurable": {"thread_id": "pytest-w10d1-approval"}}

    paused = graph.invoke(
        initial_state("Delete the old project files"),
        config,
    )

    assert paused.get("__interrupt__")

    resumed = graph.invoke(Command(resume="approve"), config)

    assert resumed["human_decision"] == "approve"
    assert "Human approval received" in resumed["response"]


def test_rejected_action_is_not_approved():
    graph = build_graph()
    config = {"configurable": {"thread_id": "pytest-w10d1-rejection"}}

    paused = graph.invoke(
        initial_state("Send an email to the team"),
        config,
    )

    assert paused.get("__interrupt__")

    result = graph.invoke(Command(resume="reject"), config)

    assert result["human_decision"] == "reject"
    assert "not approved" in result["response"].lower()

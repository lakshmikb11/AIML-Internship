
"""Offline tests for the W10D5 stateful customer-support agent."""

from langgraph.types import Command

from w10d5_stateful_customer_support_agent import (
    build_support_graph,
    classify_intent,
    route_request,
    evaluate_with_ragas,
)


def test_five_customer_intents():
    """Check classification and routing for five customer messages."""
    cases = [
        ("I have a billing question.", "billing", "draft_response"),
        ("My application has a Python error.", "technical", "draft_response"),
        ("Please delete my account.", "account_change", "human_review"),
        ("Please process a refund.", "billing", "human_review"),
        ("I need some general information.", "general", "draft_response"),
    ]

    for message, expected_intent, expected_route in cases:
        state = {"user_message": message}
        classified = {**state, **classify_intent(state)}

        assert classified["intent"] == expected_intent
        assert route_request(classified) == expected_route


def test_conversation_history_persists_between_turns():
    """Check that both turns are retained under the same thread ID."""
    graph = build_support_graph()
    config = {"configurable": {"thread_id": "test-conversation-001"}}

    first = graph.invoke(
        {
            "customer_id": "test-customer",
            "user_message": "I have a Python error.",
            "conversation_history": [],
        },
        config=config,
    )

    second = graph.invoke(
        {"user_message": "How can I troubleshoot it?"},
        config=config,
    )

    assert len(first["conversation_history"]) == 1
    assert len(second["conversation_history"]) == 2
    assert second["conversation_history"][0]["user"] == (
        "I have a Python error."
    )
    assert second["conversation_history"][1]["user"] == (
        "How can I troubleshoot it?"
    )
    assert second["response"]


def test_sensitive_request_pauses_and_resumes_after_approval():
    """Verify the human approval interrupt and resume flow."""
    graph = build_support_graph()
    config = {"configurable": {"thread_id": "test-approval-001"}}

    paused = graph.invoke(
        {
            "customer_id": "test-customer",
            "user_message": "Please delete my account.",
            "conversation_history": [],
        },
        config=config,
    )

    assert paused.get("__interrupt__")
    assert paused["intent"] == "account_change"

    resumed = graph.invoke(Command(resume="approved"), config=config)

    assert resumed["approval_decision"] == "approved"
    assert "NOT executed" in resumed["response"]
    assert len(resumed["conversation_history"]) == 1


def test_rejected_sensitive_request_does_not_execute_action():
    """Verify rejected requests are not reported as completed."""
    graph = build_support_graph()
    config = {"configurable": {"thread_id": "test-rejection-001"}}

    graph.invoke(
        {
            "customer_id": "test-customer",
            "user_message": "Please process a refund.",
            "conversation_history": [],
        },
        config=config,
    )

    rejected = graph.invoke(Command(resume="rejected"), config=config)

    assert rejected["approval_decision"] == "rejected"
    assert "No account or billing action was performed" in rejected["response"]


def test_ragas_gracefully_handles_missing_context():
    """Check evaluation skips empty evidence without requiring an LLM."""
    result = evaluate_with_ragas(
        question="How do I fix this?",
        answer="Please share the error message.",
        context="",
    )

    assert result["status"] == "skipped"
    assert "empty" in result["reason"].lower()

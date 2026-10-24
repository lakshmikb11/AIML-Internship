
"""W10D3: LangGraph + Memory — Persistent Conversations."""

from pathlib import Path
from tempfile import TemporaryDirectory
import re

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command, interrupt


EVIDENCE_DIR = Path("output_evidence/w10d3")
EVIDENCE_FILE = EVIDENCE_DIR / "run_results.txt"


class AgentState(TypedDict, total=False):
    user_input: str
    category: str
    route: str
    response: str
    human_feedback: str
    conversation_history: list[dict[str, str]]


def classify_node(state: AgentState) -> dict:
    """Classify a message using whole-word matching."""
    words = set(re.findall(r"\b\w+\b", state["user_input"].lower()))

    if words.intersection({"urgent", "emergency", "critical"}):
        category = "urgent"
    elif words.intersection({"error", "bug", "python", "code"}):
        category = "technical"
    elif words.intersection({"hello", "hi", "hey"}):
        category = "greeting"
    else:
        category = "general"

    return {"category": category}


def choose_after_classify(state: AgentState) -> str:
    """Select the next node through a conditional edge."""
    if state["category"] == "greeting":
        return "respond"
    return "route"


def route_node(state: AgentState) -> dict:
    """Map classifications to their handling routes."""
    routes = {
        "urgent": "priority_support",
        "technical": "technical_support",
        "general": "general_support",
        "greeting": "friendly_response",
    }
    return {"route": routes.get(state["category"], "general_support")}


def respond_node(state: AgentState) -> dict:
    """Pause for human review, then save the conversation turn."""
    feedback = interrupt(
        {
            "message": "Human review required.",
            "user_input": state["user_input"],
            "category": state["category"],
            "route": state.get("route", "friendly_response"),
            "question": "Approve this response or provide feedback.",
        }
    )

    route = state.get("route", "friendly_response")
    category = state["category"]
    response = (
        f"Handled as '{category}' via '{route}'. "
        f"Human review: {feedback}"
    )

    history = list(state.get("conversation_history", []))
    history.append(
        {
            "user": state["user_input"],
            "assistant": response,
        }
    )

    return {
        "human_feedback": str(feedback),
        "response": response,
        "conversation_history": history,
    }


def build_graph(checkpointer):
    """Build the three-node persistent LangGraph."""
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

    return builder.compile(checkpointer=checkpointer)


def run_routing_tests(log: list[str]) -> None:
    """Verify five classifications and their expected routes."""
    cases = [
        ("Hello there!", "greeting", "friendly_response"),
        ("I have a Python error", "technical", "technical_support"),
        ("This is an urgent emergency", "urgent", "priority_support"),
        ("Explain machine learning", "general", "general_support"),
        ("There is a bug in my code", "technical", "technical_support"),
    ]

    log.append("=== FIVE ROUTING TESTS ===")
    passed = 0

    for number, (message, expected_category, expected_route) in enumerate(
        cases, start=1
    ):
        state = {"user_input": message}
        classified = classify_node(state)
        category = classified["category"]

        if choose_after_classify(classified) == "respond":
            route = "friendly_response"
        else:
            route = route_node({**state, **classified})["route"]

        success = (
            category == expected_category
            and route == expected_route
        )
        passed += int(success)

        line = (
            f"Test {number}: {message!r} | category={category} | "
            f"route={route} | {'PASS' if success else 'FAIL'}"
        )
        print(line)
        log.append(line)

    log.append(f"Routing tests: {passed}/5 passed")
    assert passed == 5, "Routing tests failed."


def start_and_resume(graph, thread_id: str, message: str, feedback: str):
    """Run a conversation turn through pause and human resume."""
    config = {"configurable": {"thread_id": thread_id}}

    paused = graph.invoke({"user_input": message}, config=config)
    interruptions = paused.get("__interrupt__", ())

    assert interruptions, "Graph did not pause for human review."
    print(f"Paused for review: {interruptions[0].value}")

    resumed = graph.invoke(Command(resume=feedback), config=config)
    assert resumed.get("response"), "Graph did not resume successfully."

    print(f"Resumed response: {resumed['response']}")
    return resumed, config


def main():
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    log = [
        "W10D3: LangGraph + Memory — Persistent Conversations",
        "Graph: classify -> route -> respond",
        "Persistence: SQLite checkpoint store",
    ]

    run_routing_tests(log)

    # Use a temporary SQLite database so no generated database
    # needs to be committed to the repository.
    with TemporaryDirectory(prefix="w10d3_") as temp_dir:
        database = str(Path(temp_dir) / "conversations.sqlite")
        thread_id = "w10d3-persistent-demo"

        log.append("\n=== FIRST CONVERSATION TURN ===")

        with SqliteSaver.from_conn_string(database) as saver:
            graph = build_graph(saver)

            first, _ = start_and_resume(
                graph,
                thread_id,
                "I have a Python bug.",
                "Approved. Suggest a simple debugging approach.",
            )

            assert len(first["conversation_history"]) == 1
            log.append(f"First turn response: {first['response']}")
            log.append("First turn pause/resume: PASS")

        # The checkpoint store is closed here. Reopen the same
        # SQLite file to verify saved conversation state is recoverable.
        log.append("\n=== REOPEN CHECKPOINT STORE ===")

        with SqliteSaver.from_conn_string(database) as saver:
            graph = build_graph(saver)
            config = {"configurable": {"thread_id": thread_id}}

            saved_state = graph.get_state(config)
            previous_history = saved_state.values.get(
                "conversation_history", []
            )

            assert len(previous_history) == 1
            log.append(
                f"Recovered conversation turns: {len(previous_history)}"
            )
            print("Recovered previous conversation after reopening SQLite.")

            second, _ = start_and_resume(
                graph,
                thread_id,
                "Thanks, explain it simply.",
                "Approved. Keep the explanation beginner-friendly.",
            )

            history = second["conversation_history"]
            assert len(history) == 2
            assert history[0]["user"] == "I have a Python bug."
            assert history[1]["user"] == "Thanks, explain it simply."

            log.append(f"Second turn response: {second['response']}")
            log.append(f"Conversation turns after resume: {len(history)}")
            log.append("Persistent conversation recovery: PASS")
            log.append("Human-in-the-loop pause/resume: PASS")

    log.append("\nOverall result: PASS")
    EVIDENCE_FILE.write_text("\n".join(log) + "\n", encoding="utf-8")

    print("\nPersistent conversation test: PASS")
    print(f"Evidence saved to: {EVIDENCE_FILE}")
    print("W10D3 completed successfully.")


if __name__ == "__main__":
    main()

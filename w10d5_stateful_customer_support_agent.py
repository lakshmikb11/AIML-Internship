
"""W10D5: Stateful Customer Support Agent using LangGraph.

Includes intent classification, persistent conversation history,
human approval for sensitive actions, optional CrewAI/Ollama,
optional Ragas evaluation, MLflow tracking, and output evidence.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


BASE_DIR = Path(__file__).resolve().parent
EVIDENCE_FILE = (
    BASE_DIR / "output_evidence" / "w10d5"
    / "support_agent_results.txt"
)
EXPERIMENT_NAME = "W10D5_Stateful_Customer_Support"


class SupportState(TypedDict, total=False):
    customer_id: str
    user_message: str
    conversation_history: list[dict[str, Any]]
    intent: str
    requires_human_approval: bool
    approval_status: str
    approval_decision: str
    draft_response: str
    response: str



def normalize_response(response: Any) -> str:
    """Normalize whitespace and repair merged words in responses."""
    text = str(response or "")

    # Repair the merged phrase and preserve a space after "describe".
    text = re.sub(
        r"(?i)please\s*describe\s*",
        "Please describe ",
        text,
    )

    return re.sub(r"\s+", " ", text).strip()



def classify_intent(message: Any) -> Any:
    """Classify customer intent and approval requirements.

    Dictionary input returns classification fields in a dictionary.
    String input returns only the intent label for compatibility.
    """
    input_is_dict = isinstance(message, dict)

    if input_is_dict:
        text = str(
            message.get("user_message", message.get("message", ""))
        ).lower()
    else:
        text = str(message or "").lower()

    technical_terms = (
        "error",
        "bug",
        "crash",
        "not working",
        "doesn't work",
        "does not work",
        "failed",
        "failure",
        "exception",
        "install",
        "installation",
        "loading",
    )

    account_action_terms = (
        "delete my account",
        "delete account",
        "close my account",
        "close account",
        "change my account",
        "change account",
        "update my account",
        "update account",
        "change my email",
        "change email",
        "reset my password",
        "reset password",
        "cancel my subscription",
        "cancel subscription",
    )

    billing_action_terms = (
        "process a refund",
        "process refund",
        "issue a refund",
        "issue refund",
        "refund me",
        "refund my",
        "make a payment",
        "process payment",
        "charge my",
        "charge the",
    )

    billing_question_terms = (
        "billing",
        "bill",
        "refund",
        "payment",
        "charge",
        "invoice",
    )

    account_question_terms = (
        "account",
        "password",
        "username",
        "profile",
        "email address",
        "sign in",
        "log in",
        "login",
    )

    order_terms = (
        "order status",
        "track my order",
        "delivery",
    )

    requires_approval = any(
        term in text for term in account_action_terms + billing_action_terms
    )

    if any(term in text for term in account_action_terms):
        intent = "account_change"
    elif any(term in text for term in technical_terms):
        intent = "technical"
    elif any(term in text for term in billing_question_terms):
        intent = "billing"
    elif any(term in text for term in account_question_terms):
        intent = "account"
    elif any(term in text for term in order_terms):
        intent = "order"
    else:
        intent = "general"

    if input_is_dict:
        return {
            **message,
            "intent": intent,
            "requires_human_approval": requires_approval,
        }

    return intent


def classify_node(state: SupportState) -> dict[str, Any]:
    """Classify the latest message and set its approval requirement."""
    result = classify_intent(
        {"user_message": state.get("user_message", "")}
    )
    return {
        "intent": result["intent"],
        "requires_human_approval": result["requires_human_approval"],
    }


def route_request(state: SupportState) -> str:
    """Route only sensitive actions to human review."""
    if state.get("requires_human_approval", False):
        return "human_review"
    return "draft_response"


def human_review(state: SupportState) -> dict[str, str]:
    """Pause execution until a human approves or rejects the request."""
    decision = interrupt(
        {
            "type": "human_approval_required",
            "customer_id": state.get("customer_id", "unknown"),
            "user_message": state.get("user_message", ""),
            "intent": state.get("intent", "general"),
            "message": (
                "Approve or reject this sensitive request. "
                "This demo will not execute account or billing actions."
            ),
        }
    )

    if isinstance(decision, dict):
        raw_decision = decision.get(
            "approval_decision",
            decision.get("decision", decision.get("approved", "rejected")),
        )
    else:
        raw_decision = decision

    if isinstance(raw_decision, bool):
        approved = raw_decision
    else:
        approved = str(raw_decision).strip().lower() in {
            "approved", "approve", "yes", "true", "1"
        }

    status = "approved" if approved else "rejected"
    return {
        "approval_status": status,
        "approval_decision": status,
    }


def generate_with_crewai(state: SupportState) -> str:
    """Optionally generate a response using CrewAI and local Ollama."""
    message = state.get("user_message", "")
    intent = state.get("intent", "general")

    if os.getenv("W10D5_USE_CREWAI", "0") != "1":
        if intent == "technical":
            return (
                "I can help troubleshoot this issue. Please share the "
                "error message, what you were doing, and the steps to "
                "reproduce it."
            )
        if intent in {"account", "account_change"}:
            return (
                "I can help with your account question. Please describe "
                "the issue without sharing your password or other "
                "sensitive credentials."
            )
        if intent == "billing":
            return (
                "I can help with your billing question. Please describe "
                "the issue without sharing full payment details."
            )
        if intent == "order":
            return (
                "I can help with your order question. Please provide "
                "your order reference without sharing unnecessary "
                "personal information."
            )
        return (
            "Thanks for contacting support. Please describe the issue "
            "and I will help you find the next step."
        )

    try:
        from crewai import Agent, Crew, LLM, Process, Task

        model_name = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        base_url = os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )

        llm = LLM(
            model=f"ollama/{model_name}",
            base_url=base_url,
        )

        agent = Agent(
            role="Customer Support Specialist",
            goal=(
                "Provide clear, safe support guidance. Never claim a "
                "sensitive action was completed when it was not."
            ),
            backstory=(
                "You are a careful support specialist who protects "
                "customer privacy and never requests passwords or "
                "full payment credentials."
            ),
            llm=llm,
            verbose=False,
        )

        task = Task(
            description=(
                f"Customer intent: {intent}\n"
                f"Customer message: {message}\n\n"
                "Write a concise, helpful response. Do not claim to "
                "perform refunds, account changes, or payments. Ask "
                "only for information necessary to help."
            ),
            expected_output="A safe, helpful customer support response.",
            agent=agent,
        )

        result = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=False,
        ).kickoff()

        return normalize_response(result)

    except Exception as exc:
        print(
            "CrewAI/Ollama unavailable; using fallback response: "
            f"{exc}"
        )

        if intent == "technical":
            return (
                "I can help troubleshoot this issue. Please share the "
                "error message, what you were doing, and the steps to "
                "reproduce it."
            )

        return (
            "Thanks for contacting support. Please describe the issue "
            "and I will help you find the next step."
        )


def draft_response(state: SupportState) -> dict[str, str]:
    """Draft a safe response without executing sensitive actions."""
    if state.get("requires_human_approval", False):
        approval = str(
            state.get("approval_status", "rejected")
        ).strip().lower()

        if approval == "approved":
            response = (
                "Your request has been approved for the next step. "
                "No account or billing action was performed. "
                "The action was NOT executed by this demo."
            )
        else:
            response = (
                "Your request was rejected. "
                "No account or billing action was performed. "
                "The action was NOT executed. "
                "Please contact the appropriate support team "
                "if you need further help."
            )

        return {"draft_response": response}

    return {"draft_response": generate_with_crewai(state)}


def finalize_response(state: SupportState) -> dict[str, Any]:
    """Normalize the final response and append the conversation turn."""
    user_message = normalize_response(state.get("user_message", ""))
    response = normalize_response(state.get("draft_response", ""))

    history = list(state.get("conversation_history", []))
    history.append(
        {
            "user": user_message,
            "assistant": response,
            "user_message": user_message,
            "intent": state.get("intent", "general"),
            "response": response,
        }
    )

    return {
        "response": response,
        "conversation_history": history,
    }


def build_support_graph():
    """Create the stateful LangGraph workflow."""
    builder = StateGraph(SupportState)

    builder.add_node("classify", classify_node)
    builder.add_node("human_review", human_review)
    builder.add_node("draft_response", draft_response)
    builder.add_node("finalize_response", finalize_response)

    builder.add_edge(START, "classify")
    builder.add_conditional_edges(
        "classify",
        route_request,
        {
            "human_review": "human_review",
            "draft_response": "draft_response",
        },
    )
    builder.add_edge("human_review", "draft_response")
    builder.add_edge("draft_response", "finalize_response")
    builder.add_edge("finalize_response", END)

    return builder.compile(checkpointer=InMemorySaver())


def evaluate_with_ragas(
    question: str,
    answer: str,
    context: Any,
) -> dict[str, Any]:
    """Evaluate faithfulness if question, answer, and context are present."""
    if isinstance(context, (list, tuple)):
        contexts = [str(item).strip() for item in context if str(item).strip()]
    else:
        contexts = [str(context).strip()] if str(context or "").strip() else []

    if not question or not answer or not contexts:
        return {
            "status": "skipped",
            "reason": "Question, answer, or context is empty.",
        }

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import Faithfulness

        dataset = Dataset.from_dict(
            {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
            }
        )
        result = evaluate(dataset, metrics=[Faithfulness()])

        return {
            "status": "completed",
            "faithfulness": str(result),
        }
    except Exception as exc:
        return {"status": "error", "reason": str(exc)}


def log_mlflow_run(
    conversation_turns: int,
    first_intent: str,
    second_intent: str,
    ragas_status: str,
) -> str:
    """Log demo parameters and metrics to local MLflow tracking."""
    try:
        import mlflow

        tracking_path = BASE_DIR / "mlflow.db"
        mlflow.set_tracking_uri(f"sqlite:///{tracking_path.as_posix()}")
        mlflow.set_experiment(EXPERIMENT_NAME)

        with mlflow.start_run(run_name="support_agent_demo"):
            mlflow.log_param("first_intent", first_intent)
            mlflow.log_param("second_intent", second_intent)
            mlflow.log_param("ragas_status", ragas_status)
            mlflow.log_metric("conversation_turns", conversation_turns)

    except Exception as exc:
        print(f"MLflow logging unavailable: {exc}")

    return EXPERIMENT_NAME


def run_demo() -> dict[str, Any]:
    """Run two support turns and save JSON evidence."""
    graph = build_support_graph()
    config = {
        "configurable": {"thread_id": "w10d5-demo-customer-001"}
    }

    first_result = graph.invoke(
        {
            "customer_id": "demo-customer-001",
            "user_message": (
                "My application has an error and is not working."
            ),
            "conversation_history": [],
        },
        config=config,
    )

    second_result = graph.invoke(
        {
            "customer_id": "demo-customer-001",
            "user_message": "Can you explain what support can do?",
            "conversation_history": first_result.get(
                "conversation_history", []
            ),
        },
        config=config,
    )

    history = second_result.get("conversation_history", [])
    latest_response = (
        normalize_response(history[-1]["assistant"])
        if history
        else normalize_response(second_result.get("response", ""))
    )

    ragas_result = evaluate_with_ragas(
        question="Can you explain what support can do?",
        answer=latest_response,
        context=[],
    )

    experiment_name = log_mlflow_run(
        conversation_turns=len(history),
        first_intent=str(first_result.get("intent", "unknown")),
        second_intent=str(second_result.get("intent", "unknown")),
        ragas_status=str(ragas_result.get("status", "unknown")),
    )

    evidence = {
        "task": "W10D5 Stateful Customer Support Agent",
        "thread_id": config["configurable"]["thread_id"],
        "first_intent": first_result.get("intent", "unknown"),
        "second_intent": second_result.get("intent", "unknown"),
        "conversation_turns": len(history),
        "latest_response": latest_response,
        "conversation_history": history,
        "ragas_evaluation": ragas_result,
        "mlflow_experiment": experiment_name,
    }

    EVIDENCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_FILE.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("W10D5 support-agent demo completed.")
    print(f"Conversation turns: {len(history)}")
    print(f"Ragas evaluation status: {ragas_result['status']}")
    print(f"Evidence saved to: {EVIDENCE_FILE}")

    return evidence


if __name__ == "__main__":
    run_demo()


"""
W9D5: Automated Research Report Agent

Technologies:
- CrewAI: Researcher, Writer, and Reviewer agents
- LangGraph: workflow orchestration
- MLflow: experiment and run tracking using SQLite
- Ragas: optional faithfulness evaluation
- DDGS: live web-source retrieval
"""

from __future__ import annotations

import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

import mlflow
from crewai import Agent, Crew, LLM, Process, Task
from langgraph.graph import END, START, StateGraph

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
EVIDENCE_DIR = BASE_DIR / "output_evidence" / "w9d5"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

# Use SQLite instead of MLflow's legacy filesystem tracking backend.
MLFLOW_DB = BASE_DIR / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")
mlflow.set_experiment("W9D5_Automated_Research_Report")

TOPIC = os.getenv(
    "RESEARCH_TOPIC",
    "Recent practical applications of generative AI in software development",
)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)

llm = LLM(
    model=f"ollama/{OLLAMA_MODEL}",
    base_url=OLLAMA_BASE_URL,
    temperature=0.2,
)


# ---------------------------------------------------------------------------
# Evidence helpers
# ---------------------------------------------------------------------------

def save_evidence(filename: str, content: str) -> Path:
    """Save a UTF-8 evidence file in the W9D5 directory."""
    path = EVIDENCE_DIR / filename
    path.write_text(content.rstrip() + "\n", encoding="utf-8")
    print(f"Evidence saved: {path.relative_to(BASE_DIR)}")
    return path


def format_sources(sources: list[dict]) -> str:
    """Format retrieved sources as readable text."""
    if not sources:
        return "No web sources were retrieved."

    sections = []

    for index, source in enumerate(sources, start=1):
        title = source.get("title") or "Untitled"
        url = source.get("url") or ""
        snippet = source.get("snippet") or ""

        sections.append(
            f"{index}. {title}\n"
            f"URL: {url}\n"
            f"Summary: {snippet}"
        )

    return "\n\n".join(sections)


def validate_report(report: str) -> bool:
    """Check that the report contains a reasonable amount of text."""
    if not isinstance(report, str):
        return False

    cleaned = report.strip()
    return len(cleaned) >= 30


# ---------------------------------------------------------------------------
# Web research
# ---------------------------------------------------------------------------

def search_web(topic: str, max_results: int = 5) -> list[dict]:
    """Retrieve source titles, URLs, and snippets using DDGS."""
    try:
        from ddgs import DDGS

        with DDGS() as search:
            results = list(search.text(topic, max_results=max_results))

        sources = []

        for result in results:
            url = result.get("href", "")
            if not url:
                continue

            sources.append(
                {
                    "title": result.get("title", "Untitled"),
                    "url": url,
                    "snippet": result.get("body", ""),
                }
            )

        return sources

    except Exception as exc:
        print(f"Web search failed: {type(exc).__name__}: {exc}")
        return []


# ---------------------------------------------------------------------------
# CrewAI agents and tasks
# ---------------------------------------------------------------------------

def create_agents() -> tuple[Agent, Agent, Agent]:
    """Create the Researcher, Writer, and Reviewer agents."""

    researcher = Agent(
        role="Research Analyst",
        goal=(
            "Analyze the supplied research sources, identify useful findings, "
            "and distinguish sourced information from unsupported claims."
        ),
        backstory=(
            "You are a careful technology researcher. You do not invent "
            "sources, URLs, publication dates, or evidence."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    writer = Agent(
        role="Technical Report Writer",
        goal=(
            "Create a clear, well-organized technical report grounded in "
            "the research findings and supplied sources."
        ),
        backstory=(
            "You write technical reports for software developers. You balance "
            "practical applications, benefits, risks, and limitations."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    reviewer = Agent(
        role="Research Quality Reviewer",
        goal=(
            "Review the report for clarity, completeness, source attribution, "
            "unsupported claims, and practical usefulness."
        ),
        backstory=(
            "You are a critical reviewer. You identify evidence gaps and "
            "recommend concrete improvements without claiming independent "
            "verification that was not performed."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    return researcher, writer, reviewer


def run_research_crew(source_text: str) -> tuple[str, str]:
    """Run three sequential CrewAI tasks and return report and review."""

    researcher, writer, reviewer = create_agents()

    research_task = Task(
        description=(
            f"Research topic: {TOPIC}\n\n"
            "Analyze the supplied source material below. Identify key "
            "applications, benefits, limitations, implementation risks, "
            "and useful findings. Do not invent source details.\n\n"
            f"SOURCE MATERIAL:\n{source_text}"
        ),
        expected_output=(
            "Structured research notes with key findings, limitations, "
            "and source URLs where available."
        ),
        agent=researcher,
    )

    writing_task = Task(
        description=(
            f"Write a technical report about: {TOPIC}\n"
            "Use the research notes from the preceding task. Include a title, "
            "overview, practical applications, benefits, limitations, "
            "implementation considerations, conclusion, and source URLs. "
            "If no sources were retrieved, disclose this limitation. "
            "Do not invent citations."
        ),
        expected_output=(
            "A structured Markdown report with clear headings and honest "
            "source attribution."
        ),
        agent=writer,
        context=[research_task],
    )

    review_task = Task(
        description=(
            "Review the report produced by the writer. Assess clarity, "
            "completeness, evidence quality, unsupported claims, limitations, "
            "and practical relevance. Provide specific improvements. "
            "Do not claim that URLs were independently verified. "
            "Return the review findings and a final revised report."
        ),
        expected_output=(
            "A quality review with strengths, weaknesses, evidence gaps, "
            "recommendations, and a final revised report."
        ),
        agent=reviewer,
        context=[research_task, writing_task],
    )

    crew = Crew(
        agents=[researcher, writer, reviewer],
        tasks=[research_task, writing_task, review_task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()

    # Preserve each task's output separately when CrewAI exposes it.
    tasks_output = getattr(result, "tasks_output", None) or []

    report = ""
    review = str(result)

    if len(tasks_output) >= 2:
        report = str(tasks_output[1].raw)
    elif len(tasks_output) == 1:
        report = str(tasks_output[0].raw)

    if tasks_output:
        review = str(tasks_output[-1].raw)

    # Fall back to the final crew output if task details are unavailable.
    if not report.strip():
        report = str(result)

    return report, review


# ---------------------------------------------------------------------------
# Ragas evaluation
# ---------------------------------------------------------------------------

def evaluate_report(report: str, sources: list[dict]) -> dict:
    """
    Attempt a Ragas faithfulness evaluation.

    The evaluation is marked unavailable if the installed Ragas API,
    local model, or evaluation dependencies cannot complete the operation.
    """
    if not sources:
        return {
            "status": "skipped",
            "reason": "No retrieved sources were available for evaluation.",
        }

    if not validate_report(report):
        return {
            "status": "skipped",
            "reason": "The report was empty or too short to evaluate.",
        }

    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import Faithfulness

        from langchain_ollama import ChatOllama

        judge = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0,
        )
        evaluator_llm = LangchainLLMWrapper(judge)

        source_context = "\n\n".join(
            item.get("snippet", "")
            for item in sources
            if item.get("snippet")
        )

        if not source_context.strip():
            return {
                "status": "skipped",
                "reason": "Retrieved sources contained no usable snippets.",
            }

        dataset = EvaluationDataset.from_list(
            [
                {
                    "user_input": TOPIC,
                    "response": report,
                    "retrieved_contexts": [source_context],
                }
            ]
        )

        result = evaluate(
            dataset=dataset,
            metrics=[Faithfulness(llm=evaluator_llm)],
        )

        scores = result.to_pandas().to_dict(orient="records")

        return {
            "status": "completed",
            "scores": scores,
        }

    except Exception as exc:
        return {
            "status": "unavailable",
            "reason": f"{type(exc).__name__}: {exc}",
        }


# ---------------------------------------------------------------------------
# LangGraph workflow
# ---------------------------------------------------------------------------

class ResearchState(TypedDict, total=False):
    topic: str
    sources: list[dict]
    report: str
    review: str
    evaluation: dict
    errors: list[str]
    status: str


def collect_sources(state: ResearchState) -> dict:
    """LangGraph node: retrieve and record research sources."""
    topic = state.get("topic", TOPIC)
    errors = list(state.get("errors", []))

    sources = search_web(topic)

    if sources:
        search_status = f"Retrieved {len(sources)} web sources."
    else:
        search_status = (
            "No web sources were retrieved. The report must disclose "
            "this evidence limitation."
        )

    save_evidence(
        "source_search_log.txt",
        (
            f"# W9D5 Source Search Log\n\n"
            f"- Topic: {topic}\n"
            f"- Time (UTC): {datetime.now(timezone.utc).isoformat()}\n"
            f"- Status: {search_status}\n\n"
            f"{format_sources(sources)}"
        ),
    )

    return {
        "sources": sources,
        "errors": errors,
    }


def generate_report(state: ResearchState) -> dict:
    """LangGraph node: run the CrewAI research and writing agents."""
    sources = state.get("sources", [])
    source_text = format_sources(sources)

    try:
        report, review = run_research_crew(source_text)

        save_evidence("research_report.md", report)
        save_evidence("quality_review.txt", review)

        return {
            "report": report,
            "review": review,
        }

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        errors = list(state.get("errors", []))
        errors.append(f"CrewAI execution failed: {error}")

        save_evidence(
            "pipeline_error.txt",
            traceback.format_exc(),
        )

        return {
            "report": "",
            "review": f"CrewAI execution failed: {error}",
            "errors": errors,
        }


def review_and_evaluate(state: ResearchState) -> dict:
    """LangGraph node: validate the report and attempt Ragas evaluation."""
    report = state.get("report", "")
    sources = state.get("sources", [])
    errors = list(state.get("errors", []))

    if not validate_report(report):
        errors.append("Report validation failed: report is empty or too short.")
        evaluation = {
            "status": "skipped",
            "reason": "No valid report was available.",
        }
        status = "failed"
    else:
        evaluation = evaluate_report(report, sources)
        status = "completed"

    save_evidence(
        "ragas_evaluation.json",
        json.dumps(evaluation, indent=2, ensure_ascii=False),
    )

    return {
        "evaluation": evaluation,
        "errors": errors,
        "status": status,
    }


def build_workflow():
    """Build and compile the LangGraph workflow."""
    graph = StateGraph(ResearchState)

    graph.add_node("collect_sources", collect_sources)
    graph.add_node("generate_report", generate_report)
    graph.add_node("review_and_evaluate", review_and_evaluate)

    graph.add_edge(START, "collect_sources")
    graph.add_edge("collect_sources", "generate_report")
    graph.add_edge("generate_report", "review_and_evaluate")
    graph.add_edge("review_and_evaluate", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# MLflow tracking and execution
# ---------------------------------------------------------------------------

def main() -> int:
    """Run the automated research report pipeline and log its results."""
    started_at = datetime.now(timezone.utc)

    print("=" * 72)
    print("W9D5: Automated Research Report Agent")
    print(f"Topic: {TOPIC}")
    print(f"Model: {OLLAMA_MODEL}")
    print("=" * 72)

    final_state: ResearchState = {
        "topic": TOPIC,
        "sources": [],
        "report": "",
        "review": "",
        "evaluation": {},
        "errors": [],
        "status": "running",
    }

    try:
        with mlflow.start_run(run_name="automated_research_report"):
            mlflow.log_param("topic", TOPIC)
            mlflow.log_param("model", OLLAMA_MODEL)
            mlflow.log_param("framework", "CrewAI + LangGraph")
            mlflow.log_param("evaluation_framework", "Ragas")
            mlflow.log_param("source_search", "DDGS")

            workflow = build_workflow()
            final_state = workflow.invoke(final_state)

            sources = final_state.get("sources", [])
            report = final_state.get("report", "")
            review = final_state.get("review", "")
            evaluation = final_state.get("evaluation", {})
            errors = final_state.get("errors", [])

            mlflow.log_metric("source_count", len(sources))
            mlflow.log_metric(
                "report_valid",
                1 if validate_report(report) else 0,
            )

            mlflow.set_tag(
                "ragas_status",
                str(evaluation.get("status", "unknown")),
            )
            mlflow.set_tag(
                "pipeline_status",
                str(final_state.get("status", "unknown")),
            )

            # Upload evidence files to the MLflow run.
            for filename in (
                "source_search_log.txt",
                "research_report.md",
                "quality_review.txt",
                "ragas_evaluation.json",
            ):
                evidence_path = EVIDENCE_DIR / filename
                if evidence_path.exists():
                    mlflow.log_artifact(
                        str(evidence_path),
                        artifact_path="w9d5_evidence",
                    )

            run = mlflow.active_run()
            run_id = run.info.run_id if run else "unknown"

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        errors = list(final_state.get("errors", []))
        errors.append(error)
        final_state["errors"] = errors
        final_state["status"] = "failed"

        save_evidence("pipeline_error.txt", traceback.format_exc())
        run_id = "not available"

    finished_at = datetime.now(timezone.utc)

    metadata = {
        "task": "W9D5 Automated Research Report Agent",
        "topic": TOPIC,
        "model": OLLAMA_MODEL,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "duration_seconds": round(
            (finished_at - started_at).total_seconds(),
            2,
        ),
        "source_count": len(final_state.get("sources", [])),
        "report_valid": validate_report(final_state.get("report", "")),
        "ragas_evaluation": final_state.get("evaluation", {}),
        "pipeline_status": final_state.get("status", "unknown"),
        "errors": final_state.get("errors", []),
        "mlflow_tracking_uri": mlflow.get_tracking_uri(),
        "mlflow_run_id": run_id,
    }

    save_evidence(
        "run_metadata.json",
        json.dumps(metadata, indent=2, ensure_ascii=False),
    )

    print("\nPipeline summary")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))

    return 0 if metadata["report_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

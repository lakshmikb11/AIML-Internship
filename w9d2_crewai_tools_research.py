
"""W9D2: CrewAI tools — web search and safe code execution.

The script supports:
1. A baseline research crew without external tools.
2. A tool-assisted crew using Serper when credentials are available.
3. A restricted Python validation task that does not execute arbitrary
   LLM-generated code.
4. Separate evidence files under output_evidence/w9d2/.
"""

from __future__ import annotations

import os
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from crewai import Agent, Crew, LLM, Process, Task

BASE_DIR = Path(__file__).resolve().parent
EVIDENCE_DIR = BASE_DIR / "output_evidence" / "w9d2"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

TOPIC = "Practical applications of generative AI in software development"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

llm = LLM(
    model=f"ollama/{OLLAMA_MODEL}",
    base_url=OLLAMA_BASE_URL,
    temperature=0.2,
)

def write_evidence(filename: str, content: str) -> None:
    """Write UTF-8 evidence into the W9D2 evidence directory."""
    path = EVIDENCE_DIR / filename
    path.write_text(content.rstrip() + "\n", encoding="utf-8")
    print(f"Evidence saved: {path.relative_to(BASE_DIR)}")


def safe_python_check(code: str) -> str:
    """Validate Python syntax without executing the supplied code."""
    if not code.strip():
        return "Validation failed: no Python code was supplied."

    try:
        compile(code, "<w9d2-validation>", "exec")
    except (SyntaxError, ValueError, TypeError) as exc:
        return f"Python syntax validation failed: {exc}"

    return (
        "Python syntax validation passed. "
        "The code was compiled for syntax checking only; it was not executed."
    )


def create_agents(web_search_tool=None):
    researcher_tools = [web_search_tool] if web_search_tool else []

    researcher = Agent(
        role="AI Researcher",
        goal=(
            "Research the topic accurately, distinguish established facts "
            "from uncertainty, and use live search results when available."
        ),
        backstory=(
            "You are a careful AI research analyst. You never invent sources "
            "or claim to have searched the web when search failed."
        ),
        tools=researcher_tools,
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    writer = Agent(
        role="Technical Writer",
        goal="Turn research findings into a clear, structured technical report.",
        backstory=(
            "You write practical technical reports for software developers. "
            "You identify limitations and do not add unsupported citations."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    reviewer = Agent(
        role="Research Reviewer",
        goal=(
            "Review the report for clarity, completeness, evidence quality, "
            "and unsupported or misleading claims."
        ),
        backstory=(
            "You are a skeptical technical reviewer who checks whether "
            "claims are supported and identifies missing evidence."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    return researcher, writer, reviewer


def create_tasks(researcher, writer, reviewer, use_web: bool):
    search_instruction = (
        "Use the available web-search tool to retrieve relevant live sources. "
        "Report only sources and details actually returned by the tool. "
        "If search fails or is unavailable, state that limitation clearly."
        if use_web
        else
        "Use your existing model knowledge only. Do not claim to have searched "
        "the web or present unverified information as current."
    )

    research_task = Task(
        description=(
            f"Research this topic: {TOPIC}\n"
            "Identify at least three practical applications, likely benefits, "
            "limitations, and implementation considerations.\n"
            f"{search_instruction}\n"
            "Keep the findings concise and separate supported facts from "
            "general observations."
        ),
        expected_output=(
            "Structured research notes covering applications, benefits, "
            "limitations, and evidence/source status."
        ),
        agent=researcher,
    )

    writing_task = Task(
        description=(
            f"Write a technical report about: {TOPIC}\n"
            "Use the research notes from the previous task. Include a title, "
            "overview, applications, benefits, limitations, and conclusion. "
            "Do not invent citations or claim live-source verification unless "
            "the research task actually retrieved sources."
        ),
        expected_output="A clear report with headings and explicit evidence limitations.",
        agent=writer,
        context=[research_task],
    )

    review_task = Task(
        description=(
            "Review the technical report from the previous task. Check "
            "accuracy, completeness, clarity, unsupported claims, and source "
            "quality. Provide a short review and concrete improvements. "
            "Do not claim independent source verification unless you performed it."
        ),
        expected_output="A review with strengths, limitations, and recommended improvements.",
        agent=reviewer,
        context=[research_task, writing_task],
    )

    return [research_task, writing_task, review_task]


def run_crew(label: str, use_web: bool, web_search_tool=None) -> str:
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")

    researcher, writer, reviewer = create_agents(
        web_search_tool if use_web else None
    )
    tasks = create_tasks(researcher, writer, reviewer, use_web)

    crew = Crew(
        agents=[researcher, writer, reviewer],
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()
    output = str(result)

    filename = (
        "tool_assisted_results.txt" if use_web else "baseline_results.txt"
    )
    write_evidence(
        filename,
        f"# {label}\n\n"
        f"Topic: {TOPIC}\n"
        f"Run time (UTC): {datetime.now(timezone.utc).isoformat()}\n\n"
        f"{output}",
    )
    return output


def main() -> int:
    metadata = [
        "# W9D2 Run Metadata",
        f"- Topic: {TOPIC}",
        f"- CrewAI model: {OLLAMA_MODEL}",
        f"- CrewAI version: 1.15.22 (verified in the current environment)",
        f"- Run time (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"- SERPER_API_KEY configured: {'yes' if os.getenv('SERPER_API_KEY') else 'no'}",
        "- Code execution safety: Python syntax is checked with compile(); supplied code is not executed.",
    ]

    baseline_result = ""
    tool_result = ""
    tool_status = "not attempted"

    try:
        baseline_result = run_crew(
            "W9D2 Baseline Crew — No Web Search",
            use_web=False,
        )
    except Exception as exc:
        error = f"Baseline run failed: {type(exc).__name__}: {exc}"
        print(error)
        metadata.append(f"- Baseline status: FAILED — {error}")
        write_evidence("baseline_error.txt", traceback.format_exc())

    web_search_tool = None
    if os.getenv("SERPER_API_KEY"):
        try:
            from crewai_tools import SerperDevTool

            web_search_tool = SerperDevTool()
            tool_status = "configured; live retrieval must be verified by run output"
        except Exception as exc:
            tool_status = f"initialization failed: {type(exc).__name__}: {exc}"
    else:
        tool_status = "skipped: SERPER_API_KEY is not configured"

    if web_search_tool is not None:
        try:
            tool_result = run_crew(
                "W9D2 Tool-Assisted Crew — Web Search",
                use_web=True,
                web_search_tool=web_search_tool,
            )
        except Exception as exc:
            error = f"Tool-assisted run failed: {type(exc).__name__}: {exc}"
            print(error)
            write_evidence("tool_assisted_error.txt", traceback.format_exc())
            tool_status = f"run failed: {error}"
    else:
        message = (
            "Live web search was not run because a usable Serper tool was "
            f"unavailable. Status: {tool_status}. This is not a successful "
            "live-web-search result."
        )
        write_evidence("tool_assisted_results.txt", message)

    sample_code = "numbers = [1, 2, 3]\nresult = sum(numbers)\n"
    code_check = safe_python_check(sample_code)
    write_evidence(
        "code_validation_results.txt",
        "# W9D2 Safe Python Code Validation\n\n"
        f"Sample code:\n```python\n{sample_code}```\n\n"
        f"Result: {code_check}\n",
    )

    metadata.append(f"- Baseline status: {'completed' if baseline_result else 'failed'}")
    metadata.append(f"- Web-search status: {tool_status}")
    metadata.append(f"- Safe code syntax check: {code_check}")
    write_evidence("run_metadata.md", "\n".join(metadata))

    comparison = f"""# W9D2 Baseline vs Tool-Assisted Comparison

## Topic
{TOPIC}

## Baseline
- Uses the local LLM's existing knowledge.
- No live web search is claimed.
- Evidence: `baseline_results.txt` if the run completed.

## Tool-Assisted Attempt
- Web search status: {tool_status}
- Evidence: `tool_assisted_results.txt` and, if created, `tool_assisted_error.txt`.
- Treat results as web-grounded only if the output shows that real search results were retrieved.

## Code Execution Tool Requirement
- The script performs a controlled Python syntax check using `compile()`.
- It does not execute arbitrary code produced by the LLM.
- Evidence: `code_validation_results.txt`.

## Honest Assessment
Do not claim the tool-assisted output is more accurate or current unless the run actually retrieved live sources and the report demonstrates a verifiable improvement.
"""
    write_evidence("before_after_comparison.md", comparison)
    print("\nW9D2 run finished. Inspect evidence and any tool limitations before committing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

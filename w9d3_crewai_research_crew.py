
"""W9D3: Building a Research Crew — Multi-Agent Pipeline."""

from __future__ import annotations

import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Type

from pydantic import BaseModel, Field
from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import BaseTool

BASE_DIR = Path(__file__).resolve().parent
EVIDENCE_DIR = BASE_DIR / "output_evidence" / "w9d3"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

TOPIC = (
    "Recent developments in generative AI and their practical "
    "applications in software development"
)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

SEARCH_EVENTS: list[str] = []

llm = LLM(
    model=f"ollama/{OLLAMA_MODEL}",
    base_url=OLLAMA_BASE_URL,
    temperature=0.2,
)


def save_evidence(filename: str, content: str) -> None:
    """Save UTF-8 evidence without overwriting other days' evidence."""
    path = EVIDENCE_DIR / filename
    path.write_text(content.rstrip() + "\n", encoding="utf-8")
    print(f"Evidence saved: {path.relative_to(BASE_DIR)}")


class SearchInput(BaseModel):
    """Arguments accepted by the web search tool."""

    query: str = Field(
        ...,
        description="A concise search query about generative AI.",
    )


class DuckDuckGoSearchTool(BaseTool):
    """Search the web without requiring a Serper API key."""

    name: str = "DuckDuckGo Web Search"
    description: str = (
        "Search the web for current information. Returns result titles, "
        "snippets, and URLs. Use the results as evidence and never invent "
        "sources or claim a result was retrieved if it was not."
    )
    args_schema: Type[BaseModel] = SearchInput

    def _run(self, query: str) -> str:
        try:
            from ddgs import DDGS

            with DDGS() as search:
                results = list(search.text(query, max_results=5))

            if not results:
                message = f"NO_RESULTS: No results returned for: {query}"
                SEARCH_EVENTS.append(message)
                return message

            formatted = []
            for item in results:
                title = item.get("title", "Untitled")
                url = item.get("href", "")
                snippet = item.get("body", "")

                formatted.append(
                    f"Title: {title}\nURL: {url}\nSnippet: {snippet}"
                )

            message = (
                f"SEARCH_SUCCEEDED: {len(results)} results for {query}\n\n"
                + "\n\n".join(formatted)
            )
            SEARCH_EVENTS.append(message)
            return message

        except Exception as exc:
            message = (
                f"SEARCH_FAILED: {type(exc).__name__}: {exc}"
            )
            SEARCH_EVENTS.append(message)
            return message


def create_agents(use_web: bool):
    """Create the Researcher, Writer, and Reviewer agents."""

    researcher = Agent(
        role="Researcher",
        goal=(
            "Find useful, accurate information about generative AI. "
            "Identify applications, benefits, limitations, and evidence."
        ),
        backstory=(
            "You are a careful technology researcher. You distinguish "
            "verified evidence from assumptions and never invent sources."
        ),
        tools=[DuckDuckGoSearchTool()] if use_web else [],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    writer = Agent(
        role="Writer",
        goal=(
            "Turn the research findings into a clear, structured "
            "technical report for software developers."
        ),
        backstory=(
            "You are an experienced technical writer who values clarity, "
            "practical examples, balanced analysis, and traceable sources."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    reviewer = Agent(
        role="Reviewer",
        goal=(
            "Evaluate the report for clarity, completeness, factual "
            "support, source quality, and unsupported claims."
        ),
        backstory=(
            "You are a skeptical research reviewer. You flag missing "
            "evidence, weak reasoning, and misleading conclusions."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    return researcher, writer, reviewer


def create_tasks(researcher, writer, reviewer, use_web: bool):
    """Create three linked tasks for the sequential pipeline."""

    if use_web:
        evidence_rule = (
            "Use the web search tool to retrieve relevant sources. Include "
            "the actual URLs returned by the tool. If search fails or "
            "returns no results, clearly disclose this limitation. Never "
            "invent URLs, citations, publication dates, or search findings."
        )
    else:
        evidence_rule = (
            "Use existing model knowledge only. Do not claim to have "
            "searched the web or verified current developments. Clearly "
            "identify claims that require current-source verification."
        )

    research_task = Task(
        description=(
            f"Research the following topic: {TOPIC}\n\n"
            "Identify at least three practical applications, benefits, "
            "limitations, and implementation considerations.\n"
            f"{evidence_rule}\n"
            "Distinguish sourced findings from general analysis."
        ),
        expected_output=(
            "Structured research notes covering applications, benefits, "
            "limitations, evidence, and source URLs when available."
        ),
        agent=researcher,
    )

    writing_task = Task(
        description=(
            f"Write a technical report about: {TOPIC}\n"
            "Use the research task's findings. Include a title, overview, "
            "at least three applications, benefits, limitations, "
            "implementation considerations, and conclusion. Preserve "
            "source URLs accurately. Do not invent citations."
        ),
        expected_output=(
            "A structured technical report with clear headings, practical "
            "examples, limitations, and honest evidence attribution."
        ),
        agent=writer,
        context=[research_task],
    )

    review_task = Task(
        description=(
            "Review the report and research notes. Assess clarity, "
            "completeness, factual support, relevance, and source quality. "
            "Identify unsupported or potentially outdated claims. For "
            "web research, check that URLs are present in the supplied "
            "research results; do not claim independent verification "
            "unless actually performed. Give concrete improvements and "
            "a final revised report."
        ),
        expected_output=(
            "A quality review with strengths, limitations, evidence "
            "gaps, suggested improvements, and a final revised report."
        ),
        agent=reviewer,
        context=[research_task, writing_task],
    )

    return [research_task, writing_task, review_task]


def run_pipeline(label: str, use_web: bool) -> str:
    """Run and save one three-agent research pipeline."""

    print(f"\n{'=' * 70}")
    print(label)
    print(f"{'=' * 70}")

    researcher, writer, reviewer = create_agents(use_web)
    tasks = create_tasks(
        researcher, writer, reviewer, use_web
    )

    crew = Crew(
        agents=[researcher, writer, reviewer],
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()
    output = str(result)

    filename = (
        "web_research_results.txt"
        if use_web
        else "baseline_results.txt"
    )

    save_evidence(
        filename,
        f"# {label}\n\n"
        f"Topic: {TOPIC}\n"
        f"Model: {OLLAMA_MODEL}\n"
        f"Run time (UTC): {datetime.now(timezone.utc).isoformat()}\n\n"
        f"{output}",
    )
    return output


def main() -> int:
    start_time = datetime.now(timezone.utc)
    baseline = ""
    web_result = ""
    baseline_status = "NOT RUN"
    web_status = "NOT RUN"
    errors = []

    try:
        baseline = run_pipeline(
            "W9D3 Baseline — No Web Search",
            use_web=False,
        )
        baseline_status = "SUCCESS"
    except Exception as exc:
        baseline_status = f"FAILED: {type(exc).__name__}: {exc}"
        errors.append("BASELINE\n" + traceback.format_exc())
        print(baseline_status)

    try:
        web_result = run_pipeline(
            "W9D3 Web-Enhanced Research Crew",
            use_web=True,
        )
        web_status = "PIPELINE COMPLETED"
    except Exception as exc:
        web_status = f"FAILED: {type(exc).__name__}: {exc}"
        errors.append("WEB PIPELINE\n" + traceback.format_exc())
        print(web_status)

    successful_searches = sum(
        "SEARCH_SUCCEEDED:" in event for event in SEARCH_EVENTS
    )
    failed_searches = sum(
        "SEARCH_FAILED:" in event or "NO_RESULTS:" in event
        for event in SEARCH_EVENTS
    )

    search_log = [
        "# W9D3 Web Search Evidence",
        f"Search calls recorded: {len(SEARCH_EVENTS)}",
        f"Successful search calls: {successful_searches}",
        f"Failed or empty search calls: {failed_searches}",
        "",
    ]
    search_log.extend(SEARCH_EVENTS or ["No search calls were recorded."])
    save_evidence("web_search_log.txt", "\n\n".join(search_log))

    comparison = f"""# W9D3 Baseline vs Web-Enhanced Comparison

## Topic
{TOPIC}

## Baseline status
{baseline_status}

## Web-enhanced pipeline status
{web_status}

## Search evidence
- Successful search calls recorded: {successful_searches}
- Failed or empty search calls recorded: {failed_searches}
- Search details: `web_search_log.txt`
- Web report: `web_research_results.txt`
- Baseline report: `baseline_results.txt`

## Evaluation criteria
Compare both reports for:
1. Relevance and coverage of applications.
2. Evidence quality and presence of usable source URLs.
3. Specificity and practical usefulness.
4. Treatment of limitations and uncertainty.
5. Unsupported claims and clarity.

## Honest conclusion
A completed web-enhanced crew run does not by itself prove that
the report improved. Claim a source-grounding improvement only if
the search log confirms actual results and the output contains
relevant source URLs. Record quality improvements only when supported
by the two saved reports. Search results and URLs have not necessarily
been independently verified.

## Run metadata
- Model: {OLLAMA_MODEL}
- Started (UTC): {start_time.isoformat()}
- Completed (UTC): {datetime.now(timezone.utc).isoformat()}
"""
    save_evidence("comparison_report.md", comparison)

    metadata = f"""# W9D3 Run Metadata

- Topic: {TOPIC}
- Model: {OLLAMA_MODEL}
- CrewAI pipeline: sequential
- Agents: Researcher, Writer, Reviewer
- Baseline status: {baseline_status}
- Web pipeline status: {web_status}
- Successful search calls: {successful_searches}
- Failed or empty search calls: {failed_searches}
- Baseline output saved: {bool(baseline)}
- Web output saved: {bool(web_result)}
- Python version: {sys.version.split()[0]}
- Run completed (UTC): {datetime.now(timezone.utc).isoformat()}
"""
    save_evidence("run_metadata.md", metadata)

    if errors:
        save_evidence("run_errors.txt", "\n\n".join(errors))

    if not baseline or not web_result or successful_searches == 0:
        print(
            "\nPipeline finished with incomplete deliverables. "
            "Inspect run_metadata.md, run_errors.txt if present, "
            "and web_search_log.txt."
        )
        return 1

    print("\nBoth pipelines ran and web search returned results.")
    print("Inspect the comparison before claiming an improvement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

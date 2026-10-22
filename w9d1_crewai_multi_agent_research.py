
"""
W9D1: CrewAI Fundamentals — Agents, Tasks & Crews

A sequential multi-agent research crew with:
1. Researcher agent
2. Writer agent
3. Reviewer agent

The script supports:
- Baseline run without web search
- Enhanced run with Serper web search
- Before/after comparison evidence
"""

from pathlib import Path
import os

from crewai import Agent, Crew, LLM, Process, Task
from crewai_tools import SerperDevTool


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
EVIDENCE_DIR = BASE_DIR / "output_evidence" / "w9d1"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

TOPIC = "Recent developments in Generative AI"

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

llm = LLM(
    model=f"ollama/{OLLAMA_MODEL}",
    base_url=OLLAMA_BASE_URL,
    temperature=0.2,
)


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

def create_agents(web_search_tool=None):
    """Create the Researcher, Writer and Reviewer agents."""

    researcher_tools = [web_search_tool] if web_search_tool else []

    researcher = Agent(
        role="AI Researcher",
        goal=(
            "Research the assigned topic carefully and identify accurate, "
            "useful and relevant information."
        ),
        backstory=(
            "You are a careful AI/ML researcher who collects factual "
            "information, identifies important developments and organizes "
            "research findings clearly."
        ),
        tools=researcher_tools,
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    writer = Agent(
        role="Technical Writer",
        goal=(
            "Transform research findings into a clear, structured and "
            "easy-to-understand technical report."
        ),
        backstory=(
            "You are an experienced technical writer who explains complex "
            "AI concepts in a simple and organized manner while preserving "
            "important technical details."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    reviewer = Agent(
        role="Research Reviewer",
        goal=(
            "Review the research report for accuracy, completeness, "
            "clarity and unsupported claims."
        ),
        backstory=(
            "You are a strict but constructive reviewer who checks whether "
            "research reports are logically organized, sufficiently "
            "supported and useful to readers."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    return researcher, writer, reviewer


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

def create_tasks(researcher, writer, reviewer, web_enabled=False):
    """Create the three sequential research tasks."""

    source_instruction = (
        "Use the available web search tool to obtain current information "
        "from real web sources. Clearly distinguish sourced facts from "
        "general background knowledge. Mention the sources used where "
        "appropriate."
        if web_enabled
        else
        "Perform the research using your existing knowledge. Do not claim "
        "to have accessed live web sources."
    )

    research_task = Task(
        description=(
            f"Research the topic: '{TOPIC}'.\n\n"
            f"{source_instruction}\n\n"
            "Identify at least five important developments, concepts or "
            "trends. For each item, provide a short explanation and explain "
            "why it matters. Keep the findings factual and concise."
        ),
        expected_output=(
            "A structured research brief containing at least five relevant "
            "findings about recent Generative AI developments."
        ),
        agent=researcher,
    )

    writing_task = Task(
        description=(
            f"Using the research findings about '{TOPIC}', write a concise "
            "technical report.\n\n"
            "Organize the report with:\n"
            "1. Introduction\n"
            "2. Major developments\n"
            "3. Practical significance\n"
            "4. Conclusion\n\n"
            "Use clear language and do not invent facts or sources."
        ),
        expected_output=(
            "A well-structured technical report based directly on the "
            "researcher's findings."
        ),
        agent=writer,
        context=[research_task],
    )

    review_task = Task(
        description=(
            "Review the technical report produced by the writer.\n\n"
            "Check:\n"
            "- factual consistency\n"
            "- completeness\n"
            "- clarity\n"
            "- logical organization\n"
            "- unsupported or overly broad claims\n\n"
            "List specific strengths, weaknesses and improvements. "
            "Finish with a concise overall review."
        ),
        expected_output=(
            "A structured quality review containing strengths, weaknesses, "
            "recommended improvements and an overall assessment."
        ),
        agent=reviewer,
        context=[research_task, writing_task],
    )

    return research_task, writing_task, review_task


# ---------------------------------------------------------------------------
# Crew execution
# ---------------------------------------------------------------------------

def run_crew(web_enabled=False):
    """Run the research crew."""

    web_search_tool = None

    if web_enabled:
        if not os.getenv("SERPER_API_KEY"):
            raise RuntimeError(
                "SERPER_API_KEY is not configured. "
                "Set a valid Serper API key before running the "
                "web-enabled crew."
            )

        web_search_tool = SerperDevTool()

    researcher, writer, reviewer = create_agents(web_search_tool)

    research_task, writing_task, review_task = create_tasks(
        researcher,
        writer,
        reviewer,
        web_enabled=web_enabled,
    )

    crew = Crew(
        agents=[researcher, writer, reviewer],
        tasks=[research_task, writing_task, review_task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()

    return result


# ---------------------------------------------------------------------------
# Evidence helpers
# ---------------------------------------------------------------------------

def save_result(filename, result):
    """Save a crew result as evidence."""

    output_path = EVIDENCE_DIR / filename
    output_path.write_text(str(result), encoding="utf-8")
    return output_path


def save_comparison(baseline_result, web_result):
    """Save a concise before/after comparison."""

    comparison = f"""# W9D1 Before/After Comparison

## Baseline: Knowledge-Only Crew

The baseline crew performed the research without live web search.

### Baseline Result

{baseline_result}

---

## Enhanced: Web-Enabled Crew

The enhanced crew used the Serper web-search tool to retrieve current
information from web sources.

### Web-Enabled Result

{web_result}

---

## Observed Improvement

The baseline demonstrates how a multi-agent CrewAI workflow can research,
write and review a topic using the model's existing knowledge.

The web-enabled workflow adds access to current web information. This makes
the research process better suited to topics where freshness and source
verification are important.

The comparison should be interpreted as a workflow demonstration rather than
as a quantitative benchmark. The two runs may produce different findings,
wording and source coverage because the web-enabled run has access to
current external information.
"""

    return save_result("before_after_comparison.md", comparison)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("W9D1 — CrewAI Multi-Agent Research Crew")
    print("=" * 70)
    print(f"Topic: {TOPIC}")
    print(f"Ollama model: {OLLAMA_MODEL}")
    print()

    # --------------------------------------------------------------
    # Baseline run
    # --------------------------------------------------------------

    print("Running baseline crew without web search...")
    baseline_result = run_crew(web_enabled=False)

    baseline_path = save_result(
        "baseline_research_results.txt",
        baseline_result,
    )

    print()
    print(f"Baseline evidence saved to: {baseline_path}")
    print()
    print("Baseline run completed successfully.")

    # --------------------------------------------------------------
    # Web-enabled run
    # --------------------------------------------------------------

    print()
    print("Running web-enabled crew with Serper search...")

    web_result = run_crew(web_enabled=True)

    web_path = save_result(
        "web_enabled_research_results.txt",
        web_result,
    )

    comparison_path = save_comparison(
        baseline_result,
        web_result,
    )

    print()
    print(f"Web-enabled evidence saved to: {web_path}")
    print(f"Before/after comparison saved to: {comparison_path}")
    print()
    print("W9D1 baseline and web-enabled executions completed successfully.")


if __name__ == "__main__":
    main()


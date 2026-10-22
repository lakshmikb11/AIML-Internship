
import pytest

from w9d5_automated_research_report import (
    ResearchState,
    build_workflow,
    format_sources,
    validate_report,
)


def test_format_sources_with_results():
    sources = [
        {
            "title": "Example Research",
            "url": "https://example.com/research",
            "snippet": "A research summary.",
        }
    ]

    result = format_sources(sources)

    assert "Example Research" in result
    assert "https://example.com/research" in result
    assert "A research summary." in result


def test_format_sources_empty():
    result = format_sources([])

    assert isinstance(result, str)
    assert result.strip()


def test_validate_report_accepts_nonempty_report():
    report = (
        "# Research Report\n\n"
        "## Summary\nA useful summary.\n\n"
        "## Sources\n- https://example.com\n"
    )

    assert validate_report(report) is True


def test_validate_report_rejects_empty_report():
    assert validate_report("") is False
    assert validate_report("   ") is False


def test_validate_report_rejects_short_report():
    assert validate_report("Too short") is False


def test_ragas_no_sources_fallback():
    from w9d5_automated_research_report import evaluate_report

    result = evaluate_report(
        report="A sample research report with enough content.",
        sources=[],
    )

    assert isinstance(result, dict)


def test_langgraph_workflow_compiles():
    workflow = build_workflow()

    assert workflow is not None
    assert callable(workflow.invoke)

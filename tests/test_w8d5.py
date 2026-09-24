"""Tests for the W8D5 Local AI Research Assistant."""

from pathlib import Path

from w8d5_local_ai_research_assistant import (
    build_prompt,
    load_documents,
    retrieve_documents,
    tokenize,
)


def test_load_documents():
    """All W8D5 research notes should load successfully."""
    documents = load_documents()

    assert len(documents) == 3
    assert "python_research.txt" in documents
    assert "machine_learning_research.txt" in documents
    assert "rag_research.txt" in documents


def test_tokenize():
    """Tokenization should normalize words."""
    tokens = tokenize("Python, Machine Learning!")

    assert "python" in tokens
    assert "machine" in tokens
    assert "learning" in tokens


def test_retrieval_returns_relevant_document():
    """RAG-related questions should retrieve the RAG notes."""
    documents = load_documents()

    results = retrieve_documents(
        "What is retrieval augmented generation?",
        documents,
    )

    assert results
    assert results[0][0] == "rag_research.txt"


def test_prompt_contains_question_and_source():
    """Prompts should contain the question and source context."""
    retrieved = [
        (
            "rag_research.txt",
            "RAG combines retrieval with generation.",
            3,
        )
    ]

    prompt = build_prompt("What is RAG?", retrieved)

    assert "What is RAG?" in prompt
    assert "rag_research.txt" in prompt
    assert "RAG combines retrieval with generation." in prompt


def test_evidence_directory_structure():
    """The W8D5 evidence directory should be well-defined."""
    evidence_path = (
        Path(__file__).resolve().parents[1]
        / "output_evidence"
        / "w8d5"
    )

    assert evidence_path.name == "w8d5"
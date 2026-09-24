"""
W8D5 - Local AI Research Assistant

A small local AI research assistant using a 3M-style architecture:

1. Model      - Ollama local language model
2. Memory     - Local research-note retrieval
3. MLOps      - JSON run metadata for reproducibility

The assistant retrieves relevant local notes before asking the local
LLM to generate an answer. No external LLM API is required.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "w8d5_data"
EVIDENCE_DIR = BASE_DIR / "output_evidence" / "w8d5"
RUN_LOG = EVIDENCE_DIR / "run_metadata.json"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"


def load_documents(data_dir: Path = DATA_DIR) -> dict[str, str]:
    """Load all local research notes."""
    documents = {}

    for path in sorted(data_dir.glob("*.txt")):
        documents[path.name] = path.read_text(encoding="utf-8")

    return documents


def tokenize(text: str) -> set[str]:
    """Convert text into normalized word tokens."""
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def retrieve_documents(
    question: str,
    documents: dict[str, str],
    top_k: int = 2,
) -> list[tuple[str, str, int]]:
    """
    Retrieve documents using simple keyword overlap.

    Returns:
        (filename, document_text, score)
    """
    question_tokens = tokenize(question)
    ranked = []

    for filename, content in documents.items():
        document_tokens = tokenize(content)
        score = len(question_tokens & document_tokens)

        if score > 0:
            ranked.append((filename, content, score))

    ranked.sort(key=lambda item: item[2], reverse=True)

    return ranked[:top_k]


def build_prompt(question: str, retrieved: list[tuple[str, str, int]]) -> str:
    """Build a grounded prompt from retrieved local context."""
    context_parts = []

    for filename, content, _score in retrieved:
        context_parts.append(f"SOURCE: {filename}\n{content}")

    context = "\n\n".join(context_parts)

    return f"""
You are a local AI research assistant.

Answer the user's research question using ONLY the supplied local
research notes.

If the notes do not contain enough information, clearly say that the
available research notes do not provide enough information.

Keep the answer concise and factual.

Research question:
{question}

Local research notes:
{context}

Answer:
""".strip()


def generate_with_ollama(prompt: str, model: str = MODEL_NAME) -> str:
    """Generate an answer using the local Ollama server."""
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
    ).encode("utf-8")

    request = Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
    except URLError as exc:
        raise RuntimeError(
            "Could not connect to Ollama. Make sure Ollama is running."
        ) from exc

    return result.get("response", "").strip()


def save_run_metadata(
    question: str,
    retrieved: list[tuple[str, str, int]],
    latency_seconds: float,
    success: bool,
) -> None:
    """Save basic MLOps/reproducibility metadata."""
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    metadata = {
        "model": MODEL_NAME,
        "question": question,
        "retrieved_sources": [
            {"file": filename, "score": score}
            for filename, _content, score in retrieved
        ],
        "retrieval_count": len(retrieved),
        "latency_seconds": round(latency_seconds, 3),
        "success": success,
    }

    RUN_LOG.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def run_assistant(question: str) -> dict:
    """Run the complete local research assistant pipeline."""
    documents = load_documents()

    if not documents:
        raise RuntimeError("No research documents were found.")

    retrieved = retrieve_documents(question, documents)

    if not retrieved:
        return {
            "answer": (
                "The available local research notes do not contain "
                "enough information to answer this question."
            ),
            "sources": [],
        }

    prompt = build_prompt(question, retrieved)

    start = time.perf_counter()
    answer = generate_with_ollama(prompt)
    latency = time.perf_counter() - start

    save_run_metadata(
        question=question,
        retrieved=retrieved,
        latency_seconds=latency,
        success=True,
    )

    return {
        "answer": answer,
        "sources": [filename for filename, _content, _score in retrieved],
        "latency_seconds": round(latency, 3),
    }


def main() -> None:
    """Run a demonstration question."""
    question = (
        "What is RAG and how can a local AI assistant use retrieved "
        "research notes?"
    )

    result = run_assistant(question)

    print("\n=== W8D5 Local AI Research Assistant ===")
    print(f"Model: {MODEL_NAME}")
    print(f"Question: {question}")
    print("\nAnswer:")
    print(result["answer"])

    print("\nSources:")
    for source in result["sources"]:
        print(f"- {source}")

    if "latency_seconds" in result:
        print(f"\nGeneration latency: {result['latency_seconds']} seconds")


if __name__ == "__main__":
    main()
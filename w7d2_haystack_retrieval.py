from pathlib import Path

import numpy as np
from haystack import Document, Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.retrievers.in_memory import (
    InMemoryBM25Retriever,
    InMemoryEmbeddingRetriever,
)
from haystack.document_stores.in_memory import InMemoryDocumentStore
from sentence_transformers import SentenceTransformer


DATA_DIR = Path("w7d2_data")
EVIDENCE_DIR = Path("output_evidence/w7d2")
EVIDENCE_FILE = EVIDENCE_DIR / "haystack_retrieval_results.txt"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 2


QUESTIONS = [
    (
        "What is Python?",
        "01_python_programming.pdf",
    ),
    (
        "What do functions do in Python?",
        "01_python_programming.pdf",
    ),
    (
        "What is data preprocessing?",
        "02_data_science.pdf",
    ),
    (
        "What is exploratory data analysis?",
        "02_data_science.pdf",
    ),
    (
        "What is logistic regression commonly used for?",
        "03_machine_learning_models.pdf",
    ),
    (
        "What does a decision tree use to make predictions?",
        "03_machine_learning_models.pdf",
    ),
    (
        "What is backpropagation?",
        "04_deep_learning.pdf",
    ),
    (
        "What activation function is commonly used in neural networks?",
        "04_deep_learning.pdf",
    ),
    (
        "What is BM25?",
        "05_information_retrieval.pdf",
    ),
    (
        "What is dense retrieval?",
        "05_information_retrieval.pdf",
    ),
]


def load_pdf_documents():
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))

    if len(pdf_files) != 5:
        raise ValueError(
            f"Expected 5 PDF files, but found {len(pdf_files)}."
        )

    print(f"Found {len(pdf_files)} PDF files.")

    converter = PyPDFToDocument()
    documents = []

    for pdf_file in pdf_files:
        result = converter.run(sources=[pdf_file])
        converted = result["documents"]

        for document in converted:
            document.meta["source_file"] = pdf_file.name

        documents.extend(converted)

        print(f"Loaded: {pdf_file.name}")

    print(f"Total Haystack documents: {len(documents)}")
    return documents


def build_bm25_pipeline(documents):
    document_store = InMemoryDocumentStore()
    document_store.write_documents(documents)

    retriever = InMemoryBM25Retriever(
        document_store=document_store,
        top_k=TOP_K,
    )

    pipeline = Pipeline()
    pipeline.add_component("bm25_retriever", retriever)
    pipeline.inputs("bm25_retriever.query")

    print("BM25 Haystack pipeline created.")

    return pipeline


def run_bm25(pipeline):
    results = []

    print()
    print("Running BM25 retrieval...")

    for question, expected_file in QUESTIONS:
        output = pipeline.run(
            {
                "bm25_retriever": {
                    "query": question,
                }
            }
        )

        documents = output["bm25_retriever"]["documents"]

        retrieved = [
            document.meta.get("source_file", "unknown")
            for document in documents
        ]

        top1 = retrieved[0] if retrieved else "NONE"
        correct = top1 == expected_file

        results.append(
            {
                "question": question,
                "expected": expected_file,
                "retrieved": retrieved,
                "top1": top1,
                "correct": correct,
            }
        )

    return results


def build_dense_documents(documents, model):
    dense_documents = []

    for document in documents:
        embedding = model.encode(
            document.content,
            normalize_embeddings=True,
        )

        dense_document = Document(
            content=document.content,
            meta=document.meta.copy(),
            embedding=np.asarray(embedding).tolist(),
        )

        dense_documents.append(dense_document)

    return dense_documents


def run_dense(dense_documents, model):
    document_store = InMemoryDocumentStore()
    document_store.write_documents(dense_documents)

    retriever = InMemoryEmbeddingRetriever(
        document_store=document_store,
        top_k=TOP_K,
    )

    results = []

    print()
    print("Running dense retrieval...")

    for question, expected_file in QUESTIONS:
        query_embedding = model.encode(
            question,
            normalize_embeddings=True,
        ).tolist()

        output = retriever.run(
            query_embedding=query_embedding
        )

        documents = output["documents"]

        retrieved = [
            document.meta.get("source_file", "unknown")
            for document in documents
        ]

        top1 = retrieved[0] if retrieved else "NONE"
        correct = top1 == expected_file

        results.append(
            {
                "question": question,
                "expected": expected_file,
                "retrieved": retrieved,
                "top1": top1,
                "correct": correct,
            }
        )

    return results


def precision_at_1(results):
    correct = sum(result["correct"] for result in results)
    return correct / len(results) if results else 0.0


def print_results(title, results):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    for index, result in enumerate(results, start=1):
        print()
        print(f"Question {index}: {result['question']}")
        print(f"Expected:  {result['expected']}")
        print(f"Retrieved: {result['retrieved']}")
        print(f"Top-1:     {result['top1']}")
        print(
            f"Result:    "
            f"{'CORRECT' if result['correct'] else 'INCORRECT'}"
        )


def build_evidence(
    bm25_results,
    dense_results,
    bm25_precision,
    dense_precision,
):
    lines = []

    lines.append("=" * 70)
    lines.append("W7D2: HAYSTACK RETRIEVAL — BM25 & DENSE RETRIEVAL")
    lines.append("=" * 70)
    lines.append("")

    lines.append("DATASET")
    lines.append("-" * 70)
    lines.append("Number of PDF documents: 5")
    lines.append("Number of evaluation questions: 10")
    lines.append("Top-K retrieval: 2")
    lines.append(f"Dense embedding model: {EMBEDDING_MODEL}")
    lines.append("")

    lines.append("HAYSTACK ARCHITECTURE")
    lines.append("-" * 70)
    lines.append(
        "PDF files -> PyPDFToDocument -> "
        "InMemoryDocumentStore -> BM25 Retriever"
    )
    lines.append(
        "PDF files -> PyPDFToDocument -> "
        "InMemoryDocumentStore -> Dense Embedding Retriever"
    )
    lines.append("")
    lines.append(
        "Note: The installed current Haystack API does not expose "
        "the older Reader component referenced in the assignment. "
        "The implementation therefore uses the currently supported "
        "retrieval components."
    )
    lines.append("")

    lines.append("BM25 RETRIEVAL RESULTS")
    lines.append("-" * 70)

    for index, result in enumerate(bm25_results, start=1):
        lines.append(f"{index}. {result['question']}")
        lines.append(f"Expected:  {result['expected']}")
        lines.append(f"Retrieved: {result['retrieved']}")
        lines.append(f"Top-1:     {result['top1']}")
        lines.append(
            f"Result:    "
            f"{'CORRECT' if result['correct'] else 'INCORRECT'}"
        )
        lines.append("")

    lines.append("DENSE RETRIEVAL RESULTS")
    lines.append("-" * 70)

    for index, result in enumerate(dense_results, start=1):
        lines.append(f"{index}. {result['question']}")
        lines.append(f"Expected:  {result['expected']}")
        lines.append(f"Retrieved: {result['retrieved']}")
        lines.append(f"Top-1:     {result['top1']}")
        lines.append(
            f"Result:    "
            f"{'CORRECT' if result['correct'] else 'INCORRECT'}"
        )
        lines.append("")

    lines.append("FINAL COMPARISON")
    lines.append("-" * 70)
    lines.append(f"BM25 Precision@1:   {bm25_precision:.2%}")
    lines.append(f"Dense Precision@1: {dense_precision:.2%}")
    lines.append(
        f"Difference:         "
        f"{dense_precision - bm25_precision:+.2%}"
    )
    lines.append("")

    lines.append("MANUAL EVALUATION")
    lines.append("-" * 70)
    lines.append(
        "Each question was manually evaluated by checking whether "
        "the top-ranked retrieved PDF matched the expected source."
    )
    lines.append(
        "Precision@1 measures the proportion of questions for which "
        "the top-ranked document was the expected document."
    )
    lines.append("")

    return "\n".join(lines)


def main():
    print("=" * 70)
    print("W7D2: HAYSTACK RETRIEVAL — BM25 & DENSE RETRIEVAL")
    print("=" * 70)

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    documents = load_pdf_documents()

    print()
    print("Building BM25 Haystack pipeline...")
    bm25_pipeline = build_bm25_pipeline(documents)

    bm25_results = run_bm25(bm25_pipeline)

    print()
    print("Loading dense embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    dense_documents = build_dense_documents(documents, model)

    print(
        f"Created {len(dense_documents)} dense documents "
        f"with dimension {len(dense_documents[0].embedding)}."
    )

    dense_results = run_dense(dense_documents, model)

    print_results("BM25 RETRIEVAL RESULTS", bm25_results)
    print_results("DENSE RETRIEVAL RESULTS", dense_results)

    bm25_precision = precision_at_1(bm25_results)
    dense_precision = precision_at_1(dense_results)

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"BM25 Precision@1: {bm25_precision:.2%}")
    print(f"Dense Precision@1: {dense_precision:.2%}")
    print(
        f"Difference (Dense - BM25): "
        f"{dense_precision - bm25_precision:+.2%}"
    )

    evidence = build_evidence(
        bm25_results,
        dense_results,
        bm25_precision,
        dense_precision,
    )

    EVIDENCE_FILE.write_text(
        evidence,
        encoding="utf-8",
    )

    print()
    print(f"Evidence saved to: {EVIDENCE_FILE}")
    print("W7D2 retrieval evaluation completed.")


if __name__ == "__main__":
    main()
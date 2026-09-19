"""
W7D1: Haystack Pipeline Architecture

Tasks:
1. Convert 5 PDF documents using Haystack.
2. Store documents in a Haystack InMemoryDocumentStore.
3. Build and run a Haystack BM25 pipeline.
4. Run 10 retrieval questions.
5. Evaluate BM25 retrieval quality.
6. Replace BM25 with dense retrieval.
7. Compare BM25 and dense Precision@1.
8. Save evaluation evidence.
"""

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


# ================================================================
# CONFIGURATION
# ================================================================

DATA_DIR = Path("w7d1_data")
EVIDENCE_DIR = Path("output_evidence") / "w7d1"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 2


# ================================================================
# 10 EVALUATION QUESTIONS
# ================================================================

QUESTIONS = [
    {
        "question": "What is supervised learning?",
        "expected": "01_machine_learning_fundamentals.pdf",
    },
    {
        "question": "What is overfitting in machine learning?",
        "expected": "01_machine_learning_fundamentals.pdf",
    },
    {
        "question": "What activation function is commonly used in hidden layers of neural networks?",
        "expected": "02_neural_networks.pdf",
    },
    {
        "question": "What does backpropagation calculate?",
        "expected": "02_neural_networks.pdf",
    },
    {
        "question": "What is tokenization in natural language processing?",
        "expected": "03_natural_language_processing.pdf",
    },
    {
        "question": "What do transformers use to model relationships between tokens?",
        "expected": "03_natural_language_processing.pdf",
    },
    {
        "question": "What is object detection in computer vision?",
        "expected": "04_computer_vision.pdf",
    },
    {
        "question": "What is image segmentation?",
        "expected": "04_computer_vision.pdf",
    },
    {
        "question": "What is BM25?",
        "expected": "05_retrieval_augmented_generation.pdf",
    },
    {
        "question": "What is dense retrieval?",
        "expected": "05_retrieval_augmented_generation.pdf",
    },
]


# ================================================================
# LOAD AND CONVERT PDF DOCUMENTS
# ================================================================

def load_pdfs():
    """Convert the five PDF files into Haystack Documents."""

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))

    if len(pdf_files) != 5:
        raise ValueError(
            f"Expected 5 PDF files, but found {len(pdf_files)}."
        )

    print(f"Found {len(pdf_files)} PDF files.")

    converter = PyPDFToDocument()
    documents = []

    for pdf_path in pdf_files:
        result = converter.run(
            sources=[pdf_path]
        )

        converted_documents = result["documents"]

        for document in converted_documents:
            document.meta["source_file"] = pdf_path.name
            documents.append(document)

        print(f"Loaded: {pdf_path.name}")

    print(
        f"Total Haystack documents: {len(documents)}"
    )

    return documents


# ================================================================
# BUILD BM25 HAYSTACK PIPELINE
# ================================================================

def build_bm25_pipeline(documents):
    """
    Build an explicit Haystack Pipeline.

    Architecture:
        Query -> BM25Retriever -> Retrieved Documents

    The DocumentStore is attached to the retriever.
    PDF conversion has already been completed by PyPDFToDocument.
    """

    print("\nBuilding BM25 Haystack pipeline...")

    document_store = InMemoryDocumentStore()

    document_store.write_documents(documents)

    retriever = InMemoryBM25Retriever(
        document_store=document_store,
        top_k=TOP_K,
    )

    pipeline = Pipeline()

    pipeline.add_component(
        "bm25_retriever",
        retriever,
    )

    pipeline.inputs(
        "bm25_retriever.query"
    )

    print("BM25 Haystack pipeline created.")

    return pipeline


# ================================================================
# RUN BM25 RETRIEVAL
# ================================================================

def run_bm25(pipeline):
    """Run the BM25 Haystack pipeline on all questions."""

    print("\nRunning BM25 retrieval...")

    results = []

    for item in QUESTIONS:

        question = item["question"]
        expected = item["expected"]

        output = pipeline.run(
            {
                "bm25_retriever": {
                    "query": question,
                    "top_k": TOP_K,
                }
            }
        )

        retrieved_documents = output[
            "bm25_retriever"
        ]["documents"]

        retrieved_sources = [
            document.meta.get(
                "source_file",
                "unknown",
            )
            for document in retrieved_documents
        ]

        top1_source = (
            retrieved_sources[0]
            if retrieved_sources
            else "NO_RESULT"
        )

        correct = (
            top1_source == expected
        )

        results.append(
            {
                "question": question,
                "expected": expected,
                "retrieved": retrieved_sources,
                "top1": top1_source,
                "correct": correct,
            }
        )

    return results


# ================================================================
# CREATE DENSE DOCUMENTS
# ================================================================

def create_dense_documents(documents):
    """Create Haystack Documents containing dense embeddings."""

    print("\nPreparing dense documents...")
    print("\nLoading dense embedding model...")

    model = SentenceTransformer(
        MODEL_NAME
    )

    texts = [
        document.content
        for document in documents
    ]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
    )

    dense_documents = []

    for document, embedding in zip(
        documents,
        embeddings,
    ):

        dense_document = Document(
            content=document.content,
            meta=document.meta.copy(),
            embedding=np.asarray(
                embedding,
                dtype=np.float32,
            ).tolist(),
        )

        dense_documents.append(
            dense_document
        )

    dimension = len(
        dense_documents[0].embedding
    )

    print(
        f"Created {len(dense_documents)} "
        f"dense documents with dimension "
        f"{dimension}."
    )

    return dense_documents, model


# ================================================================
# RUN DENSE RETRIEVAL
# ================================================================

def run_dense(documents, model):
    """Run dense retrieval on all ten questions."""

    print("\nRunning dense retrieval...")

    document_store = InMemoryDocumentStore()

    document_store.write_documents(
        documents
    )

    retriever = InMemoryEmbeddingRetriever(
        document_store=document_store,
        top_k=TOP_K,
    )

    results = []

    for item in QUESTIONS:

        question = item["question"]
        expected = item["expected"]

        # Haystack expects a Python list of floats.
        query_embedding = model.encode(
            question,
            normalize_embeddings=True,
        ).tolist()

        output = retriever.run(
            query_embedding=query_embedding,
            top_k=TOP_K,
        )

        retrieved_documents = output[
            "documents"
        ]

        retrieved_sources = [
            document.meta.get(
                "source_file",
                "unknown",
            )
            for document in retrieved_documents
        ]

        top1_source = (
            retrieved_sources[0]
            if retrieved_sources
            else "NO_RESULT"
        )

        correct = (
            top1_source == expected
        )

        results.append(
            {
                "question": question,
                "expected": expected,
                "retrieved": retrieved_sources,
                "top1": top1_source,
                "correct": correct,
            }
        )

    return results


# ================================================================
# PRECISION@1
# ================================================================

def calculate_precision_at_1(results):
    """Calculate Precision@1."""

    if not results:
        return 0.0

    correct = sum(
        result["correct"]
        for result in results
    )

    return correct / len(results)


# ================================================================
# PRINT RESULTS
# ================================================================

def print_results(title, results):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    for index, result in enumerate(
        results,
        start=1,
    ):

        status = (
            "CORRECT"
            if result["correct"]
            else "INCORRECT"
        )

        print(
            f"\nQuestion {index}: "
            f"{result['question']}"
        )

        print(
            f"Expected:  "
            f"{result['expected']}"
        )

        print(
            f"Retrieved: "
            f"{result['retrieved']}"
        )

        print(
            f"Top-1:     "
            f"{result['top1']}"
        )

        print(
            f"Result:    "
            f"{status}"
        )


# ================================================================
# WRITE EVIDENCE
# ================================================================

def write_results(
    bm25_results,
    dense_results,
):

    EVIDENCE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        EVIDENCE_DIR
        / "haystack_retrieval_results.txt"
    )

    bm25_precision = (
        calculate_precision_at_1(
            bm25_results
        )
    )

    dense_precision = (
        calculate_precision_at_1(
            dense_results
        )
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "=" * 70 + "\n"
        )

        file.write(
            "W7D1: HAYSTACK RETRIEVAL EVALUATION\n"
        )

        file.write(
            "=" * 70 + "\n\n"
        )

        file.write(
            "HAYSTACK PIPELINE ARCHITECTURE\n"
        )

        file.write(
            "-" * 70 + "\n"
        )

        file.write(
            "PDF Converter: PyPDFToDocument\n"
        )

        file.write(
            "Document Store: InMemoryDocumentStore\n"
        )

        file.write(
            "BM25 Retriever: InMemoryBM25Retriever\n"
        )

        file.write(
            "Dense Retriever: InMemoryEmbeddingRetriever\n"
        )

        file.write(
            "Dense Model: "
            f"{MODEL_NAME}\n"
        )

        file.write(
            "\n"
        )

        file.write(
            "DATASET\n"
        )

        file.write(
            "-" * 70 + "\n"
        )

        file.write(
            "Number of PDF documents: 5\n"
        )

        file.write(
            "Number of evaluation questions: 10\n"
        )

        file.write(
            f"Dense embedding dimension: 384\n"
        )

        file.write(
            f"Top-K retrieval: {TOP_K}\n"
        )

        file.write(
            "\n"
        )

        # ----------------------------------------------------------
        # QUESTIONS
        # ----------------------------------------------------------

        file.write(
            "EVALUATION QUESTIONS\n"
        )

        file.write(
            "-" * 70 + "\n"
        )

        for index, item in enumerate(
            QUESTIONS,
            start=1,
        ):

            file.write(
                f"{index}. "
                f"{item['question']}\n"
            )

            file.write(
                f"   Expected source: "
                f"{item['expected']}\n"
            )

        # ----------------------------------------------------------
        # BM25
        # ----------------------------------------------------------

        file.write(
            "\n\nBM25 RETRIEVAL RESULTS\n"
        )

        file.write(
            "-" * 70 + "\n"
        )

        for index, result in enumerate(
            bm25_results,
            start=1,
        ):

            file.write(
                f"\nQuestion {index}: "
                f"{result['question']}\n"
            )

            file.write(
                f"Expected: "
                f"{result['expected']}\n"
            )

            file.write(
                f"Retrieved: "
                f"{', '.join(result['retrieved'])}\n"
            )

            file.write(
                f"Top-1: "
                f"{result['top1']}\n"
            )

            file.write(
                f"Correct: "
                f"{result['correct']}\n"
            )

        file.write(
            f"\nBM25 Precision@1: "
            f"{bm25_precision:.2%}\n"
        )

        # ----------------------------------------------------------
        # DENSE
        # ----------------------------------------------------------

        file.write(
            "\n\nDENSE RETRIEVAL RESULTS\n"
        )

        file.write(
            "-" * 70 + "\n"
        )

        for index, result in enumerate(
            dense_results,
            start=1,
        ):

            file.write(
                f"\nQuestion {index}: "
                f"{result['question']}\n"
            )

            file.write(
                f"Expected: "
                f"{result['expected']}\n"
            )

            file.write(
                f"Retrieved: "
                f"{', '.join(result['retrieved'])}\n"
            )

            file.write(
                f"Top-1: "
                f"{result['top1']}\n"
            )

            file.write(
                f"Correct: "
                f"{result['correct']}\n"
            )

        file.write(
            f"\nDense Precision@1: "
            f"{dense_precision:.2%}\n"
        )

        # ----------------------------------------------------------
        # COMPARISON
        # ----------------------------------------------------------

        file.write(
            "\n\nBM25 VS DENSE RETRIEVAL\n"
        )

        file.write(
            "-" * 70 + "\n"
        )

        file.write(
            f"BM25 Precision@1: "
            f"{bm25_precision:.2%}\n"
        )

        file.write(
            f"Dense Precision@1: "
            f"{dense_precision:.2%}\n"
        )

        file.write(
            f"Difference (Dense - BM25): "
            f"{dense_precision - bm25_precision:+.2%}\n"
        )

        # ----------------------------------------------------------
        # MANUAL EVALUATION
        # ----------------------------------------------------------

        file.write(
            "\n\nMANUAL EVALUATION NOTE\n"
        )

        file.write(
            "-" * 70 + "\n"
        )

        file.write(
            "Retrieval quality was evaluated using "
            "the top-1 retrieved document for each "
            "of the 10 questions. A result was marked "
            "correct when the top-ranked document "
            "matched the expected source PDF.\n"
        )

        file.write(
            "\n"
        )

        file.write(
            "The same ten questions were used for "
            "both BM25 and dense retrieval to provide "
            "a consistent comparison.\n"
        )

    print(
        f"\nEvidence saved to: "
        f"{output_file}"
    )


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 70)
    print("W7D1: HAYSTACK PIPELINE ARCHITECTURE")
    print("=" * 70)

    # --------------------------------------------------------------
    # 1. Convert five PDFs
    # --------------------------------------------------------------

    documents = load_pdfs()

    # --------------------------------------------------------------
    # 2. Build BM25 Haystack Pipeline
    # --------------------------------------------------------------

    bm25_pipeline = build_bm25_pipeline(
        documents
    )

    # --------------------------------------------------------------
    # 3. Run BM25 on ten questions
    # --------------------------------------------------------------

    bm25_results = run_bm25(
        bm25_pipeline
    )

    # --------------------------------------------------------------
    # 4. Create dense document embeddings
    # --------------------------------------------------------------

    dense_documents, model = (
        create_dense_documents(
            documents
        )
    )

    # --------------------------------------------------------------
    # 5. Run dense retrieval
    # --------------------------------------------------------------

    dense_results = run_dense(
        dense_documents,
        model,
    )

    # --------------------------------------------------------------
    # 6. Print results
    # --------------------------------------------------------------

    print_results(
        "BM25 RETRIEVAL RESULTS",
        bm25_results,
    )

    print_results(
        "DENSE RETRIEVAL RESULTS",
        dense_results,
    )

    # --------------------------------------------------------------
    # 7. Final metrics
    # --------------------------------------------------------------

    bm25_precision = (
        calculate_precision_at_1(
            bm25_results
        )
    )

    dense_precision = (
        calculate_precision_at_1(
            dense_results
        )
    )

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        f"BM25 Precision@1: "
        f"{bm25_precision:.2%}"
    )

    print(
        f"Dense Precision@1: "
        f"{dense_precision:.2%}"
    )

    print(
        f"Difference (Dense - BM25): "
        f"{dense_precision - bm25_precision:+.2%}"
    )

    # --------------------------------------------------------------
    # 8. Save evidence
    # --------------------------------------------------------------

    write_results(
        bm25_results,
        dense_results,
    )

    print(
        "\nW7D1 retrieval evaluation completed."
    )


if __name__ == "__main__":
    main()

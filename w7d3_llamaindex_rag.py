"""
W7D3: LlamaIndex — Document Indexing & RAG

Tasks:
1. Index text files using LlamaIndex VectorStoreIndex with Ollama embeddings.
2. Build a QueryEngine and run 10 queries.
3. Connect LlamaIndex to ChromaDB and re-run the same queries.
4. Compare query latency.
"""

from pathlib import Path
from time import perf_counter

import chromadb

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore


# ---------------------------------------------------------------------
# PATHS AND MODELS
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "w7d3_data"
EVIDENCE_DIR = BASE_DIR / "output_evidence" / "w7d3"
EVIDENCE_FILE = EVIDENCE_DIR / "llamaindex_rag_results.txt"

CHROMA_DIR = BASE_DIR / "w7d3_chroma_db"

LLM_MODEL = "llama3.2:3b"
EMBED_MODEL = "nomic-embed-text"


# ---------------------------------------------------------------------
# 10 EVALUATION QUESTIONS
# ---------------------------------------------------------------------

QUESTIONS = [
    "What is Python?",
    "What do functions do in Python?",
    "What is data preprocessing?",
    "What is exploratory data analysis?",
    "What is logistic regression?",
    "What does a decision tree use to make predictions?",
    "What is backpropagation?",
    "What is an activation function?",
    "What is BM25?",
    "What is dense retrieval?",
]


EXPECTED_SOURCES = [
    "01_python_programming.txt",
    "01_python_programming.txt",
    "02_data_science.txt",
    "02_data_science.txt",
    "03_machine_learning_models.txt",
    "03_machine_learning_models.txt",
    "04_deep_learning.txt",
    "04_deep_learning.txt",
    "05_information_retrieval.txt",
    "05_information_retrieval.txt",
]


# ---------------------------------------------------------------------
# CREATE SOURCE TEXT FILES
# ---------------------------------------------------------------------

def create_text_documents():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    documents = {
        "01_python_programming.txt": """
Python is a high-level, general-purpose programming language.
Python uses readable syntax and supports procedural, object-oriented,
and functional programming.

Functions are reusable blocks of code that perform a specific task.
They help organize programs, improve code reuse, and reduce repetition.
""",
        "02_data_science.txt": """
Data science combines statistics, programming, and domain knowledge
to extract useful information from data.

Data preprocessing prepares raw data for analysis by handling missing
values, duplicates, inconsistent formats, and other data quality issues.

Exploratory data analysis (EDA) uses summaries and visualizations to
understand patterns, distributions, relationships, and unusual
observations in a dataset.
""",
        "03_machine_learning_models.txt": """
Logistic regression is commonly used for classification problems,
especially binary classification. It estimates the probability of
a class.

A decision tree makes predictions by recursively splitting data using
features and decision rules. Splits can be selected using measures
such as Gini impurity or information gain.
""",
        "04_deep_learning.txt": """
Deep learning uses neural networks with multiple layers to learn
patterns from data.

Backpropagation calculates gradients of the loss with respect to
model parameters so the parameters can be updated during training.

Activation functions introduce non-linearity into neural networks.
Common activation functions include ReLU, sigmoid, and tanh.
""",
        "05_information_retrieval.txt": """
Information retrieval finds relevant information from a collection
of documents.

BM25 is a term-based ranking algorithm that scores documents using
query term frequency, document frequency, and document length.

Dense retrieval represents queries and documents as vectors and
retrieves documents using vector similarity, allowing semantic
matching.
""",
    }

    for filename, content in documents.items():
        path = DATA_DIR / filename
        path.write_text(content.strip(), encoding="utf-8")


# ---------------------------------------------------------------------
# CONFIGURE OLLAMA
# ---------------------------------------------------------------------

def setup_models():

    # Smaller context window prevents excessive KV-cache memory usage.
    Settings.llm = Ollama(
        model=LLM_MODEL,
        request_timeout=120.0,
        context_window=2048,
    )

    Settings.embed_model = OllamaEmbedding(
        model_name=EMBED_MODEL,
        request_timeout=120.0,
    )


# ---------------------------------------------------------------------
# LOAD TEXT DOCUMENTS
# ---------------------------------------------------------------------

def load_documents():

    return SimpleDirectoryReader(
        input_dir=str(DATA_DIR),
        required_exts=[".txt"],
        filename_as_id=True,
    ).load_data()


# ---------------------------------------------------------------------
# RUN QUERIES
# ---------------------------------------------------------------------

def run_queries(query_engine, label):

    results = []

    for number, question in enumerate(QUESTIONS, start=1):

        print()
        print(f"{label} {number}/10: {question}")

        start = perf_counter()

        response = query_engine.query(question)

        elapsed_ms = (perf_counter() - start) * 1000

        # Collect retrieved source filenames.
        source_names = []

        for node in response.source_nodes:

            metadata = node.node.metadata

            source = (
                metadata.get("file_name")
                or metadata.get("filename")
            )

            if source and source not in source_names:
                source_names.append(source)

        expected = EXPECTED_SOURCES[number - 1]

        source_match = expected in source_names

        result = {
            "number": number,
            "question": question,
            "answer": str(response),
            "sources": source_names,
            "expected": expected,
            "source_match": source_match,
            "latency_ms": elapsed_ms,
        }

        results.append(result)

        print(f"Sources: {source_names}")
        print(f"Expected: {expected}")
        print(f"Source verified: {source_match}")
        print(f"Latency: {elapsed_ms:.2f} ms")
        print("-" * 70)

    return results


# ---------------------------------------------------------------------
# WRITE EVIDENCE FILE
# ---------------------------------------------------------------------

def write_results(basic_results, chroma_results):

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    basic_correct = sum(
        result["source_match"]
        for result in basic_results
    )

    chroma_correct = sum(
        result["source_match"]
        for result in chroma_results
    )

    basic_avg = (
        sum(result["latency_ms"] for result in basic_results)
        / len(basic_results)
    )

    chroma_avg = (
        sum(result["latency_ms"] for result in chroma_results)
        / len(chroma_results)
    )

    with EVIDENCE_FILE.open("w", encoding="utf-8") as file:

        file.write("=" * 70 + "\n")
        file.write("W7D3: LLAMAINDEX — DOCUMENT INDEXING & RAG\n")
        file.write("=" * 70 + "\n\n")

        file.write("CONFIGURATION\n")
        file.write("-" * 70 + "\n")

        file.write(
            f"Text documents: "
            f"{len(list(DATA_DIR.glob('*.txt')))}\n"
        )

        file.write(
            f"Evaluation questions: {len(QUESTIONS)}\n"
        )

        file.write(
            f"Ollama LLM: {LLM_MODEL}\n"
        )

        file.write(
            f"Ollama embedding model: {EMBED_MODEL}\n"
        )

        file.write(
            "LlamaIndex LLM context window: 2048\n"
        )

        file.write(
            "Similarity top-k: 1\n\n"
        )

        # -------------------------------------------------------------
        # DEFAULT VECTOR STORE RESULTS
        # -------------------------------------------------------------

        file.write(
            "LLAMAINDEX DEFAULT VECTOR STORE\n"
        )

        file.write("-" * 70 + "\n")

        for result in basic_results:

            file.write(
                f"{result['number']}. "
                f"{result['question']}\n"
            )

            file.write(
                f"Answer: {result['answer']}\n"
            )

            file.write(
                f"Retrieved sources: "
                f"{result['sources']}\n"
            )

            file.write(
                f"Expected source: "
                f"{result['expected']}\n"
            )

            file.write(
                f"Source verified: "
                f"{result['source_match']}\n"
            )

            file.write(
                f"Latency: "
                f"{result['latency_ms']:.2f} ms\n\n"
            )

        file.write(
            f"Source verification: "
            f"{basic_correct}/{len(basic_results)}\n"
        )

        file.write(
            f"Average query latency: "
            f"{basic_avg:.2f} ms\n\n"
        )

        # -------------------------------------------------------------
        # CHROMADB RESULTS
        # -------------------------------------------------------------

        file.write(
            "LLAMAINDEX + CHROMADB VECTOR STORE\n"
        )

        file.write("-" * 70 + "\n")

        for result in chroma_results:

            file.write(
                f"{result['number']}. "
                f"{result['question']}\n"
            )

            file.write(
                f"Answer: {result['answer']}\n"
            )

            file.write(
                f"Retrieved sources: "
                f"{result['sources']}\n"
            )

            file.write(
                f"Expected source: "
                f"{result['expected']}\n"
            )

            file.write(
                f"Source verified: "
                f"{result['source_match']}\n"
            )

            file.write(
                f"Latency: "
                f"{result['latency_ms']:.2f} ms\n\n"
            )

        file.write(
            f"Source verification: "
            f"{chroma_correct}/{len(chroma_results)}\n"
        )

        file.write(
            f"Average query latency: "
            f"{chroma_avg:.2f} ms\n\n"
        )

        # -------------------------------------------------------------
        # LATENCY COMPARISON
        # -------------------------------------------------------------

        file.write("LATENCY COMPARISON\n")
        file.write("-" * 70 + "\n")

        file.write(
            f"Default vector store average: "
            f"{basic_avg:.2f} ms\n"
        )

        file.write(
            f"ChromaDB average: "
            f"{chroma_avg:.2f} ms\n"
        )

        file.write(
            f"Latency difference: "
            f"{chroma_avg - basic_avg:.2f} ms\n\n"
        )

        # -------------------------------------------------------------
        # MANUAL EVALUATION
        # -------------------------------------------------------------

        file.write("MANUAL EVALUATION\n")
        file.write("-" * 70 + "\n")

        file.write(
            "Each answer was checked against the expected source "
            "document. Source verification confirms whether the "
            "query engine retrieved the document containing the "
            "expected information.\n"
        )

        file.write(
            "The same 10 questions were evaluated using both the "
            "default LlamaIndex vector store and ChromaDB.\n"
        )

    print()
    print("=" * 70)
    print("W7D3 RESULTS")
    print("=" * 70)

    print(
        f"Evidence saved to: "
        f"{EVIDENCE_FILE}"
    )

    print(
        f"Default source verification: "
        f"{basic_correct}/10"
    )

    print(
        f"ChromaDB source verification: "
        f"{chroma_correct}/10"
    )

    print(
        f"Default average latency: "
        f"{basic_avg:.2f} ms"
    )

    print(
        f"ChromaDB average latency: "
        f"{chroma_avg:.2f} ms"
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print("W7D3: LLAMAINDEX DOCUMENT INDEXING & RAG")
    print("=" * 70)

    # Create the 5 source text files.
    print("\nCreating source text documents...")
    create_text_documents()

    # Configure LlamaIndex with Ollama.
    print("\nConfiguring Ollama models...")
    setup_models()

    # Load documents.
    print("\nLoading text documents...")

    documents = load_documents()

    print(
        f"Loaded {len(documents)} documents."
    )

    # -------------------------------------------------------------
    # TASK 1 + TASK 2
    # -------------------------------------------------------------

    print(
        "\nBuilding LlamaIndex VectorStoreIndex..."
    )

    index = VectorStoreIndex.from_documents(
        documents
    )

    print(
        "LlamaIndex VectorStoreIndex created."
    )

    query_engine = index.as_query_engine(
        similarity_top_k=1
    )

    print(
        "\nRunning 10 queries with default vector store..."
    )

    basic_results = run_queries(
        query_engine,
        "DEFAULT"
    )

    # -------------------------------------------------------------
    # TASK 3 — CHROMADB
    # -------------------------------------------------------------

    print(
        "\nCreating ChromaDB vector store..."
    )

    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = chroma_client.get_or_create_collection(
        "w7d3_documents"
    )

    chroma_store = ChromaVectorStore(
        chroma_collection=collection
    )

    storage_context = StorageContext.from_defaults(
        vector_store=chroma_store
    )

    print(
        "\nBuilding LlamaIndex index backed by ChromaDB..."
    )

    chroma_index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )

    chroma_query_engine = chroma_index.as_query_engine(
        similarity_top_k=1
    )

    print(
        "\nRunning same 10 queries with ChromaDB..."
    )

    chroma_results = run_queries(
        chroma_query_engine,
        "CHROMA"
    )

    # -------------------------------------------------------------
    # SAVE EVIDENCE
    # -------------------------------------------------------------

    write_results(
        basic_results,
        chroma_results
    )


if __name__ == "__main__":
    main()
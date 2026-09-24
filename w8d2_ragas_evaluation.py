"""
W8D2 - Ragas Evaluation of Multi-Document RAG

Evaluates:
- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall

Local models:
- LLM: llama3.2:3b
- Embeddings: nomic-embed-text:latest

Baseline:
- chunk_size=300
- top_k=2

Optimization candidate:
- chunk_size=300
- top_k=3
"""

import json
import math
from pathlib import Path
from typing import Any, Type

import pandas as pd
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragas import evaluate
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics._answer_relevance import answer_relevancy
from ragas.metrics._context_precision import context_precision
from ragas.metrics._context_recall import context_recall
from ragas.metrics._faithfulness import faithfulness

# Compatible with the installed Ragas version.
try:
    from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
except ImportError:
    from ragas.samples import SingleTurnSample, EvaluationDataset


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "w7d5_data"
EVIDENCE_DIR = BASE_DIR / "output_evidence" / "w8d2"

EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DOCUMENTS
# ============================================================

EXPECTED_DOCUMENTS = [
    "01_python.txt",
    "02_supervised_learning.txt",
    "03_overfitting.txt",
    "04_rag.txt",
    "05_vector_database.txt",
]


# ============================================================
# EVALUATION QUESTIONS
# ============================================================

QUESTIONS = [
    {
        "question": "What is Python?",
        "reference": (
            "Python is a high-level, general-purpose programming language "
            "known for readable syntax and broad use in software development "
            "and data science."
        ),
        "reference_context": "01_python.txt",
    },
    {
        "question": "What are some important features of Python?",
        "reference": (
            "Important Python features include readable syntax, dynamic typing, "
            "a large standard library, and support for multiple programming paradigms."
        ),
        "reference_context": "01_python.txt",
    },
    {
        "question": "What is supervised learning?",
        "reference": (
            "Supervised learning is a machine learning approach in which a model "
            "learns from labeled training data to predict outputs for new inputs."
        ),
        "reference_context": "02_supervised_learning.txt",
    },
    {
        "question": "What is the difference between classification and regression?",
        "reference": (
            "Classification predicts discrete categories or classes, while "
            "regression predicts continuous numerical values."
        ),
        "reference_context": "02_supervised_learning.txt",
    },
    {
        "question": "What is overfitting?",
        "reference": (
            "Overfitting occurs when a machine learning model learns the training "
            "data too closely and performs poorly on unseen data."
        ),
        "reference_context": "03_overfitting.txt",
    },
    {
        "question": "How can overfitting be reduced?",
        "reference": (
            "Overfitting can be reduced using techniques such as regularization, "
            "cross-validation, simpler models, more training data, and early stopping."
        ),
        "reference_context": "03_overfitting.txt",
    },
    {
        "question": "What is retrieval augmented generation?",
        "reference": (
            "Retrieval augmented generation, or RAG, retrieves relevant information "
            "from an external knowledge source and provides it to a language model "
            "as context for generating an answer."
        ),
        "reference_context": "04_rag.txt",
    },
    {
        "question": "What are the main steps in a RAG pipeline?",
        "reference": (
            "A typical RAG pipeline loads documents, splits them into chunks, "
            "creates embeddings, stores them in a vector database, retrieves "
            "relevant chunks, and supplies them to a language model."
        ),
        "reference_context": "04_rag.txt",
    },
    {
        "question": "What is a vector database?",
        "reference": (
            "A vector database stores vector embeddings and supports similarity "
            "search to retrieve information that is semantically related to a query."
        ),
        "reference_context": "05_vector_database.txt",
    },
    {
        "question": "Why are vector databases useful for RAG?",
        "reference": (
            "Vector databases are useful for RAG because they efficiently retrieve "
            "semantically relevant document chunks that can be supplied to a language model."
        ),
        "reference_context": "05_vector_database.txt",
    },
]


# ============================================================
# OLLAMA EMBEDDINGS -> RAGAS
# ============================================================

class OllamaRagasEmbeddings(BaseRagasEmbeddings):
    def __init__(
        self,
        model: str = "nomic-embed-text:latest",
    ):
        self.embeddings = OllamaEmbeddings(
            model=model
        )

    def embed_documents(self, texts):
        return self.embeddings.embed_documents(
            texts
        )

    def embed_query(self, text):
        return self.embeddings.embed_query(
            text
        )

    async def aembed_documents(self, texts):
        return await self.embeddings.aembed_documents(
            texts
        )

    async def aembed_query(self, text):
        return await self.embeddings.aembed_query(
            text
        )


# ============================================================
# OLLAMA LLM -> RAGAS
# ============================================================

class OllamaRagasLLM(InstructorBaseRagasLLM):
    """
    Adapter for the installed legacy Ragas metrics.

    Uses ChatOllama structured JSON-schema output so
    Ragas receives the required Pydantic objects.
    """

    def __init__(
        self,
        model: str = "llama3.2:3b",
    ):
        self.llm = ChatOllama(
            model=model,
            temperature=0,
        )

        # Required by installed Ragas.
        self.is_async = False

    def generate(
        self,
        prompt: str,
        response_model: Type[Any],
    ) -> Any:

        structured_llm = (
            self.llm.with_structured_output(
                response_model,
                method="json_schema",
            )
        )

        result = structured_llm.invoke(
            prompt
        )

        if isinstance(
            result,
            response_model,
        ):
            return result

        if isinstance(
            result,
            dict,
        ):
            return response_model.model_validate(
                result
            )

        if hasattr(
            result,
            "model_dump",
        ):
            return response_model.model_validate(
                result.model_dump()
            )

        return response_model.model_validate(
            result
        )

    async def agenerate(
        self,
        prompt: str,
        response_model: Type[Any],
    ) -> Any:

        structured_llm = (
            self.llm.with_structured_output(
                response_model,
                method="json_schema",
            )
        )

        result = await structured_llm.ainvoke(
            prompt
        )

        if isinstance(
            result,
            response_model,
        ):
            return result

        if isinstance(
            result,
            dict,
        ):
            return response_model.model_validate(
                result
            )

        if hasattr(
            result,
            "model_dump",
        ):
            return response_model.model_validate(
                result.model_dump()
            )

        return response_model.model_validate(
            result
        )


# ============================================================
# LOAD DOCUMENTS
# ============================================================

def load_documents():

    documents = {}

    print("\nLoading W7D5 documents...")

    for filename in EXPECTED_DOCUMENTS:

        path = DATA_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Required document missing: {path}"
            )

        documents[filename] = path.read_text(
            encoding="utf-8"
        )

        print(
            f"  OK - {filename}"
        )

    print(
        f"Loaded {len(documents)} documents."
    )

    return documents


# ============================================================
# BUILD VECTOR STORE
# ============================================================

def build_vectorstore(
    documents,
    chunk_size,
    collection_name,
):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=50,
    )

    texts = []
    metadatas = []

    for filename, content in documents.items():

        chunks = splitter.split_text(
            content
        )

        for chunk in chunks:

            texts.append(chunk)

            metadatas.append(
                {
                    "source": filename
                }
            )

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text:latest"
    )

    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        collection_name=collection_name,
    )

    return vectorstore


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question,
    contexts,
    llm,
):

    context_text = "\n\n".join(
        contexts
    )

    prompt = f"""
You are a helpful AI/ML learning assistant.

Answer the question using ONLY the retrieved context.

Do not invent information.

Retrieved context:
{context_text}

Question:
{question}

Give a concise answer.
"""

    response = llm.invoke(
        prompt
    )

    if hasattr(
        response,
        "content",
    ):
        return str(
            response.content
        ).strip()

    return str(
        response
    ).strip()


# ============================================================
# CREATE SAMPLES
# ============================================================

def create_samples(
    documents,
    chunk_size,
    top_k,
    collection_name,
):

    vectorstore = build_vectorstore(
        documents=documents,
        chunk_size=chunk_size,
        collection_name=collection_name,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": top_k
        },
    )

    answer_llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0,
    )

    samples = []

    print(
        f"\nCreating samples: "
        f"chunk_size={chunk_size}, "
        f"top_k={top_k}"
    )

    for index, item in enumerate(
        QUESTIONS,
        start=1,
    ):

        question = item[
            "question"
        ]

        docs = retriever.invoke(
            question
        )

        contexts = [
            doc.page_content
            for doc in docs
        ]

        sources = [
            doc.metadata.get(
                "source",
                "unknown",
            )
            for doc in docs
        ]

        answer = generate_answer(
            question=question,
            contexts=contexts,
            llm=answer_llm,
        )

        reference_context = documents[
            item["reference_context"]
        ]

        sample = SingleTurnSample(
            user_input=question,
            retrieved_contexts=contexts,
            reference_contexts=[
                reference_context
            ],
            response=answer,
            reference=item["reference"],
        )

        samples.append(sample)

        print(
            f"[{index}/10] "
            f"{question}"
        )

        print(
            f"  Sources: {sources}"
        )

        print(
            f"  Answer: {answer}"
        )

    return samples


# ============================================================
# RAGAS EVALUATION
# ============================================================

def run_evaluation(
    samples,
    ragas_llm,
    ragas_embeddings,
):

    # IMPORTANT:
    # Installed Ragas evaluate() requires
    # EvaluationDataset, not a Python list.

    dataset = EvaluationDataset(
        samples=samples
    )

    # Configure legacy metrics.
    faithfulness.llm = ragas_llm

    answer_relevancy.llm = ragas_llm
    answer_relevancy.embeddings = (
        ragas_embeddings
    )

    context_precision.llm = ragas_llm

    context_recall.llm = ragas_llm

    print(
        "\nRunning Ragas evaluation..."
    )

    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        raise_exceptions=False,
        show_progress=True,
    )

    return result.to_pandas()


# ============================================================
# CALCULATE AVERAGES
# ============================================================

def calculate_averages(
    dataframe,
):

    metrics = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]

    averages = {}

    for metric in metrics:

        if metric not in dataframe.columns:

            averages[metric] = float(
                "nan"
            )

            continue

        values = pd.to_numeric(
            dataframe[metric],
            errors="coerce",
        )

        averages[metric] = float(
            values.mean()
        )

    return averages


# ============================================================
# VALIDATE RESULTS
# ============================================================

def validate_results(
    averages,
):

    valid_count = 0

    print(
        "\nMetric validation:"
    )

    for metric, value in averages.items():

        if math.isnan(value):

            print(
                f"  {metric}: INVALID / NaN"
            )

        else:

            print(
                f"  {metric}: {value:.4f}"
            )

            valid_count += 1

    if valid_count == 0:

        raise RuntimeError(
            "Ragas returned no valid metric scores."
        )


# ============================================================
# SAVE QA PAIRS
# ============================================================

def save_qa_pairs(
    samples,
):

    output = []

    for sample in samples:

        data = sample.to_dict()

        output.append(
            {
                "user_input": data.get(
                    "user_input"
                ),
                "response": data.get(
                    "response"
                ),
                "retrieved_contexts": data.get(
                    "retrieved_contexts"
                ),
                "reference": data.get(
                    "reference"
                ),
            }
        )

    path = (
        EVIDENCE_DIR
        / "qa_pairs.json"
    )

    path.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    baseline_averages,
    optimized_averages,
    baseline_mean,
    optimized_mean,
    status,
    selected_top_k,
):

    results = {
        "project": (
            "W8D2 Ragas Evaluation"
        ),
        "llm": "llama3.2:3b",
        "embedding_model": (
            "nomic-embed-text:latest"
        ),
        "metrics": [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ],
        "baseline": {
            "chunk_size": 300,
            "top_k": 2,
            "metrics": baseline_averages,
            "mean_score": baseline_mean,
        },
        "optimized": {
            "chunk_size": 300,
            "top_k": 3,
            "metrics": optimized_averages,
            "mean_score": optimized_mean,
        },
        "comparison": {
            "status": status,
            "baseline_mean": baseline_mean,
            "optimized_mean": optimized_mean,
            "selected_top_k": selected_top_k,
        },
    }

    json_path = (
        EVIDENCE_DIR
        / "ragas_evaluation_results.json"
    )

    json_path.write_text(
        json.dumps(
            results,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    text_path = (
        EVIDENCE_DIR
        / "ragas_evaluation_results.txt"
    )

    with text_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "W8D2 - RAGAS EVALUATION RESULTS\n"
        )

        file.write(
            "=" * 80 + "\n\n"
        )

        file.write(
            "LLM: llama3.2:3b\n"
        )

        file.write(
            "Embedding: "
            "nomic-embed-text:latest\n"
        )

        file.write(
            "Metrics: Faithfulness, "
            "Answer Relevancy, "
            "Context Precision, "
            "Context Recall\n\n"
        )

        file.write(
            "BASELINE\n"
        )

        file.write(
            "-" * 80 + "\n"
        )

        file.write(
            "Chunk size: 300\n"
        )

        file.write(
            "Top K: 2\n"
        )

        for metric, value in (
            baseline_averages.items()
        ):

            file.write(
                f"{metric}: {value:.4f}\n"
            )

        file.write(
            f"Mean score: "
            f"{baseline_mean:.4f}\n\n"
        )

        file.write(
            "OPTIMIZED\n"
        )

        file.write(
            "-" * 80 + "\n"
        )

        file.write(
            "Chunk size: 300\n"
        )

        file.write(
            "Top K: 3\n"
        )

        for metric, value in (
            optimized_averages.items()
        ):

            file.write(
                f"{metric}: {value:.4f}\n"
            )

        file.write(
            f"Mean score: "
            f"{optimized_mean:.4f}\n\n"
        )

        file.write(
            "COMPARISON\n"
        )

        file.write(
            "-" * 80 + "\n"
        )

        file.write(
            f"Status: {status}\n"
        )

        file.write(
            f"Baseline mean: "
            f"{baseline_mean:.4f}\n"
        )

        file.write(
            f"Optimized mean: "
            f"{optimized_mean:.4f}\n"
        )

        file.write(
            f"Selected top_k: "
            f"{selected_top_k}\n"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("W8D2 - RAGAS EVALUATION")
    print("=" * 80)

    # --------------------------------------------------------
    # Documents
    # --------------------------------------------------------

    documents = load_documents()

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    print(
        "\nInitializing evaluation models..."
    )

    ragas_llm = OllamaRagasLLM(
        model="llama3.2:3b"
    )

    ragas_embeddings = (
        OllamaRagasEmbeddings(
            model="nomic-embed-text:latest"
        )
    )

    print(
        "  OK - LLM: llama3.2:3b"
    )

    print(
        "  OK - Embeddings: "
        "nomic-embed-text:latest"
    )

    print(
        "  OK - Structured output: json_schema"
    )

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("BASELINE: top_k=2")
    print("=" * 80)

    baseline_samples = create_samples(
        documents=documents,
        chunk_size=300,
        top_k=2,
        collection_name="w8d2_baseline",
    )

    save_qa_pairs(
        baseline_samples
    )

    baseline_df = run_evaluation(
        samples=baseline_samples,
        ragas_llm=ragas_llm,
        ragas_embeddings=ragas_embeddings,
    )

    baseline_averages = (
        calculate_averages(
            baseline_df
        )
    )

    print(
        "\nBASELINE RESULTS"
    )

    for metric, value in (
        baseline_averages.items()
    ):

        if math.isnan(value):

            print(
                f"{metric}: NaN"
            )

        else:

            print(
                f"{metric}: "
                f"{value:.4f}"
            )

    validate_results(
        baseline_averages
    )

    # --------------------------------------------------------
    # OPTIMIZED
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("OPTIMIZATION CANDIDATE: top_k=3")
    print("=" * 80)

    optimized_samples = create_samples(
        documents=documents,
        chunk_size=300,
        top_k=3,
        collection_name="w8d2_optimized",
    )

    optimized_df = run_evaluation(
        samples=optimized_samples,
        ragas_llm=ragas_llm,
        ragas_embeddings=ragas_embeddings,
    )

    optimized_averages = (
        calculate_averages(
            optimized_df
        )
    )

    print(
        "\nOPTIMIZED RESULTS"
    )

    for metric, value in (
        optimized_averages.items()
    ):

        if math.isnan(value):

            print(
                f"{metric}: NaN"
            )

        else:

            print(
                f"{metric}: "
                f"{value:.4f}"
            )

    validate_results(
        optimized_averages
    )

    # --------------------------------------------------------
    # COMPARISON
    # --------------------------------------------------------

    metric_names = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]

    baseline_values = [
        baseline_averages[m]
        for m in metric_names
        if not math.isnan(
            baseline_averages[m]
        )
    ]

    optimized_values = [
        optimized_averages[m]
        for m in metric_names
        if not math.isnan(
            optimized_averages[m]
        )
    ]

    baseline_mean = (
        sum(baseline_values)
        / len(baseline_values)
    )

    optimized_mean = (
        sum(optimized_values)
        / len(optimized_values)
    )

    if optimized_mean > baseline_mean:

        status = "IMPROVED"
        selected_top_k = 3

    else:

        status = "NOT_IMPROVED"
        selected_top_k = 2

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_results(
        baseline_averages=baseline_averages,
        optimized_averages=optimized_averages,
        baseline_mean=baseline_mean,
        optimized_mean=optimized_mean,
        status=status,
        selected_top_k=selected_top_k,
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("W8D2 COMPLETE")
    print("=" * 80)

    print(
        f"\nBaseline mean:  "
        f"{baseline_mean:.4f}"
    )

    print(
        f"Optimized mean: "
        f"{optimized_mean:.4f}"
    )

    print(
        f"Status:          {status}"
    )

    print(
        f"Selected top_k:  "
        f"{selected_top_k}"
    )

    print("\nEvidence created:")

    print(
        f"  {EVIDENCE_DIR / 'qa_pairs.json'}"
    )

    print(
        f"  {EVIDENCE_DIR / 'ragas_evaluation_results.json'}"
    )

    print(
        f"  {EVIDENCE_DIR / 'ragas_evaluation_results.txt'}"
    )

    print(
        "\nSUCCESS: Valid Ragas scores were produced."
    )


if __name__ == "__main__":
    main()
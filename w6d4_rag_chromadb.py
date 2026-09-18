import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from langchain_ollama import OllamaLLM


# ============================================================
# W6D4: RAG Pipeline - LangChain + ChromaDB
# ============================================================

MODEL = "llama3.2:3b"
CHROMA_PATH = "./chroma_db"

DOCUMENT_COLLECTION = "w6d4_ai_ml_documents"
PDF_COLLECTION = "w6d4_pdf_rag"

PDF_PATH = "./data/w5d4/ai_ml_reference.pdf"


DOCUMENTS = [
    "Supervised learning uses labeled data to train a machine learning model.",
    "Unsupervised learning finds patterns in data without labeled target values.",
    "Classification predicts a discrete class or category.",
    "Regression predicts a continuous numerical value.",
    "Overfitting happens when a model learns training data too closely and performs poorly on unseen data.",
    "Underfitting happens when a model is too simple to capture important patterns in the data.",
    "Cross-validation evaluates a model across multiple training and validation splits.",
    "Feature scaling puts numerical features on comparable scales.",
    "Standardization commonly transforms features to have mean zero and unit variance.",
    "A decision tree makes predictions using a sequence of feature-based decisions.",
    "Random forests combine predictions from multiple decision trees.",
    "Support Vector Machines find decision boundaries that separate classes.",
    "K-nearest neighbors predicts using the labels of nearby training examples.",
    "Logistic regression is commonly used for binary classification.",
    "Neural networks contain interconnected layers that learn representations from data.",
    "Embeddings represent text as numerical vectors that capture semantic information.",
    "Vector databases store embeddings and support similarity-based retrieval.",
    "Cosine similarity measures the angle-based similarity between two vectors.",
    "Retrieval-augmented generation retrieves relevant information before generating an answer.",
    "ChromaDB is a vector database that can store documents, embeddings, and metadata.",
]


# ============================================================
# COMMON CHROMADB SETUP
# ============================================================

def get_client():
    return chromadb.PersistentClient(path=CHROMA_PATH)


def get_embedding_function():
    return embedding_functions.DefaultEmbeddingFunction()


# ============================================================
# TASK 1
# Create collection and add 20 documents
# ============================================================

def task1():
    print("\n")
    print("=" * 60)
    print("W6D4 TASK 1 - CHROMADB VECTOR STORE")
    print("=" * 60)

    client = get_client()
    embedding_function = get_embedding_function()

    collection = client.get_or_create_collection(
        name=DOCUMENT_COLLECTION,
        embedding_function=embedding_function,
        configuration={"hnsw": {"space": "cosine"}},
    )

    ids = [
        f"doc_{i + 1}"
        for i in range(len(DOCUMENTS))
    ]

    metadatas = [
        {
            "topic": (
                "supervised_learning"
                if i < 5
                else "machine_learning"
            ),
            "document_number": i + 1,
        }
        for i in range(len(DOCUMENTS))
    ]

    collection.upsert(
        ids=ids,
        documents=DOCUMENTS,
        metadatas=metadatas,
    )

    count = collection.count()

    print(f"Collection: {DOCUMENT_COLLECTION}")
    print(f"Documents added: {count}")
    print("Embedding function: ChromaDB default embedding")
    print("Distance metric: cosine")

    assert count == 20, (
        f"Expected 20 documents, found {count}"
    )

    print("Task 1 status: SUCCESS")

    return collection


# ============================================================
# TASK 2
# Similarity search + metadata filtering
# ============================================================

def task2(collection):
    print("\n")
    print("=" * 60)
    print("W6D4 TASK 2 - SIMILARITY SEARCH")
    print("=" * 60)

    # --------------------------------------------------------
    # Cosine similarity search
    # --------------------------------------------------------

    print("\n1. Similarity Search")
    print("-" * 60)

    results = collection.query(
        query_texts=[
            "What is overfitting in machine learning?"
        ],
        n_results=3,
    )

    for i in range(3):
        print(f"\nResult {i + 1}")
        print(f"ID: {results['ids'][0][i]}")
        print(
            f"Distance: "
            f"{results['distances'][0][i]:.4f}"
        )
        print(
            f"Document: "
            f"{results['documents'][0][i]}"
        )
        print(
            f"Metadata: "
            f"{results['metadatas'][0][i]}"
        )

    assert len(results["ids"][0]) == 3

    # --------------------------------------------------------
    # Metadata filtering
    # --------------------------------------------------------

    print("\n2. Metadata Filtering")
    print("-" * 60)

    filtered_results = collection.query(
        query_texts=["machine learning"],
        n_results=5,
        where={
            "topic": "supervised_learning"
        },
    )

    print(
        "Filter: "
        "topic = supervised_learning"
    )

    for i in range(
        len(filtered_results["ids"][0])
    ):
        print(
            f"\nFiltered Result {i + 1}"
        )

        print(
            f"ID: "
            f"{filtered_results['ids'][0][i]}"
        )

        print(
            f"Document: "
            f"{filtered_results['documents'][0][i]}"
        )

        print(
            f"Metadata: "
            f"{filtered_results['metadatas'][0][i]}"
        )

    assert len(
        filtered_results["ids"][0]
    ) == 5

    assert all(
        item["topic"] == "supervised_learning"
        for item in filtered_results["metadatas"][0]
    )

    print("\nTask 2 status: SUCCESS")


# ============================================================
# TASK 3
# PDF + ChromaDB + Ollama RAG
# ============================================================

def load_pdf():
    reader = PdfReader(PDF_PATH)

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):
        text = page.extract_text()

        if text and text.strip():
            pages.append(
                {
                    "page": page_number,
                    "text": text.strip(),
                }
            )

    return pages


def split_into_chunks(pages):
    chunks = []

    chunk_size = 500

    for page in pages:
        text = page["text"]

        for start in range(
            0,
            len(text),
            chunk_size
        ):
            chunk = text[
                start:start + chunk_size
            ].strip()

            if chunk:
                chunks.append(
                    {
                        "text": chunk,
                        "page": page["page"],
                    }
                )

    return chunks


def create_pdf_collection(chunks):
    client = get_client()
    embedding_function = get_embedding_function()

    collection = client.get_or_create_collection(
        name=PDF_COLLECTION,
        embedding_function=embedding_function,
        configuration={
            "hnsw": {
                "space": "cosine"
            }
        },
    )

    ids = [
        f"pdf_chunk_{i + 1}"
        for i in range(len(chunks))
    ]

    documents = [
        chunk["text"]
        for chunk in chunks
    ]

    metadatas = [
        {
            "source": "ai_ml_reference.pdf",
            "page": chunk["page"],
            "chunk": i + 1,
        }
        for i, chunk in enumerate(chunks)
    ]

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )

    return collection


def retrieve_top_3(collection, question):
    return collection.query(
        query_texts=[question],
        n_results=3,
    )


def generate_answer(
    question,
    retrieved_documents
):
    context = "\n\n".join(
        retrieved_documents
    )

    prompt = f"""
You are answering a question using only
the provided PDF context.

PDF Context:
{context}

Question:
{question}

Give a short and factual answer based
only on the provided context.
"""

    llm = OllamaLLM(model=MODEL)

    answer = llm.invoke(prompt)

    return answer


def task3():
    print("\n")
    print("=" * 60)
    print("W6D4 TASK 3 - PDF RAG WITH OLLAMA")
    print("=" * 60)

    # --------------------------------------------------------
    # Load PDF
    # --------------------------------------------------------

    pages = load_pdf()

    print(f"PDF: {PDF_PATH}")
    print(
        f"Pages with text: {len(pages)}"
    )

    assert len(pages) > 0, (
        "No text could be extracted from PDF"
    )

    # --------------------------------------------------------
    # Create chunks
    # --------------------------------------------------------

    chunks = split_into_chunks(pages)

    print(
        f"Chunks created: {len(chunks)}"
    )

    assert len(chunks) > 0, (
        "No chunks were created"
    )

    # --------------------------------------------------------
    # Store PDF chunks in ChromaDB
    # --------------------------------------------------------

    collection = create_pdf_collection(
        chunks
    )

    print(
        f"ChromaDB collection: "
        f"{PDF_COLLECTION}"
    )

    print(
        f"Stored chunks: "
        f"{collection.count()}"
    )

    assert collection.count() > 0

    # --------------------------------------------------------
    # Retrieve top 3 chunks
    # --------------------------------------------------------

    question = (
        "What is machine learning "
        "and how is it used?"
    )

    results = retrieve_top_3(
        collection,
        question
    )

    print("\nTop-3 Retrieved Chunks")
    print("-" * 60)

    for i in range(
        len(results["ids"][0])
    ):
        print(
            f"\nChunk {i + 1}"
        )

        print(
            f"ID: "
            f"{results['ids'][0][i]}"
        )

        print(
            f"Distance: "
            f"{results['distances'][0][i]:.4f}"
        )

        print(
            f"Metadata: "
            f"{results['metadatas'][0][i]}"
        )

        print(
            f"Text: "
            f"{results['documents'][0][i]}"
        )

    assert len(
        results["ids"][0]
    ) == 3

    # --------------------------------------------------------
    # Send retrieved context to Ollama
    # --------------------------------------------------------

    answer = generate_answer(
        question,
        results["documents"][0],
    )

    print("\nOllama Answer")
    print("-" * 60)
    print(answer)

    assert answer and answer.strip()

    print("\nTask 3 status: SUCCESS")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("#" * 60)
    print("# W6D4: RAG PIPELINE - LANGCHAIN + CHROMADB")
    print("#" * 60)

    collection = task1()

    task2(collection)

    task3()

    print("\n")
    print("#" * 60)
    print("# W6D4 ALL TASKS COMPLETED SUCCESSFULLY")
    print("#" * 60)

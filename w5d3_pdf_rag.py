
import requests
import chromadb
from pypdf import PdfReader


# ---------------------------------------------------------
# W5D3: PDF + ChromaDB + Ollama RAG
# ---------------------------------------------------------

PDF_PATH = "./data/w5d3/ai_ml_reference.pdf"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "w5d3_pdf_chunks"
OLLAMA_MODEL = "llama3.2:3b"


# ---------------------------------------------------------
# 1. Extract text from PDF
# ---------------------------------------------------------

reader = PdfReader(PDF_PATH)

pdf_text = ""

for page in reader.pages:
    text = page.extract_text()
    if text:
        pdf_text += text + "\n"

print("=" * 60)
print("W5D3: PDF + ChromaDB + Ollama RAG")
print("=" * 60)

print(f"\nPDF: {PDF_PATH}")
print(f"Pages extracted: {len(reader.pages)}")


# ---------------------------------------------------------
# 2. Split PDF text into chunks
# ---------------------------------------------------------

chunk_size = 500

chunks = [
    pdf_text[i:i + chunk_size]
    for i in range(0, len(pdf_text), chunk_size)
]

chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

print(f"Total chunks created: {len(chunks)}")


# ---------------------------------------------------------
# 3. Create ChromaDB collection
# ---------------------------------------------------------

client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    configuration={
        "hnsw": {
            "space": "cosine"
        }
    }
)

ids = [f"pdf_chunk_{i:03d}" for i in range(len(chunks))]

metadatas = [
    {
        "source": "ai_ml_reference.pdf",
        "chunk": i + 1
    }
    for i in range(len(chunks))
]


# ---------------------------------------------------------
# 4. Add PDF chunks with embeddings
# ---------------------------------------------------------

collection.upsert(
    ids=ids,
    documents=chunks,
    metadatas=metadatas
)

print(f"Chunks stored in ChromaDB: {collection.count()}")


# ---------------------------------------------------------
# 5. Similarity search - retrieve top 3 chunks
# ---------------------------------------------------------

question = "How can overfitting be prevented in machine learning?"

results = collection.query(
    query_texts=[question],
    n_results=3
)

retrieved_ids = results["ids"][0]
retrieved_documents = results["documents"][0]
retrieved_distances = results["distances"][0]


print("\n" + "=" * 60)
print("TOP-3 RETRIEVED PDF CHUNKS")
print("=" * 60)

for i, (chunk_id, document, distance) in enumerate(
    zip(
        retrieved_ids,
        retrieved_documents,
        retrieved_distances
    ),
    start=1
):
    print(f"\nChunk {i}")
    print(f"ID: {chunk_id}")
    print(f"Cosine distance: {distance:.4f}")
    print(f"Content:\n{document}")


# ---------------------------------------------------------
# 6. Send retrieved context to Ollama
# ---------------------------------------------------------

context = "\n\n".join(
    f"Retrieved Chunk {i + 1}:\n{document}"
    for i, document in enumerate(retrieved_documents)
)

prompt = f"""
Answer the question using ONLY information explicitly stated
in the retrieved PDF chunks.

Do not add information from your general knowledge.
Do not infer additional techniques that are not explicitly stated.
If a technique is not mentioned in the retrieved chunks,
do not include it.

Question:
{question}

Retrieved PDF chunks:
{context}

Give a concise answer using only the retrieved information.
"""


response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    },
    timeout=120
)

response.raise_for_status()

answer = response.json()["response"]


# ---------------------------------------------------------
# 7. Display Ollama answer
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("OLLAMA ANSWER")
print("=" * 60)

print(answer)


# ---------------------------------------------------------
# 8. Manual Verification
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("MANUAL VERIFICATION")
print("=" * 60)

print("\nVerification points:")
print("1. PDF text was successfully extracted.")
print("2. PDF text was split into chunks.")
print("3. Chunks were embedded and stored in ChromaDB.")
print("4. Top-3 chunks were retrieved using cosine similarity.")
print("5. Retrieved chunks were passed to Ollama.")
print("6. Ollama generated an answer using only retrieved context.")

print("\nW5D3 PDF RAG completed successfully.")


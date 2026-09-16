import requests
import chromadb
from pypdf import PdfReader

PDF_PATH = "./data/w5d4/ai_ml_reference.pdf"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "w5d4_pdf_chunks"
OLLAMA_MODEL = "llama3.2:3b"

reader = PdfReader(PDF_PATH)

pdf_text = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        pdf_text += text + "\n"

print("=" * 60)
print("W5D4: PDF + ChromaDB + Ollama RAG")
print("=" * 60)

print(f"\nPDF: {PDF_PATH}")
print(f"Pages extracted: {len(reader.pages)}")

chunk_size = 500
chunks = [
    pdf_text[i:i + chunk_size].strip()
    for i in range(0, len(pdf_text), chunk_size)
    if pdf_text[i:i + chunk_size].strip()
]

print(f"Total chunks created: {len(chunks)}")

client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    configuration={"hnsw": {"space": "cosine"}},
)

ids = [f"pdf_chunk_{i:03d}" for i in range(len(chunks))]

metadatas = [
    {
        "source": "ai_ml_reference.pdf",
        "chunk": i + 1,
    }
    for i in range(len(chunks))
]

collection.upsert(
    ids=ids,
    documents=chunks,
    metadatas=metadatas,
)

print(f"Chunks stored in ChromaDB: {collection.count()}")

question = "How can overfitting be prevented in machine learning?"

results = collection.query(
    query_texts=[question],
    n_results=3,
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
        retrieved_distances,
    ),
    start=1,
):
    print(f"\nChunk {i}")
    print(f"ID: {chunk_id}")
    print(f"Cosine distance: {distance:.4f}")
    print(f"Content:\n{document}")

context = "\n\n".join(
    f"Retrieved Chunk {i + 1}:\n{document}"
    for i, document in enumerate(retrieved_documents)
)

prompt = f"""
Answer the question using ONLY information explicitly stated
in the retrieved PDF chunks.

Do not add information from general knowledge.

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
        "stream": False,
    },
    timeout=120,
)

response.raise_for_status()

answer = response.json()["response"]

print("\n" + "=" * 60)
print("OLLAMA ANSWER")
print("=" * 60)
print(answer)

print("\n" + "=" * 60)
print("MANUAL VERIFICATION")
print("=" * 60)

print("1. PDF text was successfully extracted.")
print("2. PDF text was split into chunks.")
print("3. Chunks were stored in ChromaDB.")
print("4. Top-3 chunks were retrieved using cosine similarity.")
print("5. Retrieved chunks were passed to Ollama.")
print("6. Ollama generated an answer using retrieved context.")

print("\nW5D4 PDF RAG completed successfully.")
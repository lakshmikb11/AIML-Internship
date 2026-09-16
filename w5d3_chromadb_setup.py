import chromadb


# ---------------------------------------------------------
# W5D3: ChromaDB Vector Store Setup
# ---------------------------------------------------------

# Create a persistent ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")

# Create or get collection with cosine similarity
collection = client.get_or_create_collection(
    name="w5d3_ai_ml_documents",
    configuration={
        "hnsw": {
            "space": "cosine"
        }
    }
)

# ---------------------------------------------------------
# 20 AI/ML documents
# ---------------------------------------------------------

documents = [
    "Supervised learning uses labeled data to train a machine learning model.",
    "Unsupervised learning finds patterns in data without labeled target values.",
    "Classification predicts discrete categories such as spam or not spam.",
    "Regression predicts continuous numerical values such as house prices.",
    "Overfitting happens when a model learns training data too closely and performs poorly on unseen data.",
    "Underfitting occurs when a model is too simple to capture important patterns in the data.",
    "Cross-validation helps estimate how well a machine learning model generalizes to unseen data.",
    "Regularization reduces model complexity and can help prevent overfitting.",
    "L1 regularization can produce sparse models by driving some feature coefficients toward zero.",
    "L2 regularization penalizes large model coefficients to reduce model complexity.",
    "A decision tree makes predictions by splitting data using feature-based conditions.",
    "Random forests combine multiple decision trees to improve generalization.",
    "Support Vector Machines find a decision boundary that separates classes.",
    "K-Nearest Neighbors predicts a sample using nearby training examples.",
    "Feature scaling is important for algorithms that depend on distances or feature magnitudes.",
    "Precision measures how many predicted positive examples are actually positive.",
    "Recall measures how many actual positive examples are correctly identified.",
    "A confusion matrix summarizes correct and incorrect classification predictions.",
    "Principal Component Analysis reduces the dimensionality of data while preserving important variation.",
    "Neural networks use interconnected layers of neurons to learn complex patterns."
]

metadata = [
    {"topic": "supervised_learning"},
    {"topic": "unsupervised_learning"},
    {"topic": "classification"},
    {"topic": "regression"},
    {"topic": "model_evaluation"},
    {"topic": "model_evaluation"},
    {"topic": "model_evaluation"},
    {"topic": "regularization"},
    {"topic": "regularization"},
    {"topic": "regularization"},
    {"topic": "classification"},
    {"topic": "classification"},
    {"topic": "classification"},
    {"topic": "classification"},
    {"topic": "preprocessing"},
    {"topic": "model_evaluation"},
    {"topic": "model_evaluation"},
    {"topic": "model_evaluation"},
    {"topic": "dimensionality_reduction"},
    {"topic": "deep_learning"}
]

ids = [f"doc_{i:02d}" for i in range(1, 21)]

# Add documents to ChromaDB
collection.upsert(
    ids=ids,
    documents=documents,
    metadatas=metadata
)

print("=" * 60)
print("W5D3: ChromaDB Vector Store Setup")
print("=" * 60)

print(f"Collection name: {collection.name}")
print(f"Total documents: {collection.count()}")

# ---------------------------------------------------------
# Similarity Search
# ---------------------------------------------------------

query = "How can I prevent a machine learning model from overfitting?"

results = collection.query(
    query_texts=[query],
    n_results=5
)

print("\n" + "=" * 60)
print("SIMILARITY SEARCH - COSINE")
print("=" * 60)

print(f"\nQuery: {query}")

for i, (doc_id, document, distance) in enumerate(
    zip(
        results["ids"][0],
        results["documents"][0],
        results["distances"][0]
    ),
    start=1
):
    print(f"\nResult {i}")
    print(f"ID: {doc_id}")
    print(f"Distance: {distance:.4f}")
    print(f"Document: {document}")

# ---------------------------------------------------------
# Metadata Filtering
# ---------------------------------------------------------

filtered_results = collection.query(
    query_texts=["machine learning classification"],
    where={"topic": "classification"},
    n_results=5
)

print("\n" + "=" * 60)
print("METADATA FILTERING")
print("=" * 60)

print("\nFilter: topic = classification")

for i, (doc_id, document) in enumerate(
    zip(
        filtered_results["ids"][0],
        filtered_results["documents"][0]
    ),
    start=1
):
    print(f"\nResult {i}")
    print(f"ID: {doc_id}")
    print(f"Document: {document}")

# ---------------------------------------------------------
# Manual Verification
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("MANUAL VERIFICATION")
print("=" * 60)

print("\nSimilarity search should return documents related to:")
print("- Overfitting")
print("- Regularization")
print("- Cross-validation")
print("- Model evaluation")

print("\nMetadata filtering should return only documents")
print("with topic = classification.")

print("\nW5D3 ChromaDB setup completed successfully.")
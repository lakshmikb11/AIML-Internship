# W8D3: Haystack API on Railway — BM25 vs Dense Retrieval

## Objective

Build and evaluate a Haystack retrieval pipeline using five PDF documents. Compare BM25 sparse retrieval with dense embedding-based retrieval using the same ten questions.

## Pipeline

PDF Documents -> PyPDFToDocument -> InMemoryDocumentStore -> Retriever

Two retrievers were evaluated:

1. InMemoryBM25Retriever
2. InMemoryEmbeddingRetriever

Embedding Model: sentence-transformers/all-MiniLM-L6-v2

## Dataset

- Documents: 5 PDF files
- Evaluation questions: 10
- Retrieval depth: Top-K = 2
- Evaluation metric: Precision@1

## Results

| Retriever | Questions | Correct Top-1 | Precision@1 |
|---|---:|---:|---:|
| BM25 | 10 | 10 | 100% |
| Dense | 10 | 10 | 100% |

## Comparison

Dense retrieval achieved the same Precision@1 as BM25 on this evaluation dataset.

Difference: 0 percentage points.

## Analysis

BM25 and dense retrieval both successfully identified the expected source document for all ten questions.

BM25 relies on lexical matching, while dense retrieval uses semantic embeddings. The identical results indicate that both approaches performed equally well for these questions and documents.

This evaluation does not establish that both methods will perform equally well on larger or more complex datasets.

## Limitations

- Small dataset of five documents.
- Only ten evaluation questions.
- Manual relevance labels.
- No latency or memory benchmark included in this evaluation.

## Conclusion

Both Haystack retrieval approaches achieved 100% Precision@1. Dense retrieval successfully matched BM25 performance on the selected evaluation questions.

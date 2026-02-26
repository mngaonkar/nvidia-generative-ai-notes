# Retrieval-Augmented Generation (RAG)

RAG extends LLMs with external knowledge by retrieving relevant documents and including them in the generation context. This addresses two fundamental LLM limitations: knowledge cutoff (the model doesn't know about recent information) and hallucination (the model generates plausible but incorrect facts).

## RAG Architecture

```
User Query
    ↓
┌──────────────┐     ┌──────────────┐
│ 1. Embed     │     │ Vector       │
│    Query     │────→│ Database     │
└──────────────┘     │ (similarity  │
                     │  search)     │
                     └──────┬───────┘
                            ↓
                     Top-k Documents
                            ↓
┌──────────────────────────────────┐
│ 2. Construct Augmented Prompt    │
│    [System] + [Retrieved Docs]   │
│    + [User Query]                │
└──────────────┬───────────────────┘
               ↓
┌──────────────────────────────────┐
│ 3. Generate Answer               │
│    (LLM with retrieved context)  │
└──────────────────────────────────┘
```

## Pipeline Components

### Document Processing

**Chunking** splits documents into pieces that fit within the embedding model's context and retrieval granularity:

| Strategy | Description | Best For |
|----------|-------------|----------|
| Fixed-size | Split every N tokens with M overlap | Simple documents |
| Recursive | Split by paragraphs → sentences → tokens | Structured text |
| Semantic | Split at topic boundaries using embeddings | Long-form content |
| Document-aware | Split respecting headers, sections, tables | Technical docs |

**Typical parameters**: 256-512 tokens per chunk, 10-20% overlap between adjacent chunks.

**Metadata preservation**: Store source document, page number, section heading, and timestamp with each chunk for citation and filtering.

### Embedding Generation

Convert text chunks and queries into dense vectors for similarity search. See [vector database embeddings](vector-db-embedding.md) for model choices.

Key considerations:
- **Dimensionality**: 384-3072 dimensions. Higher = more expressive but slower search.
- **Training objective**: Models trained for retrieval (contrastive loss) outperform general-purpose embeddings.
- **Instruction-tuned embeddings**: Some models accept task-specific prefixes (e.g., "Represent this document for retrieval:").

### Vector Database

Store and index embeddings for fast approximate nearest neighbor (ANN) search:

| Database | Type | Strengths |
|----------|------|-----------|
| **Milvus** | Open-source, distributed | Scalable, GPU-accelerated search, rich filtering |
| **Weaviate** | Open-source | Hybrid search, built-in vectorization |
| **Pinecone** | Managed service | Fully managed, low operational overhead |
| **FAISS** | Library | Fast, in-memory, GPU support, research-grade |
| **Chroma** | Open-source | Simple API, good for prototyping |
| **pgvector** | PostgreSQL extension | Integrates with existing Postgres infrastructure |

**Similarity metrics:**
- **Cosine similarity**: Measures angle between vectors. Normalized, range [-1, 1]. Most common.
- **Dot product**: Unnormalized cosine. Faster, works when embeddings are pre-normalized.
- **L2 (Euclidean)**: Measures distance. Lower is more similar.

### Retrieval

**Basic retrieval**: Embed query → find top-k nearest chunks → return.

**Re-ranking**: After initial retrieval, use a cross-encoder to score query-document pairs more accurately:

```
1. Retrieve top-50 chunks via ANN search (fast, approximate)
2. Re-rank with cross-encoder to get top-5 (slow, accurate)
3. Pass top-5 to LLM
```

Cross-encoders jointly encode the query and document, capturing fine-grained interactions that bi-encoder similarity misses.

**Hybrid search**: Combine dense retrieval (semantic similarity) with sparse retrieval (BM25 keyword matching):

```
score = α · dense_score + (1-α) · sparse_score
```

Hybrid search catches cases where semantic search fails (exact names, codes, numbers) and where keyword search fails (paraphrases, synonyms).

### Augmented Generation

Construct the final prompt with retrieved context:

```
System: You are a helpful assistant. Answer based on the provided context.
If the context doesn't contain the answer, say "I don't know."

Context:
[Retrieved Document 1]
[Retrieved Document 2]
[Retrieved Document 3]

User: {original query}
```

**Context window management**: With limited context length, prioritize:
1. Most relevant chunks first
2. Deduplicate overlapping content
3. Truncate if total exceeds budget (leave room for generation)

## Advanced RAG Techniques

### HyDE (Hypothetical Document Embeddings)

Generate a hypothetical answer first, embed it, then retrieve:
1. LLM generates a plausible (possibly wrong) answer
2. Embed this hypothetical answer
3. Retrieve documents similar to the hypothetical
4. Generate final answer from retrieved docs

HyDE improves retrieval for complex queries where the query embedding is far from relevant document embeddings.

### Multi-Query Retrieval

Generate multiple query variations, retrieve for each, merge results:
1. LLM generates 3-5 alternative phrasings of the original query
2. Run retrieval for each variation
3. Merge and deduplicate results
4. Feed unique results to the LLM

Captures different aspects of the query that a single embedding might miss.

### Self-RAG

The model decides when to retrieve and critiques its own responses:
1. Generate initial response
2. Model evaluates: "Do I need more information?"
3. If yes: retrieve, regenerate with context
4. Model evaluates: "Is my response supported by the context?"
5. Output final answer with confidence

### Contextual Compression

Rewrite retrieved chunks to remove irrelevant content:
1. Retrieve full chunks
2. LLM extracts only the relevant portions for the specific query
3. Pass compressed context to the generator

Reduces noise and fits more relevant information in the context window.

### Parent-Child Retrieval

Embed small chunks (children) for precise retrieval, but return larger chunks (parents) for context:
1. Split documents into large chunks (parents, ~2000 tokens)
2. Split each parent into small chunks (children, ~256 tokens)
3. Embed and index children
4. When a child matches, return the parent chunk

Provides both retrieval precision and generation context.

## NVIDIA Stack for RAG

| Component | NVIDIA Tool | Purpose |
|-----------|-------------|---------|
| Embedding model | NeMo + TensorRT | Encode queries and documents |
| Embedding serving | NIM (embedding) | Production embedding API |
| Re-ranking | NIM (reranking) | Cross-encoder scoring |
| Vector search | Milvus (GPU-accelerated) | Fast ANN search |
| Generator | NIM (LLM) | Answer generation |
| Safety | NeMo Guardrails | Input/output/retrieval rails |

## Evaluation

| Metric | Measures | How |
|--------|----------|-----|
| Recall@k | Retrieval quality | % of relevant docs in top-k |
| MRR (Mean Reciprocal Rank) | Retrieval ranking | Average 1/rank of first relevant doc |
| Faithfulness | Answer groundedness | Does the answer follow from retrieved context? |
| Answer relevancy | Response quality | Does the answer address the original question? |
| Context precision | Retrieval precision | Are retrieved docs actually relevant? |

Frameworks like RAGAS and TruLens automate RAG evaluation across these metrics.

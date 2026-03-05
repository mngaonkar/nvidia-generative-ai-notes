# Vector Database Reranking
Reranking is a crucial step in the retrieval process of a vector database. After an initial retrieval of relevant items based on vector similarity, reranking involves reordering the retrieved items using more sophisticated models or additional features to improve the relevance of the results.

## Reranking Techniques
1. **Cross-Encoder Models:** These models take the query and each retrieved item as input and output a relevance score. They are more accurate but computationally expensive.
2. **Bi-Encoder Models:** These models encode the query and retrieved items separately and then compute similarity scores. They are faster but may be less accurate than cross-encoders.
3. **Feature-Based Reranking:** This approach uses additional features (like metadata, user behavior, etc.) along with the initial similarity scores to rerank the results using machine learning models.
4. **Ensemble Methods:** Combining multiple reranking models to leverage their strengths and improve overall performance.


## Main Types of Rerankers

| Type | Accuracy (typical NDCG@10) | Latency | Cost | Best For | Drawbacks |
|------|---------------------------|---------|------|----------|-----------|
| Cross-Encoder | High (0.80–0.90+) | Medium (50–500 ms per pair) | Low (self-hosted) / Medium (API) | Production RAG, most common choice | Slower than bi-encoders |
| Lightweight / Optimized Cross-Encoder | High | Low–Medium | Very low | High-throughput, edge/AMD GPU setups | Slightly lower peak accuracy |
| LLM-based / Pointwise | Medium–High | High (1–5s+) | High (tokens) | Complex reasoning, zero-shot | Expensive, slow for scale |
| Multi-Vector / Late Interaction (e.g., ColBERT-style) | High | Medium | Low–Medium | Long docs, precise matching | More complex indexing |

## Most Popular Reranking Models
These show up consistently in benchmarks (MTEB-R, CMTEB-R, BEIR variants, internal NDCG/MRR gains), leaderboards, Hugging Face trending, and production RAG stacks (LangChain, LlamaIndex, Haystack, etc.):

| Model                                    | Organization                               | Architecture                 | Typical Use               | Notes                         |
| ---------------------------------------- | ------------------------------------------ | ---------------------------- | ------------------------- | ----------------------------- |
| **Cohere Rerank**                        | Cohere                                     | Cross-encoder                | RAG ranking               | API-based, very popular       |
| **bge-reranker-large**                   | Beijing Academy of Artificial Intelligence | Cross-encoder                | Dense retrieval reranking | Very strong open-source model |
| **monoT5**                               | Research (Castorini)                       | T5 seq2seq                   | Passage ranking           | Strong MS MARCO performance   |
| **duoT5**                                | Research (Castorini)                       | Pairwise T5                  | Passage ranking           | More accurate but slower      |
| **ColBERT**                              | Stanford                                   | Late-interaction transformer | Retrieval + reranking     | Very high ranking accuracy    |
| **cross-encoder/ms-marco-MiniLM-L-6-v2** | Hugging Face                               | Cross-encoder                | Lightweight reranking     | Fast and widely used          |
| **Jina Reranker v2**                     | Jina AI                                    | Cross-encoder                | Multilingual RAG          | High multilingual accuracy    |
| **RankT5**                               | Google Research                            | T5 ranking model             | Search ranking            | Used in ranking tasks         |

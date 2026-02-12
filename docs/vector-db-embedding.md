# Vector Database Embedding
Vector databases store and manage high-dimensional vector representations of data, enabling efficient similarity search and retrieval. Embeddings are crucial for transforming raw data (like text, images, or audio) into these vector representations.

## Types of Embeddings
**Text Embeddings:** Convert text into vectors. Common models include Word2Vec, GloVe, FastText, and transformer-based models like BERT and GPT. \
**Image Embeddings:** Convert images into vectors. Models like ResNet, VGG, and CLIP are commonly used for this purpose. \
**Audio Embeddings:** Convert audio signals into vectors. Models like OpenL3 and VGG are used for audio embedding.

## Popular Embedding Models for Vector DBs

| Model                          | Dimensions     | Optimized For                          |
|--------------------------------|---------------|----------------------------------------|
| OpenAI text-embedding-3-small  | 1536          | General purpose                       |
| OpenAI text-embedding-3-large  | 3072          | Higher accuracy                       |
| Cohere embed-v3                | 1024          | Multilingual                          |
| BGE (BAAI)                     | 768 / 1024    | Open source, strong performance       |
| E5 (Microsoft)                 | 768 / 1024    | Retrieval tasks                       |
| GTE (Alibaba)                  | 768           | General text                          |
| Sentence-T5                    | 768           | Sentence similarity                   |
| Instructor                     | 768           | Task-specific with instructions       |


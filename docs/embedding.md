# Embedding

Embeddings are the foundational mechanism that transforms discrete tokens (words, subwords, or characters) into continuous vector representations that a transformer can process. 

## Token Embedding
A transformer can't directly work with text—it needs numerical input. An embedding layer maps each token in your vocabulary to a dense vector of fixed dimensionality (e.g., 512 or 768 dimensions). So the word "cat" might become [0.2, -0.5, 0.8, ...].

### Key Properties
**Learned representations:** Unlike older one-hot encodings, embedding vectors are learned during training. The model discovers which dimensions capture useful semantic and syntactic relationships. \
**Semantic proximity:** Words with similar meanings end up with similar vectors. "King" and "queen" will be closer together in embedding space than "king" and "bicycle." \
**Compositionality:** The vectors encode relationships. The classic example: vector("king") - vector("man") + vector("woman") ≈ vector("queen").

## Positional Embeddings
Transformers process all tokens in parallel (unlike RNNs), so they have no inherent sense of word order. To fix this, positional embeddings are added to the token embeddings. These can be:

**Sinusoidal (original transformer):** Fixed patterns using sine/cosine functions at different frequencies \
**Learned:** Trainable vectors for each position \
**Rotary (RoPE):** Encodes position through rotation in the embedding space—now common in modern LLMs

Final embedding for transformer = token embedding + positional embedding

# Source Code
[PyTorch Embedding](https://github.com/mngaonkar/nvidia-generative-ai-notes/blob/main/src/pytorch-embedding.py) \
[Hugging Face Embedding](https://github.com/mngaonkar/nvidia-generative-ai-notes/blob/main/src/huggingface-embedding.py)
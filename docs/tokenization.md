# Tokenization

Tokenization is the process of converting raw text into a sequence of tokens that can be processed by a language model. A token can be a word, a subword, or even a character, depending on the tokenization strategy used.

## Types of Tokenization
**Word-level tokenization:** Each word is treated as a separate token. This can lead to a large vocabulary and issues with out-of-vocabulary (OOV) words.

**Subword tokenization:** Words are broken down into smaller units (subwords). This allows the model to handle OOV words by combining subwords. Common algorithms include Byte Pair Encoding (BPE) and WordPiece.

**Character-level tokenization:** Each character is treated as a token. This can capture fine-grained information but results in longer sequences. 

## Subword Tokenization Methods
**Byte Pair Encoding (BPE):** Iteratively merges the most frequent pairs of characters or subwords until a predefined vocabulary size is reached.

**WordPiece:** Similar to BPE but uses a different merging strategy based on likelihood. It is used in models like BERT.

**SentencePiece:** A more flexible tokenizer that can handle both word and subword tokenization. It is used in models like T5.

# Source Code
[Sentence Piece Tokenizer](https://github.com/mngaonkar/nvidia-generative-ai-notes/blob/main/src/sentence-piece-tokenizer.py)

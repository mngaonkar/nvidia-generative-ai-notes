# Encoder-Decoder Architecture in LLMs

## Overview

The encoder-decoder architecture is a fundamental design pattern in large language models (LLMs) and neural networks for sequence-to-sequence tasks. This architecture consists of two main components that work together to transform input sequences into output sequences.

## Architecture Components

### Encoder

The encoder processes the input sequence and creates a contextual representation (embedding) that captures the semantic meaning of the input.

**Key Characteristics:**
- **Input Processing**: Takes variable-length input sequences (text, tokens, etc.)
- **Contextual Embeddings**: Generates rich representations that encode the meaning and relationships within the input
- **Bidirectional Context**: Can attend to both past and future tokens in the input sequence
- **Compression**: Condenses the input information into a fixed or variable-length representation

**Common Encoder Architectures:**
- **BERT** (Bidirectional Encoder Representations from Transformers)
- **RoBERTa** (Robustly Optimized BERT)
- **ALBERT** (A Lite BERT)
- Encoder-only models focus on understanding and representation

### Decoder

The decoder takes the encoder's output representation and generates the target output sequence one token at a time.

**Key Characteristics:**
- **Auto-regressive Generation**: Generates output sequentially, using previously generated tokens
- **Unidirectional Context**: Typically attends only to past tokens to maintain causality
- **Conditional Generation**: Generates output conditioned on the encoder's representation
- **Variable-Length Output**: Can produce sequences of different lengths than the input

**Common Decoder Architectures:**
- **GPT** series (Generative Pre-trained Transformer)
- **GPT-2, GPT-3, GPT-4**
- Decoder-only models focus on generation

## Full Encoder-Decoder Models

Some models use both encoder and decoder components together:

### Transformer Architecture

The original Transformer model (Vaswani et al., 2017) introduced the encoder-decoder architecture with attention mechanisms.

**Components:**
1. **Encoder Stack**: Multiple layers of self-attention and feed-forward networks
2. **Decoder Stack**: Multiple layers with masked self-attention, cross-attention, and feed-forward networks
3. **Attention Mechanisms**:
   - Self-attention in encoder
   - Masked self-attention in decoder
   - Cross-attention between encoder and decoder

### Popular Encoder-Decoder Models

- **T5** (Text-to-Text Transfer Transformer)
- **BART** (Bidirectional and Auto-Regressive Transformers)
- **mBART** (Multilingual BART)
- **mT5** (Multilingual T5)

## Attention Mechanisms

### Self-Attention (Encoder)

Allows each position in the encoder to attend to all positions in the input sequence.

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

Where:
- Q (Query): Current token representation
- K (Key): All token representations
- V (Value): All token representations
- d_k: Dimension of key vectors

### Masked Self-Attention (Decoder)

Prevents positions from attending to future positions during training.

### Cross-Attention

Allows the decoder to attend to the encoder's output, enabling the model to focus on relevant parts of the input when generating output.

## Use Cases

### Encoder-Decoder Models Excel At:

1. **Machine Translation**: Translating text from one language to another
2. **Summarization**: Condensing long documents into concise summaries
3. **Question Answering**: Generating answers based on context
4. **Text-to-Text Tasks**: Any task framed as converting one text to another
5. **Dialogue Systems**: Generating responses based on conversation history

### Encoder-Only Models Excel At:

1. **Classification**: Sentiment analysis, topic classification
2. **Named Entity Recognition (NER)**
3. **Extractive Question Answering**
4. **Semantic Similarity**

### Decoder-Only Models Excel At:

1. **Text Generation**: Creative writing, code generation
2. **Completion Tasks**: Autocomplete, text continuation
3. **Few-shot Learning**: In-context learning with prompts
4. **Conversational AI**: Open-ended dialogue

## Training Objectives

### Encoder Training

- **Masked Language Modeling (MLM)**: Predicting masked tokens in the input (BERT)
- **Contrastive Learning**: Learning representations through contrast
- **Next Sentence Prediction**: Understanding sentence relationships

### Decoder Training

- **Causal Language Modeling (CLM)**: Predicting the next token given previous tokens
- **Auto-regressive Modeling**: Sequential generation

### Encoder-Decoder Training

- **Sequence-to-Sequence**: Mapping input sequences to output sequences
- **Span Corruption**: T5's approach of corrupting input spans and reconstructing them
- **Denoising**: BART's approach using various noise functions

## Advantages and Disadvantages

### Encoder-Decoder Advantages:
- ✅ Explicit separation of understanding and generation
- ✅ Better for tasks requiring input transformation
- ✅ Can handle different input/output lengths naturally
- ✅ Cross-attention provides interpretability

### Encoder-Decoder Disadvantages:
- ❌ More parameters and computational overhead
- ❌ More complex architecture
- ❌ Requires paired training data for many tasks

### Encoder-Only Advantages:
- ✅ Bidirectional context for better understanding
- ✅ Efficient for classification and extraction tasks
- ✅ Smaller model size for specific tasks

### Decoder-Only Advantages:
- ✅ Simpler architecture
- ✅ Excellent generative capabilities
- ✅ Strong few-shot learning abilities
- ✅ Can handle diverse tasks through prompting

## Modern Trends

### Decoder-Only Dominance

Recent trends show decoder-only models (like GPT series) becoming dominant because:
- Simpler architecture is easier to scale
- Strong performance on diverse tasks through prompting
- Unified training objective (next token prediction)
- Effective few-shot and zero-shot learning

### Hybrid Approaches

Some modern approaches combine benefits:
- **Prefix LM**: Encoder-like processing for prefix, decoder for completion
- **UL2**: Unified framework with different denoising objectives
- **Mixture-of-Denoisers**: Combining multiple training paradigms

## Key Takeaways

1. **Encoder-Decoder** architecture provides explicit separation between understanding (encoder) and generation (decoder)
2. **Encoders** excel at creating rich representations for understanding tasks
3. **Decoders** excel at sequential generation and auto-regressive tasks
4. **Full encoder-decoder** models are powerful for sequence-to-sequence transformations
5. **Modern LLMs** increasingly favor decoder-only architectures for their simplicity and versatility
6. **Choice of architecture** depends on the specific task requirements and constraints

## References

- Vaswani et al. (2017): "Attention Is All You Need"
- Devlin et al. (2018): "BERT: Pre-training of Deep Bidirectional Transformers"
- Radford et al. (2018): "Improving Language Understanding by Generative Pre-Training" (GPT)
- Raffel et al. (2020): "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (T5)
- Lewis et al. (2020): "BART: Denoising Sequence-to-Sequence Pre-training"

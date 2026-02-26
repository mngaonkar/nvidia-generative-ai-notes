# Multimodal Models

Multimodal models process and generate across multiple data types — text, images, audio, and video. Vision-Language Models (VLMs) are the dominant multimodal architecture, extending LLMs to understand images alongside text.

## VLM Architecture

```
Image → [Patch Embedding] → [Vision Encoder (ViT)] → [Projection Layer] ─┐
                                                                          ↓
Text  → [Token Embedding] ────────────────────────────────────→ [LLM Decoder] → Output
```

### Vision Encoder

**Vision Transformer (ViT)** processes images as sequences of patches:

1. Split image into fixed-size patches (e.g., 14×14 or 16×16 pixels)
2. Flatten each patch into a vector
3. Project through a linear embedding layer
4. Add positional embeddings
5. Process through transformer encoder layers

A 224×224 image with 14×14 patches produces 256 tokens (16×16 grid). A 336×336 image produces 576 tokens.

**CLIP encoders** are commonly used because they're pre-trained to align image and text representations via contrastive learning. This alignment provides a strong foundation for connecting vision to language.

### Projection Layer

Maps vision encoder output space to LLM embedding space:

- **Linear projection**: Simple matrix multiply. Fast, works surprisingly well.
- **MLP projection**: 2-layer MLP with activation. Better alignment, slightly more parameters.
- **Cross-attention**: Query from LLM, key/value from vision encoder. Most expressive, most expensive.

The projection layer is often the only component trained from scratch when building a VLM from pretrained vision and language components.

### LLM Decoder

Standard autoregressive transformer that processes interleaved vision and text tokens:

```
Input sequence: [IMG_1] [IMG_2] ... [IMG_N] [BOS] Describe this image: ...
                 ↑ vision tokens               ↑ text tokens
```

The LLM attends to both vision and text tokens through its standard self-attention mechanism. No architectural modification is needed — vision tokens are simply additional context tokens.

## Training Strategies

### Stage 1: Vision-Language Alignment (Pretraining)

- **Freeze**: Vision encoder + LLM
- **Train**: Projection layer only
- **Data**: Large-scale image-caption pairs (e.g., LAION, CC3M)
- **Objective**: Align vision features to LLM embedding space
- **Cost**: Low (only projection layer parameters)

### Stage 2: Visual Instruction Tuning

- **Freeze**: Vision encoder
- **Train**: Projection layer + LLM (full or LoRA)
- **Data**: Visual instruction-following data (VQA, detailed descriptions, reasoning)
- **Objective**: Teach the model to follow visual instructions
- **Cost**: Medium (LLM fine-tuning)

### Stage 3 (Optional): End-to-End Fine-Tuning

- **Train**: All components (vision encoder + projection + LLM)
- **Data**: Domain-specific visual data
- **Objective**: Maximum task performance
- **Cost**: High (entire model)

## Notable VLM Architectures

| Model | Vision Encoder | LLM | Projection | Training Approach |
|-------|---------------|-----|------------|-------------------|
| LLaVA | CLIP ViT-L | LLaMA/Vicuna | Linear/MLP | 2-stage (align + instruct) |
| LLaVA-1.5 | CLIP ViT-L-336 | LLaMA-2 | 2-layer MLP | Higher resolution, better data |
| InternVL | InternViT | InternLM | QLLaMA | Dynamic resolution |
| Qwen-VL | ViT + resampler | Qwen | Cross-attention | 3-stage training |
| VILA | SigLIP | LLaMA | Linear | Interleaved image-text pretraining |
| Phi-3-Vision | CLIP ViT | Phi-3 | MLP | Compact, efficient |

## Use Cases

- **Visual Question Answering (VQA)**: "What color is the car in this image?"
- **Image captioning**: Generate detailed descriptions of images
- **Document understanding**: Extract information from PDFs, charts, tables
- **OCR + reasoning**: Read text in images and reason about it
- **GUI understanding**: Navigate and interact with user interfaces
- **Medical imaging**: Analyze X-rays, MRIs with clinical context
- **Multimodal agents**: Vision-equipped agents that can browse, code, and interact

## Challenges

### Resolution vs. Compute

Higher resolution = more patches = more tokens = quadratically more attention compute:

| Resolution | Patch Size | Tokens | Relative Cost |
|-----------|-----------|--------|---------------|
| 224×224 | 14×14 | 256 | 1x |
| 336×336 | 14×14 | 576 | ~5x |
| 448×448 | 14×14 | 1024 | ~16x |
| 672×672 | 14×14 | 2304 | ~81x |

**Solutions:**
- Dynamic resolution: resize images to minimize padding
- Token pooling/merging: reduce vision tokens after encoding
- Tiled processing: split large images into tiles, encode separately

### Batch Size Constraints

Vision tokens significantly increase sequence length, reducing effective batch size. A batch of 8 images at 576 tokens each adds 4,608 tokens to each sequence.

### Hallucination

VLMs can hallucinate visual details — describing objects not present in the image. This is an active research area with approaches including:
- Improved training data (negative examples)
- Reinforcement learning from human feedback on visual tasks
- Constrained decoding with visual verification

## NeMo Multimodal Support

NeMo 2.0 provides infrastructure for multimodal training:

- **Model definitions**: ViT + GPT configurations for VLMs
- **Distributed training**: TP/PP support for both vision and language components
- **Data pipelines**: Image-text pair loading, augmentation, and preprocessing
- **Export**: TensorRT compilation for vision encoder and LLM jointly
- **NIM**: Vision-Language NIMs for production deployment (e.g., VILA, LLaVA)

See [NeMo 2.0](nemo-2.0.md) for framework details and [NIM](nvidia-nim.md) for deployment.

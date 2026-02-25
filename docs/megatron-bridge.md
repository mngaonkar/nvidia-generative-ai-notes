# Megatron Core
Megatron Core is an open-source PyTorch-based library that contains GPU-optimized techniques and cutting-edge system-level optimizations. It abstracts them into composable and modular APIs, allowing full flexibility for developers and model researchers to train custom transformers at-scale on NVIDIA accelerated computing infrastructure.

Megatron Core is the low-level high-performance training library extracted from Megatron-LM.

It provides:
- Tensor Parallelism (TP)
- Pipeline Parallelism (PP)
- Sequence Parallelism (SP)
- Expert Parallelism (MoE)
- Optimized transformer layers
- Fused kernels
- FP8 / BF16 / mixed precision
- Transformer Engine integration


# Megatron Bridge

NeMo Megatron Bridge is a PyTorch-native library within the NeMo Framework that provides pretraining, SFT and LoRA for popular LLM and VLM models.
Megatron Bridge is a compatibility layer between Megatron Core/NeMo and other ecosystems like HuggingFace.

The part of NeMo 2.0 that lets you:
- Seamless bi-directional conversion between Hugging Face and Megatron formats 
- Train very large models (e.g., 1B+, GPT, T5, BLOOM)
- Use Megatron-Core-style data formats
- Use tensor parallelism, pipeline parallelism
- Handles [Check Point Translation](checkpoint-translation.md) (shard/unshard checkpoints) 
- Scale to multi-node

If you’re training LLMs (not just fine-tuning small models), Megatron-Bridge is the backend that powers large-scale distributed training.

# Practical Example
## Scenario 1: Training Llama 70B on 64 GPUs
You use:
- Megatron Core
- NeMo 2.0
- torchrun / nemo run

Megatron Core handles:
- TP=8
- PP=4
- SP enabled
- Fused kernels

## Scenario 2: You want to export that model to HuggingFace
You use:
- Megatron Bridge
It:
- Reassembles sharded weights
- Converts layer names
- Creates HF-compatible config.json
- Outputs a usable HF checkpoint
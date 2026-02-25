# Megatron Core
Megatron Core is an open-source PyTorch-based library that contains GPU-optimized techniques and cutting-edge system-level optimizations. It abstracts them into composable and modular APIs, allowing full flexibility for developers and model researchers to train custom transformers at-scale on NVIDIA accelerated computing infrastructure.

# Megatron Bridge

NeMo Megatron Bridge is a PyTorch-native library within the NeMo Framework that provides pretraining, SFT and LoRA for popular LLM and VLM models.

The part of NeMo 2.0 that lets you:
- Seamless bi-directional conversion between Hugging Face and Megatron formats 
- Train very large models (e.g., 1B+, GPT, T5, BLOOM)
- Use Megatron-Core-style data formats
- Use tensor parallelism, pipeline parallelism
- Scale to multi-node

If you’re training LLMs (not just fine-tuning small models), Megatron-Bridge is the backend that powers large-scale distributed training.


# Megatron Bridge

The part of NeMo 2.0 that lets you:
- Train very large models (e.g., 1B+, GPT, T5, BLOOM)
- Use Megatron-Core-style data formats
- Use tensor parallelism, pipeline parallelism
- Scale to multi-node

If you’re training LLMs (not just fine-tuning small models), Megatron-Bridge is the backend that powers large-scale distributed training.
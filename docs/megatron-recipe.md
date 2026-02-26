# Megatron Recipes

Megatron recipes are pre-configured training setups that encode best practices for common model architectures and scales. They specify model architecture, parallelism strategy, optimizer settings, and training hyperparameters — providing a tested, reproducible starting point.

## Purpose

- Eliminate trial-and-error for standard architectures
- Encode NVIDIA's internal benchmarking results
- Provide reproducible baselines for research
- Serve as starting points for custom models

## Common Model Configurations

| Model Size | Layers | Hidden | Heads | TP | PP | Micro BS | Global BS | Seq Len |
|-----------|--------|--------|-------|----|----|----------|-----------|---------|
| 1.3B | 24 | 2048 | 16 | 1 | 1 | 4 | 256 | 2048 |
| 7B | 32 | 4096 | 32 | 1 | 1 | 2 | 512 | 4096 |
| 13B | 40 | 5120 | 40 | 2 | 1 | 1 | 512 | 4096 |
| 70B | 80 | 8192 | 64 | 8 | 4 | 1 | 1024 | 4096 |
| 175B | 96 | 12288 | 96 | 8 | 8 | 1 | 1536 | 2048 |

TP (Tensor Parallelism) stays within a node (NVLink). PP (Pipeline Parallelism) spans across nodes (InfiniBand). Global batch size scales with the number of data-parallel replicas.

## Recipe Components

A complete recipe includes:

- **Model architecture**: Layer count, hidden dimensions, attention heads, FFN size, vocabulary size
- **Parallelism strategy**: TP, PP, DP degrees, sequence parallelism, activation checkpointing
- **Optimizer**: AdamW with β1=0.9, β2=0.95, weight decay, gradient clipping
- **Learning rate schedule**: Warmup steps, cosine decay, min LR
- **Batch size**: Micro batch per GPU, global batch size, gradient accumulation
- **Mixed precision**: BF16 or FP8 (Transformer Engine)
- **Data**: Tokenizer, data blending ratios, sequence length

## Using Recipes with NeMo-Run

```python
from nemo.collections.llm import GPTConfig7B, pretrain
from nemo.lightning import MegatronStrategy

# Load a pre-defined recipe
config = GPTConfig7B()

# Override specific settings for your hardware
config.trainer.num_nodes = 4
config.trainer.devices = 8  # GPUs per node
config.model.tensor_model_parallel_size = 1
config.model.pipeline_model_parallel_size = 1

pretrain(config)
```

## Customizing Recipes

1. Start with the recipe closest to your target model size
2. Adjust parallelism for your GPU count and memory
3. Scale batch size with data parallelism degree
4. Tune learning rate: scale with `sqrt(batch_size)` or linearly
5. Enable FP8 on H100 GPUs for ~2x throughput

## When to Use Recipes vs. Custom Config

| Scenario | Approach |
|----------|----------|
| Standard architecture at known scale | Use recipe directly |
| Standard architecture, different hardware | Recipe + adjust parallelism |
| Custom architecture | Recipe as starting point, modify architecture params |
| Novel training objective | Custom config, use recipe optimizer/schedule settings |

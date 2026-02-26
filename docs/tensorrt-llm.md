# TensorRT-LLM

TensorRT-LLM is NVIDIA's library for compiling and optimizing LLMs for production inference. It transforms model checkpoints into highly optimized inference engines with kernel fusion, quantization, and advanced batching — delivering 2-5x speedup over native PyTorch inference.

## Why TensorRT-LLM?

PyTorch inference is flexible but leaves significant performance on the table:
- No kernel fusion across operations
- No graph-level optimization
- Limited quantization support
- No built-in continuous batching

TensorRT-LLM addresses all of these by compiling the model into an optimized execution graph.

## Compilation Workflow

```
Model Checkpoint (HuggingFace / NeMo / Megatron)
    ↓
1. Convert to TensorRT-LLM checkpoint format
    ↓
2. Build TensorRT engine (.engine file)
    ↓
3. Deploy via Triton or direct API
```

### Step 1: Checkpoint Conversion

```bash
# Convert HuggingFace checkpoint to TRT-LLM format
python convert_checkpoint.py \
    --model_dir /models/llama-70b-hf \
    --output_dir /checkpoints/llama-70b-trtllm \
    --dtype float16 \
    --tp_size 8
```

### Step 2: Engine Build

```bash
trtllm-build \
    --checkpoint_dir /checkpoints/llama-70b-trtllm \
    --output_dir /engines/llama-70b \
    --gemm_plugin float16 \
    --max_batch_size 64 \
    --max_input_len 2048 \
    --max_seq_len 4096 \
    --paged_kv_cache enable \
    --use_inflight_batching
```

### Step 3: Run Inference

```python
import tensorrt_llm
from tensorrt_llm.runtime import ModelRunner

runner = ModelRunner.from_dir("/engines/llama-70b")
outputs = runner.generate(input_texts=["Explain quantum computing:"], max_new_tokens=256)
```

## Key Optimizations

### Kernel Fusion

Combines multiple operations into single GPU kernels:
- QKV projection fusion (3 matmuls → 1)
- Attention + softmax + output projection
- LayerNorm + bias + activation
- Residual connections fused with subsequent operations

### Graph Optimization

- Constant folding and dead code elimination
- Memory planning and reuse across layers
- Optimal kernel selection per GPU architecture (H100, A100, etc.)

### Quantization Support

TensorRT-LLM supports multiple precision modes at build time:

| Mode | Description | Typical Speedup |
|------|-------------|----------------|
| FP16 | Standard half-precision | Baseline |
| FP8 | H100 Transformer Engine | ~2x |
| INT8 (SmoothQuant) | W8A8 with smoothing | ~1.5-2x |
| INT4 (AWQ) | 4-bit weight-only | ~2-3x |
| INT4 (GPTQ) | 4-bit weight-only | ~2-3x |

```bash
# Build with INT4 AWQ quantization
trtllm-build --checkpoint_dir /checkpoints/llama-70b-awq \
    --output_dir /engines/llama-70b-int4 \
    --quant_mode int4_awq
```

### In-Flight Batching

Also called continuous batching — new requests enter the batch as completed ones exit:
- No waiting for the longest sequence to finish
- GPU utilization stays high (>90%)
- Up to 3-5x throughput improvement over static batching

### Paged KV Cache

Manages KV cache memory like OS virtual memory:
- Non-contiguous physical memory blocks
- Block table maps logical to physical pages
- Eliminates memory fragmentation
- Enables memory sharing across beam search candidates

## Multi-GPU Inference

TensorRT-LLM supports multi-GPU inference for models that don't fit on a single GPU:

- **Tensor Parallelism (TP)**: Split layers across GPUs within a node
- **Pipeline Parallelism (PP)**: Split model stages across nodes
- **Combination**: TP within node, PP across nodes

```bash
# Build for 8-GPU tensor parallelism
trtllm-build --checkpoint_dir /checkpoints/llama-70b \
    --tp_size 8 --pp_size 1 \
    --output_dir /engines/llama-70b-tp8

# Launch with MPI
mpirun -n 8 python run_inference.py --engine_dir /engines/llama-70b-tp8
```

## Supported Architectures

TensorRT-LLM includes optimized implementations for:
- LLaMA, LLaMA 2, LLaMA 3
- GPT-2, GPT-J, GPT-NeoX
- Falcon, MPT, Phi
- Mixtral (MoE), DBRX
- Gemma, Qwen, Baichuan
- BLOOM, ChatGLM
- Encoder-decoder: T5, BART

## Performance Characteristics

Typical speedups over PyTorch (FP16, batch size 1):

| Model | PyTorch | TRT-LLM FP16 | TRT-LLM INT4 |
|-------|---------|--------------|--------------|
| LLaMA 7B | 35 tok/s | 85 tok/s | 150 tok/s |
| LLaMA 70B (8xH100) | 15 tok/s | 40 tok/s | 80 tok/s |

Actual numbers depend on hardware, batch size, sequence length, and quantization.

## Integration with Triton

TensorRT-LLM engines deploy via [Triton Inference Server](nemo-triton.md):
- Triton's TRT-LLM backend handles request scheduling
- In-flight batching managed at the Triton level
- Streaming via Server-Sent Events (SSE)
- Prometheus metrics for monitoring

## Integration with NIM

[NVIDIA NIM](nvidia-nim.md) packages TensorRT-LLM engines with Triton into pre-built containers, providing one-command deployment with automatic optimization for the target GPU.

# NVIDIA NIM (Inference Microservices)

NVIDIA NIM packages optimized LLM inference into pre-built, containerized microservices. It abstracts the complexity of TensorRT-LLM compilation, Triton configuration, and optimization tuning into a single container with an OpenAI-compatible API.

## Why NIM?

Deploying an optimized LLM in production requires:
1. Converting checkpoints to TensorRT-LLM format
2. Building optimized engines for specific hardware
3. Configuring Triton Inference Server
4. Tuning batch sizes, KV cache, and quantization
5. Setting up monitoring and health checks

NIM handles all of this automatically, reducing deployment from days of engineering to a single command.

## Architecture

```
Client (OpenAI-compatible API)
    ↓ HTTP/gRPC
┌─────────────────────────────┐
│         NIM Container        │
│  ┌───────────────────────┐  │
│  │   API Gateway          │  │
│  │   (OpenAI-compatible)  │  │
│  └───────────┬───────────┘  │
│              ↓               │
│  ┌───────────────────────┐  │
│  │   TensorRT-LLM Engine │  │
│  │   (auto-optimized)    │  │
│  └───────────┬───────────┘  │
│              ↓               │
│  ┌───────────────────────┐  │
│  │   GPU Execution       │  │
│  │   (multi-GPU support) │  │
│  └───────────────────────┘  │
└─────────────────────────────┘
```

## Quick Start

```bash
# Pull and run a NIM container
docker run -d --gpus all \
  -e NGC_API_KEY=$NGC_API_KEY \
  -p 8000:8000 \
  nvcr.io/nvidia/nim/meta-llama3-70b-instruct:latest
```

```python
# Use with OpenAI SDK
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-used")

response = client.chat.completions.create(
    model="meta-llama3-70b-instruct",
    messages=[{"role": "user", "content": "Explain transformers in 3 sentences."}],
    max_tokens=256
)
```

## Key Features

### Automatic Optimization

NIM automatically selects the best optimization profile for the target GPU:
- Quantization level (FP16, FP8, INT8, INT4) based on GPU memory
- Tensor parallelism degree based on GPU count
- KV cache configuration based on available memory
- Batch size tuning for throughput vs. latency

### OpenAI-Compatible API

Drop-in replacement for OpenAI endpoints:
- `/v1/chat/completions` — chat interface
- `/v1/completions` — text completion
- `/v1/models` — list available models
- `/v1/embeddings` — embedding generation (for embedding NIMs)
- Streaming support via SSE

### Multi-GPU Support

NIM automatically distributes models across available GPUs:
- Single container, multiple GPUs
- Tensor parallelism for large models
- No manual sharding configuration

## Available NIM Types

| NIM Type | Purpose | Examples |
|----------|---------|---------|
| LLM | Text generation | LLaMA, Mistral, Mixtral, Gemma |
| Embedding | Dense vector generation | NV-Embed, E5 |
| Reranking | Cross-encoder scoring | NV-RerankQA |
| Vision-Language | Multimodal generation | LLaVA, VILA |
| Speech | ASR and TTS | Riva, Parakeet |

## Deployment Options

### Local / On-Premises

```bash
# Single GPU deployment
docker run --gpus '"device=0"' -p 8000:8000 nvcr.io/nvidia/nim/model:latest

# Multi-GPU deployment
docker run --gpus all -p 8000:8000 nvcr.io/nvidia/nim/model:latest
```

### Kubernetes

Helm charts for production deployment:
- Horizontal pod autoscaling based on GPU utilization
- Rolling updates for model version changes
- Persistent volume claims for model caching
- Ingress configuration for load balancing

### Cloud

NIM runs on any cloud with NVIDIA GPUs:
- AWS (p4d, p5 instances)
- Azure (ND A100, ND H100)
- GCP (A3, A3 Mega)
- NVIDIA DGX Cloud

## Integration Points

- **[NeMo Guardrails](nemo-guardrails.md)**: Add runtime safety policies
- **[RAG](rag.md)**: Combine LLM NIM with Embedding NIM for retrieval pipelines
- **[Triton](nemo-triton.md)**: NIM uses Triton under the hood; custom Triton configs can be injected
- **Observability**: Prometheus metrics, OpenTelemetry tracing

## Custom Model Deployment

Deploy your own fine-tuned models via NIM:

1. Export NeMo checkpoint to HuggingFace format (via [Megatron Bridge](megatron-bridge.md))
2. Use NIM's model store to register the custom model
3. NIM handles TensorRT-LLM compilation and optimization automatically

## NIM vs. Manual Deployment

| Aspect | NIM | Manual (TRT-LLM + Triton) |
|--------|-----|---------------------------|
| Setup time | Minutes | Days |
| Optimization | Automatic | Manual tuning |
| API compatibility | OpenAI-compatible | Custom |
| Flexibility | Standard configs | Full control |
| Updates | Container pull | Manual rebuild |
| Best for | Production teams | ML infrastructure teams |

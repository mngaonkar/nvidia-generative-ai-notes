# Triton Inference Server

Triton Inference Server is NVIDIA's production-grade inference serving platform. It supports multiple frameworks (PyTorch, TensorFlow, ONNX, TensorRT, TensorRT-LLM) and provides dynamic batching, concurrent model execution, and comprehensive monitoring.

## Why Triton?

Moving from notebook inference to production requires:
- Request batching and scheduling
- Multi-model serving on shared GPUs
- Health checks, metrics, and monitoring
- Model versioning and A/B testing
- Horizontal scaling

Triton solves all of these with a single, unified serving platform.

## Deployment Workflow

### Step 1: Export from NeMo

```
NeMo Checkpoint → TensorRT-LLM Engine → Triton Model Repository
                → ONNX → Triton Model Repository
```

### Step 2: Organize Model Repository

```
model_repository/
├── llama_70b/
│   ├── config.pbtxt
│   ├── 1/
│   │   └── model.plan          # TensorRT-LLM engine
│   └── 2/
│       └── model.plan          # Version 2 (for A/B testing)
└── embedding_model/
    ├── config.pbtxt
    └── 1/
        └── model.onnx
```

### Step 3: Configure and Launch

```bash
tritonserver --model-repository=/models --log-verbose=1
```

## Key Features

### Dynamic Batching

Automatically groups incoming requests into batches:
- Configurable maximum batch size and delay
- Increases throughput by 2-5x for compute-bound models
- No client-side batching logic needed

### TensorRT-LLM Backend

Purpose-built for LLM inference:
- In-flight batching (continuous batching)
- Paged KV cache management
- Multi-GPU inference (TP, PP)
- Streaming token generation via SSE
- Speculative decoding support

### Concurrent Model Execution

- Multiple models on the same GPU (MPS or time-slicing)
- Model ensembles: chain preprocessing → inference → postprocessing
- Instance groups for GPU allocation control

### Model Analyzer

Profile and optimize model configuration:
- Sweep batch sizes, instance counts, and concurrency
- Find optimal throughput/latency tradeoff
- GPU memory utilization analysis

## Monitoring

Triton exposes Prometheus-compatible metrics:
- Request latency (p50, p95, p99)
- Throughput (requests/sec, tokens/sec)
- GPU utilization and memory
- Queue depth and batch sizes
- Per-model statistics

## Integration Points

- **TensorRT-LLM**: Primary backend for LLM workloads
- **NIM**: Uses Triton under the hood for containerized deployment
- **NeMo**: Export pipeline produces Triton-ready artifacts
- **Kubernetes**: Helm charts and autoscaling support
- **Load balancers**: HTTP/gRPC endpoints for standard infrastructure

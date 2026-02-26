# GPU Hardware and Interconnects

The NVIDIA generative AI stack is hardware-aware by design. Understanding GPU architectures, interconnects, and system topologies is essential for choosing optimal parallelism strategies and achieving peak training/inference throughput.

## GPU Architectures

### Hopper (H100, H200)

The current workhorse for LLM training and inference:

| Spec | H100 SXM | H200 SXM |
|------|----------|----------|
| Tensor Cores | 4th gen (FP8 support) | 4th gen (FP8 support) |
| HBM | 80 GB HBM3 | 141 GB HBM3e |
| Memory bandwidth | 3.35 TB/s | 4.8 TB/s |
| FP16 Tensor | 989 TFLOPS | 989 TFLOPS |
| FP8 Tensor | 1,979 TFLOPS | 1,979 TFLOPS |
| NVLink | 900 GB/s | 900 GB/s |
| TDP | 700W | 700W |

**Key features:**
- Native FP8 support via Transformer Engine — 2x throughput over FP16
- 4th-gen Tensor Cores with sparsity support (2:4 structured sparsity)
- DMA engines for overlapping communication with computation
- Thread Block Cluster for cooperative kernels across SMs

### Blackwell (B100, B200, GB200)

Next-generation architecture:

| Spec | B200 |
|------|------|
| Tensor Cores | 5th gen (FP4/FP6 support) |
| HBM | 192 GB HBM3e |
| Memory bandwidth | 8 TB/s |
| FP8 Tensor | ~4,500 TFLOPS |
| FP4 Tensor | ~9,000 TFLOPS |
| NVLink | 1,800 GB/s |

**Key improvements:**
- FP4 precision for inference — 2x throughput over FP8
- Second-generation Transformer Engine with improved dynamic scaling
- Doubled NVLink bandwidth
- Decompression engine for compressed data in memory
- Reliability and availability (RAS) engines for large-scale clusters

### Grace Hopper Superchip (GH200)

CPU + GPU unified architecture:
- Grace CPU (Arm Neoverse, 72 cores) + H200 GPU
- NVLink-C2C: 900 GB/s coherent CPU-GPU interconnect
- 624 GB unified memory (480 GB CPU + 144 GB GPU)
- Eliminates PCIe bottleneck for CPU-GPU data transfer

## Transformer Engine

Built into H100 and later GPUs, Transformer Engine automatically manages FP8 precision during training:

### How It Works

1. **Amax tracking**: Monitor the maximum absolute value of each tensor over a sliding window
2. **Scale computation**: `scale = FP8_MAX / amax` — maps the tensor range to FP8 representable range
3. **Delayed scaling**: Use amax from previous iterations (more stable than per-iteration scaling)
4. **Selective precision**: Forward pass in E4M3 (more mantissa bits), backward pass in E5M2 (more exponent range)

### FP8 Formats

| Format | Exponent | Mantissa | Range | Precision | Use |
|--------|----------|----------|-------|-----------|-----|
| E4M3 | 4 bits | 3 bits | ±448 | Higher | Forward pass |
| E5M2 | 5 bits | 2 bits | ±57344 | Lower | Backward pass (gradients) |

### Integration

Transformer Engine is integrated into Megatron-Core. Enable with:

```python
# In Megatron-Core config
--fp8-format hybrid          # E4M3 forward, E5M2 backward
--fp8-amax-history-len 1024  # Sliding window for amax tracking
--fp8-amax-compute-algo max  # How to compute amax from history
--transformer-impl transformer_engine
```

**Result:** ~2x training throughput on H100 vs. BF16, with minimal accuracy loss for most architectures.

## Interconnects

### NVLink

High-bandwidth, low-latency GPU-to-GPU interconnect:

| Generation | Per-GPU Bandwidth | GPUs Connected |
|-----------|-------------------|----------------|
| NVLink 3.0 (A100) | 600 GB/s | Up to 8 via NVSwitch |
| NVLink 4.0 (H100) | 900 GB/s | Up to 8 via NVSwitch |
| NVLink 5.0 (B200) | 1,800 GB/s | Up to 8 via NVSwitch |

**Use case:** Tensor Parallelism within a node. TP requires all-reduce after every layer, demanding highest bandwidth and lowest latency.

### NVSwitch

Chip-level switch that provides full-bandwidth NVLink connectivity between all GPUs in a node:

```
Without NVSwitch: GPU0 ←→ GPU1 ←→ GPU2 (ring, limited paths)
With NVSwitch:    GPU0 ←→ GPU1, GPU0 ←→ GPU2, GPU1 ←→ GPU2 (full mesh)
```

- Every GPU pair has full NVLink bandwidth
- All-to-all communication at full speed
- Essential for efficient tensor parallelism and expert parallelism (MoE)

### InfiniBand

Node-to-node networking for multi-node training:

| Generation | Bandwidth (per port) | Latency |
|-----------|---------------------|---------|
| HDR | 200 Gbps | ~1 μs |
| NDR | 400 Gbps | ~1 μs |
| XDR | 800 Gbps | <1 μs |

**Use case:** Pipeline Parallelism and Data Parallelism across nodes.

Features:
- RDMA (Remote Direct Memory Access) — GPU-to-GPU without CPU involvement
- GPUDirect RDMA — data moves directly between GPU memory across nodes
- Adaptive routing for congestion avoidance
- In-network computing (SHARP) for accelerated all-reduce

## DGX Systems

### DGX H100

The building block for large-scale training:
- 8x H100 SXM GPUs
- 4th-gen NVSwitch (full NVLink mesh)
- 640 GB total GPU memory
- 8x 400 Gbps InfiniBand NICs (one per GPU)
- Dual AMD EPYC CPUs, 2 TB system memory

### DGX SuperPOD

Cluster of DGX nodes for training at scale:
- 32+ DGX H100 nodes (256+ GPUs)
- Non-blocking InfiniBand fabric
- Shared parallel filesystem (Lustre, GPFS)
- Optimized for 3D parallelism

## How Hardware Topology Maps to Parallelism

The bandwidth hierarchy dictates where each parallelism strategy should operate:

```
Within GPU:     ~3 TB/s (HBM bandwidth)     → Computation
NVLink:         900 GB/s (H100)              → Tensor Parallelism (TP)
NVSwitch:       Full mesh at NVLink speed     → Expert Parallelism (EP)
InfiniBand:     400 Gbps (~50 GB/s)          → Pipeline Parallelism (PP)
Cross-rack:     Multiple IB links             → Data Parallelism (DP)
```

**Example: Training LLaMA 70B on 64 GPUs (8 DGX H100 nodes)**

```
TP=8 (within each DGX node, over NVLink)
PP=4 (across 4 nodes per pipeline, over InfiniBand)
DP=2 (2 pipeline replicas, gradient sync over InfiniBand)
```

This configuration:
- Keeps high-frequency TP communication on NVLink (fastest)
- Uses PP across nodes (lower communication frequency)
- Syncs gradients across DP replicas (infrequent, large transfers)

## Best Practices

- **Match parallelism to topology**: TP within NVLink domain, PP across IB
- **Overlap communication with compute**: NCCL supports async operations
- **Profile with Nsight Systems**: Identify communication bottlenecks
- **Use NCCL topology awareness**: `NCCL_TOPO_FILE` for optimal algorithm selection
- **Consider memory hierarchy**: Activation checkpointing trades compute for HBM capacity

See [distributed training](model-parallelization.md) for parallelism strategy details and [NCCL](pytorch-distributed-training.md) for communication primitives.

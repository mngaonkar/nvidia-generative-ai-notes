# Nvidia Nsight
NVIDIA Nsight is a family of developer tools used to profile, debug, and optimize GPU applications running on the CUDA platform.
It is widely used for:
Deep learning optimization
HPC workloads
CUDA kernel debugging
GPU performance analysis
Nsight helps engineers understand what is happening inside the GPU during execution.

## Nsight Tools
Nsight is actually a suite of tools, not a single program.

| Tool                            | Purpose                          |
| ------------------------------- | -------------------------------- |
| Nsight Systems                  | System-level performance tracing |
| Nsight Compute                  | CUDA kernel profiling            |
| Nsight Graphics                 | Graphics debugging               |
| Nsight Eclipse / VS integration | CUDA debugging in IDE            |

## Nsight Systems (System-level profiler)
Nsight Systems shows how the CPU, GPU, and other processes interact over time.
It answers questions like:
- Are GPUs idle?
- Are kernels waiting on CPU launches?
- Is communication overlapping with computation?
- Are there synchronization bottlenecks?

You can visualize:
- CUDA kernel launches
- Memory copies
- NCCL communication
- CPU threads

Example command:
```bash
nsys profile python train.py
```

## Nsight Compute (Kernel profiler)
Nsight Compute analyzes individual GPU kernels in detail.
It answers questions like:
- Is the kernel using tensor cores?
- What is GPU occupancy?
- Are memory accesses coalesced?
- Is the kernel compute-bound or memory-bound?

Metrics include:
| Metric                  | Meaning                     |
| ----------------------- | --------------------------- |
| SM occupancy            | How busy GPU cores are      |
| Tensor core utilization | Matrix unit usage           |
| Warp efficiency         | Thread execution efficiency |
| Memory throughput       | Global memory bandwidth     |
| Shared memory usage     | On-chip memory efficiency   |

Example command:
```bash
ncu --set full python train.py
```

Example output:
```
Kernel: attention_forward
SM Occupancy: 72%
Tensor Core Utilization: 90%
Memory Bandwidth: 63%
```

## Nsight in LLM Training
In distributed training (Megatron, DeepSpeed, etc.), Nsight helps analyze:
- NCCL communication overhead
- Pipeline parallelism stalls
- Tensor core utilization
- Kernel fusion efficiency


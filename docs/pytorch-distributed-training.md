# PyTorch Distributed Training

## Sample Example
Here is an example of running distributed model training on 2 nodes (with 2 GPU each) in pure PyTorch way (No Nemo/Megatron involved)

```
torchrun \
  --nnodes=2 --nproc_per_node=2 \
  --node_rank=0 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=<NODE0_IP>:29500 \
  train_byt5_ddp.py
  ```
- `--nnodes=2`: Total number of nodes in the cluster
- `--node_rank`: Node identification 0 for node 1 and 1 for node 2
- `--nproc_per_node=2` tells each node: “spawn 2 worker processes on this node (usually 1 per GPU).”
- `--rdzv_endpoint=<NODE0_IP>:29500` tells every node: “meet at the same coordinator address.”
- `--rdzv_backend=c10d` tells them which rendezvous mechanism to use. c10d is PyTorch’s distributed TCP-based co-ordination backend layer.
- `train_byt5_ddp.py` is the same entrypoint on all nodes.

## How co-ordination works
1. Each node runs the same command with different `--node_rank`.
2. On node_rank=0, it starts a small TCP store (TCPStore) server at `<NODE0_IP>:29500`. It is just a lightweight key-value store for coordination
3. Other nodes connect to it.
4. They exchange:
    - World size
    - Rank assignments
    - NCCL connection info
    - Environment setup data
5. Real GPU communication happens via NCCL
6. NCCL opens its own high-performance channels
7. That may use:
    - IB (InfiniBand)
    - NVLink
    - TCP fallback
  
## NCCL Magic (Nvidia Common Communication Library)
NCCL is a high-performance communication library optimized for NVIDIA GPUs.

It handles:
- AllReduce (gradients)
- Broadcast
- AllGather
- ReduceScatter

Over:
- NVLink
- PCIe
- InfiniBand
- Ethernet (TCP fallback)

## How PyTorch Knows About NCCL
PyTorch is compiled with NCCL support.

You can verify.
```Python
import torch
print(torch.cuda.nccl.version())
```

Here is how NCCL gets activated.
```Python
torch.distributed.init_process_group(
    backend="nccl"
)
```
PyTorch does:
- Load NCCL backend
- Create NCCL communicators
- Exchange NCCL unique IDs via rendezvous (TCPStore)
- Establish GPU-to-GPU communication

### Flow
1. torchrun → creates processes
2. Rendezvous → TCPStore exchanges info
3. Rank 0 generates NCCL unique ID
4. ID shared to all nodes
5. NCCL forms communicators
6. Gradients flow via:
    - NVLink (same node)
    - InfiniBand / TCP (cross-node)
  
After that, PyTorch doesn’t micromanage NCCL.

### Note
- Megatron does NOT configure NCCL directly.
- It uses PyTorch Distributed, and PyTorch initializes NCCL.


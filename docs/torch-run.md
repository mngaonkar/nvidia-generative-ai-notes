# Torch Run
torchrun is the official distributed training launcher from PyTorch. It is framework-agnostic. NeMo internally relies on it for multi-GPU and multi-node training.

What it does
- Spawns multiple processes (1 per GPU)
- Sets up:
    - RANK
    - WORLD_SIZE
    - LOCAL_RANK
- Initializes Distributed Data Parallel (DDP)

## Example (NeMo training via torchrun)
```bash
torchrun \
  --nproc_per_node=8 \
  --nnodes=1 \
  --master_port=29500 \
  train.py \
  trainer.devices=8 \
  trainer.num_nodes=1
```
Think of torchrun as low-level distributed process manager.

# Nemo Run - Nemo 2.0 Orchestrator
nemo run is part of NeMo 2.0’s new execution system.
It’s a higher-level orchestration CLI built by NVIDIA.

It doesn’t just launch processes — it manages:
- Config management (YAML-based)
- Data preparation utilities
- Distributed strategy
- Config resolution
- Cluster backend
- Experiment lifecycle
- Slurm/K8s integration
- Integrated with NeMo models and training pipelines
- Reproducibility
- Logging
- Metrics tracking
- Checkpointing

NeMo-Run simplifies the user experience for training NeMo models, while torchrun provides the underlying distributed execution capabilities.

## Example
```bash
nemo run \
  model=llama3_8b \
  trainer.devices=8 \
  trainer.num_nodes=1 \
  exp_manager.name=my_experiment
```
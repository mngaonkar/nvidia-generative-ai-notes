# Checkpoint Translation
Checkpoint translation (shard/unshard checkpoints) refers to converting model checkpoints between different distributed training formats:

1. Sharding: Converting a single, consolidated model checkpoint into multiple shards (pieces) that can be distributed across multiple GPUs/devices. Each shard contains a portion of the model weights.

Example: A 7B parameter model checkpoint split into 8 shards for 8-GPU training with tensor parallelism.

2. Unsharding: The reverse process—combining multiple sharded checkpoints back into a single consolidated checkpoint.

Useful when you want to save a distributed model as a standard HuggingFace model or deploy it on a single device.

3. Format Translation: Megatron Bridge specifically handles converting between:

Megatron format: Sharded checkpoints optimized for distributed training with tensor/pipeline parallelism
HuggingFace format: Standard consolidated checkpoint format used by the HF ecosystem

# Why this matters:

- Training at scale: When using tensor or pipeline parallelism, model weights are split across GPUs. You need sharded checkpoints for this.
- Interoperability: Enables seamless conversion between Megatron and HuggingFace formats, allowing you to:
Train with Megatron-Core (distributed, optimized)
Deploy with HuggingFace (standard ecosystem)
Resume training from different formats
- Checkpointing: Efficiently save and load large models during distributed training without consolidating everything into a single file (which would be memory-intensive).
  
In essence, Megatron Bridge acts as a bridge that handles the complexity of translating checkpoints between distributed and consolidated formats, abstracting away the details from the user.
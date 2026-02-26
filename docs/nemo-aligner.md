# NeMo Aligner

NeMo Aligner is NVIDIA's scalable toolkit for aligning large language models with human preferences. It integrates with Megatron-Core for distributed training and supports multiple alignment algorithms.

## Why Alignment?

Pretrained and supervised fine-tuned (SFT) models generate fluent text but may produce harmful, biased, or unhelpful responses. Alignment tunes model behavior to be helpful, harmless, and honest using human feedback signals.

The typical post-training pipeline:

```
Pretrained Model → SFT (demonstrations) → Alignment (preferences) → Production
```

## Supported Alignment Methods

### RLHF (Reinforcement Learning from Human Feedback)

The classic three-stage alignment pipeline:

1. **Supervised Fine-Tuning**: Train on high-quality instruction-response demonstrations
2. **Reward Model Training**: Train a classifier on human preference pairs (chosen vs. rejected)
3. **RL Optimization**: Use PPO to maximize reward while constraining drift from the SFT model

**Reward model loss** (Bradley-Terry preference model):

```
L_RM = -E[log σ(r(x, y_chosen) - r(x, y_rejected))]
```

**PPO objective** with KL penalty:

```
L_PPO = -E[reward(x, y)] + β · KL(π_θ || π_SFT)
```

The KL term prevents the policy from deviating too far from the SFT model, avoiding reward hacking.

### DPO (Direct Preference Optimization)

Eliminates the reward model by directly optimizing the policy on preference pairs:

```
L_DPO = -E[log σ(β · (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]
```

Where `y_w` = chosen completion, `y_l` = rejected completion, `π_ref` = reference (SFT) model.

### SteerLM

Attribute-conditioned generation: annotate training data with quality attributes (helpfulness, correctness, coherence) and condition generation on desired attribute values at inference time. Offers finer-grained control than binary preference methods.

### Method Comparison

| Method | Reward Model | Training Stability | Compute Cost | Implementation Complexity |
|--------|-------------|-------------------|-------------|--------------------------|
| RLHF (PPO) | Required | Lower (RL instability) | High (3 models in memory) | High |
| DPO | Not needed | Higher | Medium (2 models) | Medium |
| SteerLM | Not needed | High | Medium | Medium |

## NeMo Aligner Features

- **Distributed training**: Full TP, PP, and DP support via Megatron-Core backend
- **Efficient memory**: Gradient checkpointing, BF16/FP8 mixed precision
- **Flexible data**: JSON/JSONL preference datasets with chosen/rejected pairs
- **Custom reward models**: Train reward models on domain-specific preference data
- **Evaluation**: Built-in safety and quality benchmarks during alignment

## Typical Workflow

1. **Prepare preference dataset**: Format as `{"prompt": ..., "chosen": ..., "rejected": ...}`
2. **Train reward model** (RLHF only): Fine-tune a model to predict human preferences
3. **Run alignment**: Select DPO, PPO, or SteerLM via NeMo config
4. **Evaluate**: Run safety benchmarks (ToxiGen, TruthfulQA) and quality metrics

## Integration with NeMo Stack

- Works with NeMo 2.0 model definitions and checkpoints
- Orchestrated via [NeMo-Run](nemo-run.md) for local, Slurm, or Kubernetes execution
- Exports to standard checkpoint formats for [Megatron Bridge](megatron-bridge.md) conversion
- Compatible with [NeMo Guardrails](nemo-guardrails.md) for runtime safety enforcement

## Data Format

```json
{
  "prompt": "Explain quantum computing in simple terms.",
  "chosen": "Quantum computing uses quantum bits (qubits) that can exist in multiple states simultaneously...",
  "rejected": "Quantum computing is a type of computing that uses quantum mechanics. It is very complicated..."
}
```

## Common Pitfalls

- **Reward hacking**: Model exploits reward model weaknesses — mitigate with KL constraints
- **Alignment tax**: Over-alignment can degrade general capabilities — evaluate on broad benchmarks
- **Data quality**: Preference data consistency directly determines alignment quality
- **Mode collapse**: Model generates only a narrow range of safe responses — balance safety with helpfulness

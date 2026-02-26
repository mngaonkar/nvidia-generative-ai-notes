# Alignment: RLHF, DPO, and Beyond

Alignment is the process of tuning language models to follow human preferences — producing helpful, harmless, and honest outputs. This doc covers the theory and mathematics behind the major alignment methods.

## The Alignment Problem

Pretrained LLMs optimize for next-token prediction, not for following instructions or being safe. A model that maximizes `P(next_token | context)` may generate toxic content, hallucinate confidently, or refuse to help. Alignment bridges the gap between "predicts text well" and "behaves as intended."

## Stage 0: Supervised Fine-Tuning (SFT)

Before alignment, models are typically fine-tuned on demonstration data:

```
Input: instruction-response pairs from human annotators
Output: model that follows instructions in the demonstrated format
```

SFT provides the behavioral foundation — the model learns what good responses look like. Alignment then refines this using preference signals: which of two responses is better.

## RLHF: Reinforcement Learning from Human Feedback

### Stage 1: Reward Model Training

Collect human preferences: given a prompt and two completions, which is better?

The reward model `r(x, y)` is trained to predict these preferences using the Bradley-Terry model:

```
P(y_1 > y_2 | x) = σ(r(x, y_1) - r(x, y_2))
```

Loss function:

```
L_RM = -E_{(x, y_w, y_l) ~ D}[log σ(r(x, y_w) - r(x, y_l))]
```

The reward model is typically the same architecture as the LLM but with a scalar output head instead of a vocabulary head.

### Stage 2: RL Optimization (PPO)

Maximize expected reward while staying close to the SFT policy:

```
max_θ E_{x~D, y~π_θ}[r(x, y)] - β · KL(π_θ || π_SFT)
```

PPO (Proximal Policy Optimization) implements this with clipped surrogate objectives:

```
L_PPO = E[min(ratio · A, clip(ratio, 1-ε, 1+ε) · A)]
```

Where `ratio = π_θ(a|s) / π_old(a|s)` and `A` is the advantage estimate.

**Components in memory during RLHF training:**
1. Policy model (being trained)
2. Reference model (frozen SFT checkpoint)
3. Reward model (frozen)
4. Value model (critic, being trained)

This is why RLHF is memory-intensive: four models must coexist, often requiring multi-GPU setups.

### RLHF Challenges

- **Reward hacking**: Policy exploits reward model weaknesses (e.g., generating longer responses because the RM associates length with quality)
- **Training instability**: RL optimization is inherently noisy
- **Compute cost**: 4 models in memory, iterative rollout + training loop
- **Reward model quality ceiling**: Alignment quality bounded by RM accuracy

## DPO: Direct Preference Optimization

DPO reformulates the RLHF objective to eliminate the reward model entirely. Key insight: the optimal policy under the KL-constrained reward maximization objective has a closed-form relationship with the reward:

```
r(x, y) = β · log(π_θ(y|x) / π_ref(y|x)) + C(x)
```

Substituting back into the Bradley-Terry preference model:

```
L_DPO = -E_{(x, y_w, y_l)}[log σ(β · (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]
```

**Advantages over RLHF:**
- No reward model to train or maintain
- No RL optimization (standard supervised loss)
- Only 2 models in memory (policy + frozen reference)
- More stable training dynamics

**Disadvantages:**
- Cannot reuse reward model for other tasks
- Less flexible than reward-based approaches
- Sensitive to β hyperparameter

## Other Alignment Methods

### IPO (Identity Preference Optimization)

Addresses DPO's overfitting to preference data by adding a regularization term:

```
L_IPO = E[(log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x) - 1/(2β))²]
```

### KTO (Kahneman-Tversky Optimization)

Works with unpaired binary feedback (thumbs up/down) instead of pairwise preferences. Based on prospect theory — humans are more sensitive to losses than gains.

### ORPO (Odds Ratio Preference Optimization)

Combines SFT and alignment into a single training stage using odds ratio of preferred vs. dispreferred responses.

## Method Comparison

| Method | Training Data | Models in Memory | RL Required | Stability | Compute |
|--------|--------------|-----------------|-------------|-----------|---------|
| RLHF (PPO) | Pairwise preferences | 4 | Yes | Lower | Very High |
| DPO | Pairwise preferences | 2 | No | High | Medium |
| IPO | Pairwise preferences | 2 | No | High | Medium |
| KTO | Binary feedback | 2 | No | High | Medium |
| ORPO | Pairwise preferences | 1 | No | High | Low |

## Preference Data Quality

Alignment quality is bounded by preference data quality. Best practices:

- **Consistency**: Annotators should agree on preferences (inter-annotator agreement > 70%)
- **Diversity**: Cover a wide range of topics, instructions, and difficulty levels
- **Calibration**: Include examples where both responses are good but one is clearly better
- **Safety focus**: Specifically include harmful-prompt rejection examples

## Evaluation After Alignment

- **MT-Bench**: Multi-turn conversation quality (GPT-4 as judge)
- **AlpacaEval**: Instruction-following quality
- **TruthfulQA**: Factual accuracy and honesty
- **ToxiGen**: Toxicity and bias
- **General benchmarks**: MMLU, HumanEval to verify no capability regression (alignment tax)

## Common Failure Modes

1. **Mode collapse**: Model produces safe but generic responses for everything
2. **Sycophancy**: Model agrees with user regardless of correctness
3. **Refusal over-correction**: Model refuses benign requests due to over-cautious safety training
4. **Reward hacking**: Model optimizes for proxy metrics rather than true helpfulness
5. **Distribution shift**: Alignment data doesn't cover deployment distribution

See [NeMo Aligner](nemo-aligner.md) for implementation details in NVIDIA's stack.

# Quantization

Quantization reduces the numerical precision of model weights and activations — from FP32/FP16 to INT8, INT4, or FP8 — to decrease memory footprint, increase throughput, and reduce energy consumption with minimal accuracy loss.

## Why Quantize?

A 70B parameter model in FP16 requires ~140 GB of memory for weights alone. Quantization changes the equation:

| Precision | Bits | 70B Model Size | Memory Savings |
|-----------|------|---------------|----------------|
| FP32 | 32 | 280 GB | Baseline |
| FP16/BF16 | 16 | 140 GB | 2x |
| FP8 | 8 | 70 GB | 4x |
| INT8 | 8 | 70 GB | 4x |
| INT4 | 4 | 35 GB | 8x |

Beyond memory, lower precision enables:
- Faster matrix multiplications on Tensor Cores
- Higher batch sizes (more sequences in KV cache)
- Lower inference latency
- Reduced energy per token

## Quantization Fundamentals

### The Quantization Function

```
Q(x) = round(x / scale) + zero_point
x̂ = (Q(x) - zero_point) × scale
```

**Scale** maps the floating-point range to the integer range. **Zero point** handles asymmetric distributions.

### Symmetric vs. Asymmetric

- **Symmetric**: zero_point = 0, range is [-max, +max]. Simpler, used for weights.
- **Asymmetric**: zero_point ≠ 0, range is [min, max]. Better for activations with non-zero mean.

### Granularity

- **Per-tensor**: One scale for the entire tensor. Fastest but lowest accuracy.
- **Per-channel**: One scale per output channel. Standard for weights.
- **Per-group**: One scale per group of values (e.g., groups of 128). Best accuracy, used in GPTQ/AWQ.
- **Per-token**: One scale per token. Used for activations in SmoothQuant.

## Post-Training Quantization (PTQ)

Quantize a trained model without retraining. Requires a small calibration dataset (128-512 samples) to estimate activation ranges.

**Workflow:**
1. Run calibration data through the model
2. Collect activation statistics (min, max, or percentiles)
3. Compute scales and zero points
4. Replace FP16 operations with quantized versions

**Pros:** Fast (minutes to hours), no training infrastructure needed
**Cons:** Accuracy may degrade, especially at INT4

## Quantization-Aware Training (QAT)

Simulate quantization during training so the model learns to be robust to reduced precision.

**Mechanism:** Insert fake quantization operators in the forward pass:
```
Forward: x → quantize → dequantize → next_layer  (simulates quantization error)
Backward: Straight-Through Estimator (STE) — gradients pass through unchanged
```

**Pros:** Best accuracy retention, even at INT4
**Cons:** Requires full training run, expensive

## SmoothQuant

**Problem:** Activations have outlier channels with large magnitudes, making them hard to quantize. Weights are smooth and easy to quantize.

**Solution:** Migrate quantization difficulty from activations to weights:

```
Y = X · W = (X · diag(s)^-1) · (diag(s) · W) = X̂ · Ŵ
```

Where `s` is a per-channel smoothing factor: `s_j = max(|X_j|)^α / max(|W_j|)^(1-α)`, with α ∈ [0, 1].

After smoothing, both activations and weights are within quantizable ranges. Typical α = 0.5.

**Result:** W8A8 (8-bit weights, 8-bit activations) with near-lossless accuracy.

## AWQ (Activation-Aware Weight Quantization)

**Key insight:** Not all weight channels are equally important. Channels corresponding to large activation magnitudes have disproportionate impact on output quality.

**Method:**
1. Identify salient weight channels based on activation magnitudes
2. Apply per-channel scaling to protect these channels before quantization
3. Search for optimal scaling factors that minimize quantization error

```
s* = argmin_s ||Q(W · diag(s)) · diag(s)^-1 · X - W · X||
```

**Result:** INT4 weight quantization with minimal perplexity degradation. Works without retraining.

## GPTQ

Based on Optimal Brain Compression (OBC), GPTQ quantizes weights layer by layer using second-order information:

1. Compute approximate Hessian: `H = 2X^TX` (from calibration data)
2. For each weight column, find optimal quantized value and compensate remaining weights:

```
w_q = argmin_q (w - q)² / H_qq^-1
δ_remaining = -(w - w_q) / H_qq · H_q,:
```

3. Update remaining unquantized weights to compensate for quantization error

**Result:** INT4/INT3 with very low accuracy loss. Slower calibration than AWQ but sometimes better accuracy.

## FP8 with Transformer Engine

H100 GPUs support native FP8 (E4M3 for forward, E5M2 for backward) via Transformer Engine:

- **Dynamic scaling**: Automatically adjusts per-tensor scale factors each iteration
- **Amax history**: Tracks maximum absolute values over a window for stable scaling
- **Delayed scaling**: Uses amax from previous iterations to set current scale

```
FP8 forward pass:
  1. Track amax of input activation
  2. Compute scale = FP8_MAX / amax
  3. Quantize input to FP8: x_fp8 = cast_to_fp8(x * scale)
  4. Matrix multiply in FP8
  5. Output in FP16/BF16
```

**Integration:** Megatron-Core enables FP8 with `--fp8-format hybrid --transformer-impl transformer_engine`.

**Benefit:** ~2x throughput over BF16 on H100 with minimal accuracy loss for training.

## NVIDIA modelopt Toolkit

NVIDIA's unified quantization toolkit (formerly TensorRT Model Optimizer):

- **PTQ**: Calibration-based quantization for INT8, INT4, FP8
- **QAT**: Quantization-aware fine-tuning with LoRA compatibility
- **Sparsity**: 2:4 structured sparsity for additional speedup
- **Distillation**: Combine quantization with knowledge distillation

```python
import modelopt.torch.quantization as mtq

# PTQ with INT4 AWQ
model = mtq.quantize(model, mtq.INT4_AWQ_CFG, forward_loop=calibrate)

# Export to TensorRT-LLM
mtq.export(model, output_dir="quantized_checkpoint/")
```

## Choosing a Quantization Method

| Use Case | Recommended Method | Why |
|----------|-------------------|-----|
| Training speedup on H100 | FP8 (Transformer Engine) | Native hardware support, minimal accuracy loss |
| Inference INT8 | SmoothQuant | Best W8A8 accuracy, well-supported |
| Inference INT4 (quality focus) | AWQ or GPTQ | Near-lossless at 4-bit |
| Inference INT4 (speed focus) | AWQ | Faster calibration, good TRT-LLM support |
| Maximum accuracy at low precision | QAT | Retraining compensates quantization error |
| Quick deployment | PTQ (basic) | Fastest, good enough for many workloads |

## Quantization in the NVIDIA Stack

```
Training (FP8 via Transformer Engine in Megatron-Core)
    ↓
Post-Training Quantization (modelopt)
    ↓
Engine Build (TensorRT-LLM with quantized weights)
    ↓
Deployment (Triton / NIM)
```

See [TensorRT-LLM](tensorrt-llm.md) for inference engine compilation with quantized models.

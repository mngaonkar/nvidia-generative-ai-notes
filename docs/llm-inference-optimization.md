# LLM Inference Optimization

## Optimization Techniques

### 1. Continuous Batching

Don't wait for all sequences to finish—add new requests as old ones complete:
```
Time 0: [Seq A prefill] [Seq B prefill] [Seq C prefill]
Time 1: [Seq A decode]  [Seq B decode]  [Seq C decode]
Time 2: [Seq A decode]  [Seq B decode]  [Seq C done→Seq D prefill]
Time 3: [Seq A decode]  [Seq B done]    [Seq D decode]
...
```

### 2. Paged Attention (vLLM)

Manage KV cache like virtual memory—non-contiguous blocks:
```
Instead of: [Seq 1 KV][Seq 2 KV][Seq 3 KV] (fragmented)
Use:        Block table mapping virtual → physical pages
            Eliminates memory fragmentation
            Enables memory sharing for beam search
```

### 3. Speculative Decoding

Use small model to draft, large model to verify:
```
Draft model: Generates 5 tokens quickly (T1, T2, T3, T4, T5)
Large model: Verifies all 5 in ONE forward pass
Result:      Accept T1, T2, T3 (verified), reject T4, T5
             3 tokens for cost of ~1.5 forward passes
```

### 4. KV Cache Quantization

Compress the cache to reduce memory:
```
fp16 KV cache: 2 bytes per element
int8 KV cache: 1 byte per element → 2x more sequences!
int4 KV cache: 0.5 bytes → 4x more sequences!
```
Note: KV cache quantization is separate from model weight quantization.

### 5. Multi-Query / Grouped-Query Attention

Reduce KV heads to save cache memory:
```
MHA:  64 query heads, 64 KV heads → 100% cache
GQA:  64 query heads, 8 KV heads  → 12.5% cache
MQA:  64 query heads, 1 KV head   → 1.6% cache
```

### 6. Flash Attention
Rearrange computations to minimize memory access:
```
Standard: Compute full attention matrix (seq_len × seq_len) → O(n²) memory
Flash:   Compute in blocks, keep only necessary parts in memory → O(n) memory
```
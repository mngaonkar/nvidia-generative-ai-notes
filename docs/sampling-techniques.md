# Sampling Techniques for Decoder-Based Models
When a decoder model generates text, it outputs a probability distribution over the entire vocabulary for the next token. Sampling techniques determine how we select from this distribution.

The Core Problem
At each generation step, the model outputs logits that become probabilities via softmax:
```
P(token_i) = exp(logit_i) / Σ exp(logit_j)
```
The question: How do we pick the next token from these probabilities?

## Greedy Decoding
The simplest approach: always pick the highest probability token.

```
next_token = argmax(probabilities)
```
Pros:

- Deterministic and fast
- Good for tasks with clear "correct" answers (translation, summarization)

Cons:

- Often produces repetitive, boring text
- Can get stuck in loops ("I think that I think that I think...")
- Misses high-quality sequences that start with lower-probability tokens

## Temperature Sampling
Temperature (T) reshapes the probability distribution before sampling:

```
adjusted_logits = logits / temperature
probabilities = softmax(adjusted_logits)
next_token = random_sample(probabilities)
```

**Effect of temperature:**

| Temperature | Effect | Use Case |
|-------------|--------|----------|
| T → 0 | Approaches greedy (peaky distribution) | Factual, deterministic |
| T = 1.0 | Original distribution | Balanced |
| T > 1.0 | Flatter distribution (more random) | Creative writing |

**Example distribution for "The cat sat on the ___":**
```
T=0.3 (sharp):     mat: 0.85, floor: 0.10, couch: 0.04, moon: 0.01
T=1.0 (normal):    mat: 0.40, floor: 0.25, couch: 0.20, moon: 0.15
T=2.0 (flat):      mat: 0.28, floor: 0.26, couch: 0.24, moon: 0.22
```
## Top-k Sampling
Only consider the k most probable tokens, then renormalize and sample:

```
top_k_logits, top_k_indices = topk(logits, k=50)
probabilities = softmax(top_k_logits)
next_token = random_sample_from(top_k_indices, probabilities)
```

**Pros:**
- Prevents sampling extremely unlikely tokens
- Simple to implement and understand

**Cons:**
- Fixed k doesn't adapt to distribution shape
- k=50 might be too many for a peaked distribution, too few for a flat one

**Example:**
```
k=3: Only sample from {mat, floor, couch}, ignore "moon", "banana", etc.
```

## Top-p (Nucleus) Sampling
Sample from the smallest set of tokens whose cumulative probability exceeds p:

```
sorted_probs, sorted_indices = sort(probabilities, descending=True)
cumulative_probs = cumsum(sorted_probs)
cutoff_index = first_index_where(cumulative_probs > p)
nucleus = sorted_indices[:cutoff_index]
next_token = random_sample_from(nucleus)
```

**Adaptive behavior:**
- Peaked distribution (model confident): nucleus is small (few tokens)
- Flat distribution (model uncertain): nucleus is large (many tokens)

**Example with p=0.9:**
```
Confident: "The capital of France is ___"
  → Paris: 0.95 → nucleus = {Paris} (just 1 token!)

Uncertain: "I enjoy eating ___"  
  → pizza: 0.15, pasta: 0.12, sushi: 0.10, ... 
  → nucleus = {pizza, pasta, sushi, salad, ...} (many tokens)
  ```

  ##  Beam Search
  Maintain multiple candidate sequences (beams) and expand the most promising:

  ```
  beams = [("", 0.0)]  # (sequence, log_probability)

for step in range(max_length):
    all_candidates = []
    for seq, score in beams:
        probs = model(seq)
        for token, prob in top_tokens(probs, k=beam_width):
            all_candidates.append((seq + token, score + log(prob)))
    
    beams = top_k(all_candidates, k=beam_width)  # Keep best beams
  ```

Pros:

- Finds higher probability sequences than greedy
- Good for tasks with clear targets (translation, speech recognition)

Cons:

- More computationally expensive
- Tends to produce generic, safe outputs
- Often worse than sampling for open-ended generation

## Combined Strategies
In practice, multiple techniques are combined:

```
def generate_token(logits, temperature=0.8, top_k=50, top_p=0.9):
    # Step 1: Apply temperature
    logits = logits / temperature
    
    # Step 2: Top-k filtering
    top_k_logits, top_k_indices = topk(logits, k=top_k)
    
    # Step 3: Convert to probabilities
    probs = softmax(top_k_logits)
    
    # Step 4: Top-p (nucleus) filtering
    sorted_probs, sorted_idx = sort(probs, descending=True)
    cumsum_probs = cumsum(sorted_probs)
    mask = cumsum_probs <= top_p
    mask[0] = True  # Always keep at least one token
    
    filtered_probs = sorted_probs * mask
    filtered_probs = filtered_probs / sum(filtered_probs)  # Renormalize
    
    # Step 5: Sample
    return random_sample(filtered_probs)
```

## Repetition Penalty
Reduce probability of tokens that already appeared:

```
for token_id in generated_sequence:
    if logits[token_id] > 0:
        logits[token_id] /= repetition_penalty  # penalty > 1.0
    else:
        logits[token_id] *= repetition_penalty
```

## Frequency & Presence Penalties (OpenAI-style)

```
# Presence penalty: flat penalty if token appeared at all
# Frequency penalty: scales with how often token appeared

for token_id, count in token_counts.items():
    logits[token_id] -= presence_penalty  # Fixed penalty
    logits[token_id] -= frequency_penalty * count  # Scaled penalty
```

## Min-p Sampling (Newer Technique)
Only keep tokens with probability ≥ (min_p × max_probability):
```
max_prob = max(probabilities)
threshold = min_p * max_prob
mask = probabilities >= threshold
filtered_probs = probabilities * mask
```

Advantage: Naturally adapts to distribution shape like top-p, but with a simpler threshold logic.

## Typical Sampling
Sample tokens based on how "typical" they are (information-theoretic approach):

```
# Calculate entropy of distribution
entropy = -sum(p * log(p) for p in probabilities)

# Calculate "typicality" of each token
# (how close its information content is to expected entropy)
typicality = abs(-log(p) - entropy)

# Keep tokens within threshold of typical
typical_mask = typicality < threshold
```

Intuition: Avoids both very predictable tokens (boring) and very surprising tokens (incoherent)

## Contrastive Decoding
Compare expert model with amateur model, prefer tokens the expert likes more:
```
expert_logits = expert_model(context)
amateur_logits = amateur_model(context)

# Amplify differences
adjusted_logits = expert_logits - alpha * amateur_logits
```

Use case: Reduces common failure modes, improves factuality.

## Speculative Decoding (Efficiency)
Use a small "draft" model to propose multiple tokens, verify with large model:

```
# Draft model proposes k tokens quickly
draft_tokens = draft_model.generate(k_tokens)

# Large model verifies in parallel (single forward pass)
verified = large_model.verify(draft_tokens)

# Accept verified prefix, reject rest
```

Benefit: 2-3x speedup without changing output distribution.



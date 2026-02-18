# Sampling

At each step the model produces a probability distribution over the next token. Sampling means: randomly pick the next token from a filtered version of that distribution, instead of always picking the highest-probability token.

## Temperature (temperature)

What it does: reshapes the probability distribution before sampling.

Given logits z, temperature T scales them:
```
z′=z/T
```
Then softmax is applied to get probabilities.
- T = 1.0: no change (baseline distribution).
- T < 1.0: distribution becomes sharper → more deterministic (high-prob tokens more likely).
- T > 1.0: distribution becomes flatter → more randomness (low-prob tokens more likely).

## Top-k (top_k)
What it does: keeps only the k most probable tokens at each step; sets all others to probability 0; then samples from the remaining k.

Example: if top_k=50, you sample from the 50 most likely next tokens.
- top_k = 50: common default for sampling.
- top_k = 1: basically greedy sampling (only the top token can be picked).
- top_k = 0 in HuggingFace generation means: don’t apply top-k filtering (i.e., keep all tokens). In practice, when you also have do_sample=False, it’s irrelevant.

## Top-p / nucleus sampling (top_p)
What it does: keeps the smallest set of tokens whose cumulative probability is at least p, then samples from that set.

So the number of allowed tokens changes depending on how confident the model is.
- If the model is very confident, the nucleus might contain only a few tokens.
- If it’s uncertain, the nucleus might contain many tokens.

Typical values:
- top_p = 0.9 or 0.95: common for controlled sampling.
- top_p = 1.0: keeps tokens until cumulative probability reaches 1.0 → effectively keeps everything → no nucleus filtering.
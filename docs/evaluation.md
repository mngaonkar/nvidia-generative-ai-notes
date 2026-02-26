# Model Evaluation

Systematic evaluation is essential at every stage: after pretraining (does the model understand language?), after fine-tuning (does it follow instructions?), after alignment (is it safe and helpful?), and after quantization (did we lose accuracy?).

## Automatic Metrics

### Perplexity

Measures how well a model predicts the next token. Lower is better.

```
PPL = exp(-1/N Σ_{i=1}^{N} log P(x_i | x_{<i}))
```

- **Use case**: Language modeling quality, comparing model versions
- **Benchmark datasets**: WikiText-2, The Pile validation, C4
- **Limitations**: Doesn't measure instruction-following, safety, or reasoning

### BLEU (Bilingual Evaluation Understudy)

Measures n-gram overlap between generated text and reference:

```
BLEU = BP · exp(Σ_{n=1}^{4} w_n · log p_n)
```

Where `p_n` = modified n-gram precision, `BP` = brevity penalty, `w_n` = uniform weights (typically 1/4).

- **Use case**: Machine translation, summarization
- **Range**: 0-100 (higher is better)

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

Measures recall of n-grams from reference text:

- **ROUGE-1**: Unigram overlap
- **ROUGE-2**: Bigram overlap
- **ROUGE-L**: Longest common subsequence
- **Use case**: Summarization quality

### Exact Match and F1

- **Exact Match**: Binary — does the prediction exactly match the reference?
- **F1**: Token-level precision/recall harmonic mean
- **Use case**: Extractive QA (SQuAD), closed-form answers

### Pass@k

For code generation: generate k samples, check how many pass unit tests.

```
Pass@k = 1 - C(n-c, k) / C(n, k)
```

Where `n` = total samples, `c` = correct samples. Unbiased estimator of the probability that at least one of k samples is correct.

- **Use case**: Code generation (HumanEval, MBPP)

## Benchmark Suites

| Benchmark | Measures | Format | Tasks/Subjects |
|-----------|----------|--------|---------------|
| **MMLU** | World knowledge | Multiple choice | 57 subjects (STEM, humanities, social sciences) |
| **HellaSwag** | Commonsense reasoning | Completion selection | 10K scenarios |
| **HumanEval** | Code generation | Executable Python | 164 programming problems |
| **MBPP** | Code generation | Executable Python | 974 problems |
| **GSM8K** | Math reasoning | Free-form numerical | 8.5K grade school math |
| **MATH** | Advanced math | Free-form | Competition-level problems |
| **TruthfulQA** | Factual accuracy | QA pairs | 817 questions designed to elicit falsehoods |
| **ToxiGen** | Safety and bias | Classification | 274K toxic/benign statements |
| **ARC** | Science reasoning | Multiple choice | 7.7K grade-school science questions |
| **WinoGrande** | Coreference resolution | Fill-in-the-blank | 44K pronoun resolution |

## LLM-as-Judge Evaluation

Use a strong LLM (e.g., GPT-4) to evaluate model outputs:

- **MT-Bench**: Multi-turn conversation quality scored 1-10
- **AlpacaEval**: Instruction-following win rate vs. reference model
- **Chatbot Arena**: Crowdsourced pairwise comparisons (Elo rating)

Advantages: captures nuance that automatic metrics miss. Limitations: expensive, potential bias toward certain response styles.

## Evaluation Frameworks

### EleutherAI lm-evaluation-harness

The standard open-source evaluation framework:
- 200+ tasks pre-implemented
- Zero-shot and few-shot evaluation
- HuggingFace model support, extensible to custom formats
- Community-maintained benchmarks

```bash
lm_eval --model hf --model_args pretrained=meta-llama/Llama-3-70B \
    --tasks mmlu,hellaswag,gsm8k \
    --num_fewshot 5 \
    --batch_size 8
```

### HELM (Holistic Evaluation of Language Models)

Stanford's comprehensive evaluation framework:
- Evaluates across 7 metrics: accuracy, calibration, robustness, fairness, bias, toxicity, efficiency
- 42 scenarios covering diverse use cases
- Standardized reporting format

### OpenAI Evals

- Custom evaluation definition format
- Support for LLM-as-judge patterns
- Composable evaluation functions

## Evaluation at Each Stage

| Stage | Key Metrics | What to Watch |
|-------|-------------|---------------|
| Pretraining | Perplexity, MMLU, HellaSwag | Training loss convergence, benchmark trends |
| SFT | MT-Bench, instruction-following accuracy | Overfitting to training format |
| Alignment | TruthfulQA, ToxiGen, MT-Bench | Alignment tax (capability regression) |
| Quantization | Perplexity delta, MMLU delta | <1% degradation on critical metrics |
| Deployment | Latency, throughput, user satisfaction | A/B test against baseline |

## Best Practices

- **Multiple metrics**: No single metric captures model quality — use a diverse evaluation suite
- **Hold-out data**: Never evaluate on training distribution
- **Track across stages**: Compare metrics after each training/optimization stage
- **Domain-specific benchmarks**: Add evaluations specific to your production use case
- **Human evaluation**: Automated metrics supplement but don't replace human judgment for subjective qualities
- **Regression testing**: Maintain a fixed evaluation set to detect quality regressions across model versions

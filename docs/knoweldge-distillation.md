# Knowledge Distillation
Knowledge Distillation (KD) is:
Training a smaller student model to mimic a larger teacher model.

Instead of training on only ground-truth labels, the student also learns from:
- Teacher soft logits
- Probability distributions
- Hidden states (optional)
- Attention maps (advanced cases)

NeMo supports distillation in:
- ASR
- NLP
- LLM training
- Megatron-based models

## Typical Approaches
### Logit Distillation (Most Common)
Loss function
```
Loss = α * CE(student, labels)
     + β * KL(student_logits, teacher_logits)
```
- CE: Cross-entropy with true labels
- KL: KL Divergenece loss

### Hidden State Matching
Student matches:
- Transformer layer outputs
- Embeddings

### Attention Distillation
Attention Distillation

## Nemo/Megatron Core Setup
```
Teacher Model (eval mode) - weigths frozen
       ↓
Generate logits
       ↓
Student Model (train mode)
       ↓
Compute:
  CE loss + KL divergence
       ↓
Backprop on student only
```


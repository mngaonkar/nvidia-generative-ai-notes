# Understanding the NVIDIA Generative AI Stack: From Transformer Fundamentals to Production-Scale Training

Large language models have moved from research curiosity to production infrastructure. Training and deploying them at scale requires more than understanding the transformer architecture — it demands a systems-level grasp of distributed computation, memory management, and tooling. NVIDIA has built a vertically integrated stack for this purpose: from GPU communication primitives (NCCL) through training frameworks (Megatron-Core) to orchestration layers (NeMo 2.0). This article walks through that stack, layer by layer, connecting foundational concepts to the engineering decisions that make billion-parameter models practical.

---

## 1. Transformer Architectures: The Foundation

Every LLM is built on the [transformer](docs/llm-architecture.md), but not every transformer is the same. Three architectural variants dominate, each suited to different workloads:

| Architecture | Structure | Strengths | Examples |
|---|---|---|---|
| **Encoder-only** | Bidirectional self-attention, masked language modeling | Classification, embeddings, semantic search | BERT, RoBERTa |
| **Decoder-only** | Causal (masked) self-attention, autoregressive generation | Open-ended text generation, in-context learning | GPT-4, LLaMA |
| **Encoder-decoder** | Encoder processes input; decoder generates output via cross-attention | Translation, summarization, structured seq2seq tasks | T5, BART |

Decoder-only models dominate current LLM development due to their simpler training objective (next-token prediction) and strong scaling behavior. Encoder-decoder models remain relevant for tasks requiring tight input-output alignment, such as machine translation and retrieval-augmented generation (RAG).

### Attention: The Core Mechanism

The [scaled dot-product attention](docs/attention-mechanism.md) formula sits at the heart of every transformer:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
```

Multi-head attention runs `h` parallel attention heads, each learning different relationship patterns. Recent [attention variants](docs/attention-mechanism-advanced.md) trade expressiveness for efficiency:

- **Multi-Head Attention (MHA)**: Standard — each head has its own Q, K, V projections. Full KV cache.
- **Grouped-Query Attention (GQA)**: Multiple query heads share fewer KV heads. Reduces KV cache to ~12.5% of MHA.
- **Multi-Query Attention (MQA)**: All query heads share a single KV head. KV cache drops to ~1.6% of MHA.

These variants matter enormously at inference time, where KV cache memory is often the binding constraint.

### Tokenization and Embeddings

Before a transformer sees text, it must be converted to numerical representations:

- [**Tokenization**](docs/tokenization.md) splits text into subword units. BPE (Byte Pair Encoding), WordPiece, and [SentencePiece](src/sentence-piece-tokenizer.py) are the dominant algorithms. SentencePiece is particularly useful for multilingual models because it operates on raw text without pre-tokenization assumptions.
- [**Embeddings**](docs/embedding.md) map token IDs to dense vectors. A model's embedding layer combines token embeddings with positional information. Modern LLMs favor Rotary Position Embeddings (RoPE) over fixed sinusoidal or learned positional encodings, as RoPE generalizes better to sequence lengths unseen during training.

---

## 2. Inference: Where Memory Beats Compute

[LLM inference](docs/llm-inference.md) has two distinct phases with fundamentally different performance profiles:

**Prefill phase**: The entire input prompt is processed in parallel. This phase is compute-bound — GPU arithmetic units are the bottleneck. It produces the initial KV cache.

**Decode phase**: Tokens are generated one at a time, each requiring a read of the full KV cache. This phase is memory-bandwidth-bound. For a model like LLaMA 70B with a 4096-token sequence, the KV cache alone consumes ~1.34 GB per sequence.

The KV cache formula:
```
KV Cache = 2 x num_layers x num_heads x seq_len x head_dim x bytes_per_element x batch_size
```

Without caching, every new token would require recomputing attention over the entire sequence — O(n^3) complexity. The KV cache reduces this to O(n^2), but at the cost of memory that grows linearly with sequence length and batch size.

### Optimization Techniques

Six techniques form the modern [inference optimization](docs/llm-inference-optimization.md) toolkit:

1. **Continuous Batching** — New requests enter the batch as completed ones exit, eliminating idle GPU cycles from waiting for the longest sequence to finish.

2. **Paged Attention (vLLM)** — Manages KV cache like virtual memory: non-contiguous physical blocks mapped through a block table. Eliminates memory fragmentation and enables memory sharing across beam search candidates.

3. **Speculative Decoding** — A small draft model proposes multiple tokens; the large model verifies them in a single forward pass. Accepted tokens are "free" — three tokens for the cost of roughly 1.5 forward passes.

4. **KV Cache Quantization** — Compressing cache entries from fp16 to int8 or int4 doubles or quadruples the number of concurrent sequences, independent of model weight quantization.

5. **GQA/MQA** — Architectural choices made at training time that pay dividends at inference by shrinking the KV cache by 8-64x.

6. **Flash Attention** — Restructures the attention computation to process data in blocks, reducing memory usage from O(n^2) to O(n) by avoiding materialization of the full attention matrix.

The key insight: most production inference bottlenecks are memory and scheduling problems, not raw compute problems.

---

## 3. Sampling and Decoding Strategies

How a model selects the next token from its probability distribution significantly affects output quality. Here is an overview of core [sampling techniques](docs/sampling-techniques.md):

| Strategy | Mechanism | Trade-off |
|---|---|---|
| **Greedy** | Always pick the highest-probability token | Deterministic but repetitive |
| **Temperature** | Scale logits before softmax (T<1 = sharper, T>1 = flatter) | Controls randomness |
| **Top-k** | Sample from the k most probable tokens | Fixed candidate set |
| **Top-p (Nucleus)** | Sample from the smallest set whose cumulative probability >= p | Adaptive candidate set |
| **Beam Search** | Maintain multiple candidate sequences | Better for structured outputs |
| **Speculative Decoding** | Draft-then-verify with two models | Faster wall-clock time |

[Advanced techniques](docs/sampling.md) include repetition penalties, min-p sampling, contrastive decoding (comparing expert vs. amateur model outputs), and typical sampling (information-theoretic approach filtering tokens by expected information content).

---

## 4. Reasoning Frameworks: Getting More From Inference

[Prompting strategies](docs/llm-reasoning.md) can dramatically improve LLM reasoning without changing model weights:

- **Chain of Thought (CoT)**: "Let's think step by step" — prompts the model to show intermediate reasoning. Yields 20-50% accuracy gains on complex problems. Works in both [zero-shot and few-shot settings](docs/cot-examples.md).

- **ReAct (Reason + Act)**: Alternates between reasoning (Thought), tool use (Action), and feedback (Observation). Essential for tasks requiring external information retrieval or computation.

- **Tree of Thoughts (ToT)**: Explores multiple reasoning paths via BFS/DFS/beam search, evaluating each branch. Stronger than linear CoT for problems with dead ends or creative solutions.

- **Graph of Thoughts (GoT)**: Extends ToT by allowing thoughts to form arbitrary graphs — merging, referencing, and synthesizing across branches.

- **LLM Compiler**: Plans tool calls as a directed acyclic graph (DAG) upfront and executes them in parallel, rather than sequentially as in ReAct.

- **Language Agent Tree Search (LATS)**: Combines Monte Carlo Tree Search with LLM agents, learning from both successes and failures through strategic exploration.

These frameworks represent a shift toward using more compute at inference time (test-time compute) to extract better answers from existing models.

---

## 5. Data Preparation at Scale: NeMo Curator

Training data quality directly determines model quality. [NeMo Curator](docs/nemo-curator.md) is NVIDIA's GPU-accelerated pipeline for preparing datasets at petabyte scale. Its processing stages include:

1. **Ingestion** — Common Crawl (WARC/WET), HuggingFace datasets, local files. Handles JSON, JSONL, Parquet.
2. **Language identification** — fastText/langdetect classifiers filter by language.
3. **Text extraction** — JusText/Trafilatura strip HTML boilerplate, navigation, ads.
4. **Heuristic filtering** — Word count thresholds, symbol ratios, perplexity scoring via KenLM.
5. **Deduplication** — Exact (MD5/SHA hashing) and fuzzy (MinHash + Locality Sensitive Hashing).
6. **PII redaction** — Regex patterns and NER models detect names, emails, phone numbers, IPs.
7. **Classifier filtering** — ML-based toxicity, NSFW, and quality scoring.

NeMo Curator also supports multimodal pipelines: image aesthetic filtering, video scene detection, and audio transcription quality assessment. Output is clean, shuffled JSONL or Parquet ready for tokenization.

---

## 6. Distributed Training: Parallelism Strategies

Models with hundreds of billions of parameters cannot fit on a single GPU. Seven [parallelism strategies](docs/model-parallelization.md) address this, each distributing a different dimension of the computation:

### Data Parallelism (DP)
Each GPU holds a full model copy and processes a different data batch. Gradients are synchronized via all-reduce. Simple and effective, but the model must fit in a single GPU's memory.

### Tensor Parallelism (TP)
Individual layers are split across GPUs. For a linear layer with weight matrix W, column-parallel splitting sends different output dimensions to different GPUs. Requires high-bandwidth interconnect (NVLink) since communication happens within every layer.

### Pipeline Parallelism (PP)
The model is partitioned into sequential stages assigned to different GPUs. Micro-batching (GPipe) and interleaved 1F1B scheduling reduce pipeline bubble overhead. Best for cross-node communication where bandwidth is limited.

### Sequence Parallelism (SP)
The sequence dimension is split across GPUs. Ring attention passes KV pairs between devices. Enables context lengths beyond what a single GPU's memory can support.

### Context Parallelism (CP)
Splits the sequence dimension across GPUs for very long contexts (32K+ tokens). Each GPU computes local attention for its chunk, with ring-based communication for cross-chunk dependencies. Unlike SP, Context Parallelism is specifically optimized for long-sequence attention computation, often combined with Flash Attention and sequence pipelining. Enables training and inference on contexts beyond single-GPU memory limits.

### Expert Parallelism (EP)
For Mixture-of-Experts (MoE) models, different experts reside on different GPUs. All-to-all communication routes tokens to the appropriate expert.

### ZeRO / FSDP
Zero Redundancy Optimizer eliminates memory redundancy across data-parallel ranks:
- Stage 1: Shard optimizer states
- Stage 2: Shard optimizer states + gradients
- Stage 3: Shard everything — parameters, gradients, and optimizer states

### 3D Parallelism
Production training combines DP x TP x PP: tensor parallelism within nodes (over NVLink), pipeline parallelism across nodes (over InfiniBand), and data parallelism across replica groups. This is how models like LLaMA 70B are actually trained.

### Communication Infrastructure
All of this depends on [NCCL](docs/pytorch-distributed-training.md) (NVIDIA Collective Communications Library), which provides optimized primitives — AllReduce, AllGather, ReduceScatter, Broadcast — over NVLink, PCIe, InfiniBand, and TCP. PyTorch's `distributed.init_process_group("nccl")` initializes this layer, while [`torchrun`](docs/torch-run.md) handles rank assignment and rendezvous coordination.

---

## 7. The NVIDIA Software Stack: NeMo 2.0, Megatron-Core, and Megatron Bridge

Three components form the core of NVIDIA's training infrastructure:

### Megatron-Core
The low-level, high-performance training library extracted from Megatron-LM. It provides:
- Tensor, pipeline, sequence, and expert parallelism implementations
- Optimized transformer layers with fused kernels
- FP8, BF16, and mixed-precision training via Transformer Engine integration
- Composable, modular APIs for custom architectures

Megatron-Core is what you use when training a 70B model on 64 GPUs with TP=8, PP=4, and sequence parallelism enabled.

### NeMo 2.0
NVIDIA's [end-to-end framework](docs/nemo-2.0.md) for developing and deploying models across NLP, speech, audio, and vision. It wraps Megatron-Core with:
- Model definitions (BERT, GPT, T5, Whisper, and more)
- Trainer, optimizer, metrics, and checkpointing
- Export pipelines (TorchScript, ONNX, TensorRT)
- Config-driven workflows via [NeMo-Run](docs/nemo-run.md)

NeMo also provides [Megatron recipes](docs/megatron-recipe.md) — pre-configured training setups for common model sizes (1.3B, 7B, 13B, 70B, 175B). These recipes encode tested parallelism strategies, batch sizes, and hyperparameters, providing a proven starting point rather than trial-and-error configuration.

The standard NeMo workflow: prepare data (tokenize into Megatron binary format) -> train (NeMo configs + Megatron-Core backend) -> evaluate (BLEU, accuracy, perplexity) -> deploy (export or serve via [Triton](docs/nemo-triton.md)).

### Megatron Bridge
The [compatibility layer](docs/megatron-bridge.md) that connects NVIDIA's ecosystem to the broader ML world:
- Bi-directional checkpoint conversion between Megatron and HuggingFace formats
- Handles sharding/unsharding of distributed checkpoints
- Converts layer names and creates HF-compatible configs
- Enables training at scale with Megatron-Core, then exporting for HuggingFace inference

### Orchestration: torchrun and NeMo-Run
[`torchrun`](docs/torch-run.md) is PyTorch's distributed launcher — it sets RANK, WORLD_SIZE, and LOCAL_RANK environment variables, initializes DDP, and coordinates multi-node training. [NeMo-Run](docs/nemo-run.md) sits on top, providing config-driven orchestration that supports local execution, Slurm clusters, and Kubernetes environments. It replaces the older NeMo Framework Launcher.

---

## 8. Fine-Tuning and Model Optimization

### Supervised Fine-Tuning (SFT)

Before PEFT or alignment, models typically undergo Supervised Fine-Tuning on task-specific demonstration data. SFT transforms a base pretrained model into an instruction-following model by training on high-quality instruction-response pairs. Training typically runs for 1-3 epochs to avoid overfitting. SFT is the behavioral foundation for subsequent alignment (RLHF/DPO), which refines the model's behavior using human preference signals.

```
Pretrained Model → SFT (demonstrations) → Alignment (preferences) → Production
```

NeMo supports SFT with the same distributed training features as pretraining. See [NeMo fine-tuning](docs/nemo-finetuning.md) for configuration.

### Parameter-Efficient Fine-Tuning (PEFT)
Full fine-tuning of a large model is expensive. [PEFT methods](docs/peft.md) update only a small fraction of parameters:

- **LoRA**: Injects low-rank decomposition matrices into attention layers. Trains a fraction of the original parameters.
- **QLoRA**: Combines LoRA with 4-bit quantization of the base model, enabling fine-tuning of large models on consumer hardware.
- **P-Tuning**: Learns continuous prompt embeddings rather than modifying model weights.
- **Adapters**: Inserts small trainable modules between frozen transformer layers.
- **IA3**: Rescales activations via learned vectors — even fewer parameters than LoRA.

NeMo 2.0 exposes these via [config classes](docs/nemo-finetuning.md) (`LoraPEFTConfig`, `QLoraPEFTConfig`, etc.), making it straightforward to swap methods experimentally. See the [NeMo-Run fine-tuning script](src/nemo-run-finetuning.py) for a working example.

### Knowledge Distillation
[Knowledge distillation](docs/knoweldge-distillation.md) trains a smaller student model to mimic a larger teacher. The standard loss function combines ground-truth supervision with soft-label matching:

```
Loss = alpha * CE(student_output, true_labels) + beta * KL(student_logits, teacher_logits)
```

The teacher runs in eval mode with frozen weights, generating soft probability distributions. The student learns from both the hard labels and the teacher's "dark knowledge" — the relative probabilities across all tokens, not just the correct one.

Advanced variants include hidden-state matching (student reproduces intermediate layer representations) and attention distillation (student mimics the teacher's attention patterns).

In the NVIDIA stack, this workflow runs on Megatron-Core with distributed training: the teacher generates logits, the student computes combined CE + KL loss, and backpropagation updates only the student's weights. See the [distillation implementation](src/knowledge-distillation) for reference code.

---

## 9. Checkpoint Management

Moving models between training configurations and ecosystems requires [checkpoint translation](docs/checkpoint-translation.md):

- **Sharding**: Converting a single consolidated checkpoint into distributed shards for multi-GPU training.
- **Unsharding**: Reassembling distributed shards back into a single checkpoint for deployment or export.
- **Format translation**: Converting between Megatron and HuggingFace formats — different layer naming conventions, config structures, and storage layouts.

[Megatron Bridge](docs/megatron-bridge.md) handles all three operations, making it possible to train with Megatron-Core's performance optimizations and then deploy through HuggingFace's ecosystem. See the [bridge script](src/megatron-bridge.py) for a conversion example.

---

## 10. Alignment and Safety

After supervised fine-tuning, production models require alignment to ensure helpful, harmless, and honest behavior. [NeMo Aligner](docs/nemo-aligner.md) provides GPU-accelerated implementations of the core alignment methods.

### RLHF (Reinforcement Learning from Human Feedback)

The classic three-stage pipeline:

1. **Supervised Fine-Tuning (SFT)**: Train on high-quality demonstrations
2. **Reward Model Training**: Train a classifier on human preference pairs (chosen vs. rejected completions)
3. **RL Optimization**: Use PPO to maximize reward while staying close to the SFT model

The reward model learns from preference pairs using the Bradley-Terry model. The RL phase uses the reward signal with a KL divergence penalty to prevent drift:

```
L_PPO = -E[reward(x, y)] + β · KL(π_θ || π_SFT)
```

RLHF is powerful but expensive — four models must coexist in memory (policy, reference, reward, value).

### DPO (Direct Preference Optimization)

[DPO](docs/alignment.md) simplifies RLHF by eliminating the reward model entirely. It directly optimizes the policy on preference pairs:

```
L_DPO = -E[log σ(β · (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]
```

DPO is simpler to implement, more stable, and requires only two models in memory (policy + frozen reference). It often achieves comparable alignment quality to RLHF with significantly less computational overhead.

### Runtime Safety: NeMo Guardrails

While alignment shapes behavior during training, [NeMo Guardrails](docs/nemo-guardrails.md) enforces safety policies at inference time:

- **Input rails**: Block jailbreaks, PII, off-topic queries
- **Output rails**: Filter toxic or harmful generations
- **Dialog rails**: Constrain conversation flow to approved patterns
- **Fact-checking rails**: Validate responses against knowledge bases

Guardrails uses Colang, a modeling language for conversational flows, enabling programmable safety without retraining. NeMo Aligner supports distributed training with the same TP/PP/DP strategies as pretraining, making it practical to align models at any scale.

---

## 11. Quantization

[Quantization](docs/quantization.md) reduces model precision to lower memory footprint and increase throughput. A 70B model in FP16 requires ~140 GB; INT4 brings that to ~35 GB.

### Quantization Methods

| Method | When Applied | Retraining? | Accuracy | NVIDIA Support |
|--------|-------------|-------------|----------|----------------|
| **PTQ** | Post-training | No (calibration only) | Good | TensorRT-LLM, modelopt |
| **QAT** | During training | Yes | Best | Megatron-Core |
| **SmoothQuant** | Post-training | No | Very Good | TensorRT-LLM |
| **AWQ** | Post-training | No | Excellent | TensorRT-LLM |
| **GPTQ** | Post-training | No | Excellent | Community + TRT-LLM |

**PTQ (Post-Training Quantization)**: Calibrate on a small dataset, convert weights and activations. Fast but may degrade accuracy at very low precision.

**SmoothQuant**: Migrates quantization difficulty from activations (which have outliers) to weights (which are smooth) via per-channel scaling. Enables accurate W8A8 inference.

**AWQ**: Protects the most important weight channels (identified by activation magnitudes) from aggressive quantization. Excellent INT4 accuracy without retraining.

### FP8 with Transformer Engine

H100 GPUs include native FP8 Tensor Cores. The [Transformer Engine](docs/gpu-hardware.md) automatically manages FP8 training with dynamic loss scaling and amax tracking. Result: ~2x training throughput over BF16 with minimal accuracy loss. Megatron-Core and NeMo 2.0 integrate Transformer Engine, enabling FP8 with a single config flag.

### Deployment Quantization

[TensorRT-LLM](docs/tensorrt-llm.md) applies quantization during engine compilation. NVIDIA's modelopt toolkit provides a unified API for PTQ, QAT, and sparsity, with direct export to TensorRT-LLM format.

---

## 12. Model Evaluation

Production models require systematic [evaluation](docs/evaluation.md) across multiple dimensions:

### Automatic Metrics

**Perplexity** measures language modeling quality:
```
PPL = exp(-1/N Σ log P(x_i | x_{<i}))
```
Lower perplexity = better prediction. Standard benchmarks: WikiText-2, The Pile validation.

**Task-Specific**: BLEU/ROUGE for summarization and translation, Exact Match and F1 for QA, Pass@k for code generation (HumanEval).

### Benchmark Suites

| Benchmark | Measures | Format |
|-----------|----------|--------|
| **MMLU** | World knowledge (57 subjects) | Multiple choice |
| **HellaSwag** | Commonsense reasoning | Completion selection |
| **HumanEval** | Code generation correctness | Executable tests |
| **GSM8K** | Math reasoning | Free-form numerical |
| **TruthfulQA** | Factual accuracy | QA pairs |
| **ToxiGen** | Safety and bias | Classification |
| **MT-Bench** | Multi-turn conversation quality | LLM-as-judge |

### Evaluation Frameworks

**EleutherAI lm-evaluation-harness** is the standard open-source framework with 200+ pre-implemented tasks and support for zero-shot and few-shot evaluation.

### Evaluation Across the Pipeline

Track metrics at every stage: pretraining (perplexity, MMLU), SFT (instruction-following), alignment (TruthfulQA, ToxiGen — watch for alignment tax), and quantization (<1% degradation on critical metrics). See [evaluation details](docs/evaluation.md) for comprehensive guidance.

---

## 13. Inference Serving: TensorRT-LLM, Triton, and NIM

While the optimization techniques in Section 2 improve inference computation, production deployment requires a full serving stack.

### TensorRT-LLM

[TensorRT-LLM](docs/tensorrt-llm.md) compiles model checkpoints into optimized inference engines:
- **Kernel fusion**: QKV projections, attention + softmax, LayerNorm + activation combined into single kernels
- **Graph optimization**: Constant folding, memory planning, optimal kernel selection per GPU
- **Quantization**: FP8, INT8, INT4 (AWQ, GPTQ, SmoothQuant) applied at build time
- **In-flight batching**: Continuous batching at the engine level
- **Paged KV cache**: Virtual memory management for KV cache

```
NeMo/HuggingFace Checkpoint → TRT-LLM Builder → Optimized Engine (.engine)
```

Typical speedup: 2-5x over native PyTorch inference with multi-GPU support (TP, PP).

### Triton Inference Server

[Triton](docs/nemo-triton.md) provides production-grade model serving:
- **Dynamic batching**: Automatically groups requests for higher throughput
- **TRT-LLM backend**: Purpose-built for LLM workloads with streaming support
- **Multi-model serving**: Run multiple models on shared GPUs
- **Model versioning**: A/B testing and rolling updates
- **Monitoring**: Prometheus-compatible metrics (latency, throughput, GPU utilization)

### NVIDIA NIM

[NVIDIA NIM](docs/nvidia-nim.md) packages the entire inference stack into pre-built, containerized microservices:

- **Pre-optimized models**: TensorRT-LLM compilation handled automatically
- **OpenAI-compatible API**: Drop-in replacement for existing applications
- **One-command deployment**: `docker run --gpus all nvcr.io/nvidia/nim/model:latest`
- **Automatic optimization**: Selects quantization, parallelism, and batching for the target GPU

NIM abstracts TensorRT-LLM builds, Triton configuration, and optimization tuning. It integrates with [NeMo Guardrails](docs/nemo-guardrails.md) for runtime safety and supports embedding and reranking NIMs for [RAG](docs/rag.md) pipelines.

---

## 14. Retrieval-Augmented Generation (RAG)

[RAG](docs/rag.md) extends LLMs with external knowledge, addressing hallucination and knowledge cutoff:

```
User Query → Embed → Retrieve top-k from Vector DB → Augment Prompt → Generate Answer
```

### Components

**Document Chunking**: Split documents into semantically coherent chunks (typically 256-512 tokens with 10-20% overlap). Strategies range from fixed-size splitting to semantic boundary detection.

**Embedding Models**: Convert text to dense vectors for similarity search. See [vector database embeddings](docs/vector-db-embedding.md) for model choices including BGE, E5, and Cohere embed-v3.

**Vector Databases**: Store and index embeddings for fast approximate nearest neighbor (ANN) search. Options include Milvus (GPU-accelerated), Weaviate, Pinecone, FAISS, and pgvector.

**Retrieval + Re-ranking**: Initial top-k retrieval via ANN search, refined with cross-encoder re-ranking for more accurate relevance scoring. Hybrid search (dense + BM25 sparse) catches both semantic matches and exact keyword hits.

### Advanced RAG

- **HyDE**: Generate a hypothetical answer, embed it, retrieve based on that embedding
- **Multi-query**: Generate multiple query variations, retrieve for each, merge results
- **Self-RAG**: Model decides when to retrieve and critiques its own answers
- **Contextual compression**: Rewrite retrieved chunks to remove irrelevant content

### NVIDIA Stack for RAG

Combine embedding NIM + reranking NIM + LLM NIM for end-to-end RAG serving, with NeMo Guardrails for safety and Milvus for GPU-accelerated vector search.

---

## 15. GPU Hardware and Interconnects

The NVIDIA generative AI stack is optimized for specific [GPU hardware](docs/gpu-hardware.md). Understanding the hardware hierarchy is essential for choosing parallelism strategies.

### GPU Architectures

**Hopper (H100)**: 4th-gen Tensor Cores with native FP8 support via Transformer Engine. 80 GB HBM3, 3.35 TB/s memory bandwidth, 900 GB/s NVLink. The current workhorse for LLM training.

**Blackwell (B200)**: 5th-gen Tensor Cores with FP4 support. 192 GB HBM3e, 8 TB/s bandwidth, 1,800 GB/s NVLink. ~2x throughput improvement over Hopper.

### Interconnect Hierarchy

| Interconnect | Bandwidth | Latency | Use Case |
|--------------|-----------|---------|----------|
| **NVLink** | 900 GB/s (H100) | ~1 μs | Tensor Parallelism within node |
| **NVSwitch** | Full mesh at NVLink speed | ~1 μs | All-to-all GPU connectivity in DGX |
| **InfiniBand NDR** | 400 Gbps (~50 GB/s) | ~5 μs | Pipeline/Data Parallelism across nodes |

**DGX H100**: 8x H100 GPUs connected via NVSwitch for full-bandwidth mesh communication. The building block for SuperPOD clusters (32+ DGX nodes).

### Impact on Parallelism

The bandwidth hierarchy dictates optimal parallelism mapping:
- **Tensor Parallelism**: Within node over NVLink (highest bandwidth, lowest latency)
- **Pipeline Parallelism**: Across nodes over InfiniBand (lower communication frequency)
- **Data Parallelism**: Across replica groups (infrequent gradient synchronization)

This is why 3D parallelism configurations use TP=8 within DGX nodes and PP across nodes — matching communication patterns to hardware topology. See [distributed training](docs/model-parallelization.md) for strategy details and [NCCL](docs/pytorch-distributed-training.md) for communication primitives.

---

## 16. Multimodal Models

[Vision-Language Models (VLMs)](docs/multimodal.md) extend LLMs to process images alongside text:

### Architecture

```
Image → Vision Encoder (ViT) → Projection Layer ─┐
                                                   ↓
Text  → Token Embedding ────────────────→ LLM Decoder → Output
```

**Vision Encoder**: Vision Transformer (ViT) processes image patches as tokens. CLIP encoders are commonly used for their pre-trained alignment between image and text representations.

**Projection Layer**: Maps vision embedding space to LLM token embedding space (linear layer or small MLP). Often the only component trained from scratch when combining pretrained vision and language models.

**LLM Decoder**: Standard autoregressive transformer processes interleaved vision and text tokens through its self-attention mechanism.

### Training Strategies

| Strategy | What's Trained | Cost | Quality |
|----------|---------------|------|---------|
| Frozen encoders + train projection | Projection layer only | Low | Baseline |
| Frozen vision + LoRA on LLM | Projection + LLM adapters | Medium | Good |
| End-to-end fine-tuning | Everything | High | Best |

### Challenges

Resolution directly impacts compute: a 336×336 image produces 576 vision tokens vs. 256 for 224×224 — with attention cost scaling quadratically. Solutions include dynamic resolution, token pooling, and tiled processing.

### NeMo Multimodal Support

NeMo 2.0 provides multimodal model definitions, distributed training for VLMs (TP/PP across vision and language components), and export to TensorRT for joint vision-text inference. Vision-Language NIMs enable production deployment of models like LLaVA and VILA.

---

## Putting It All Together

The NVIDIA generative AI stack is best understood as a pipeline where each layer addresses a specific engineering challenge:

```
┌───────────────────────────────────────────────────────────┐
│  Hardware Layer                                           │
│  H100 / Blackwell GPUs + NVLink / NVSwitch / InfiniBand   │
└─────────────────────────┬─────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────┐
│  Data Curation (NeMo Curator)           │
│  Ingestion → Filtering → Deduplication  │
└─────────────────────────┬───────────────┘
                          ↓
┌─────────────────────────────────────────┐
│  Tokenization + Binary Dataset Creation │
└─────────────────────────┬───────────────┘
                          ↓
┌─────────────────────────────────────────┐
│  Pretraining                            │
│  Megatron-Core + NeMo 2.0               │
│  TP / PP / CP / DP / ZeRO              │
│  FP8 (Transformer Engine) + NCCL        │
│  torchrun / NeMo-Run orchestration      │
└─────────────────────────┬───────────────┘
                          ↓
┌─────────────────────────────────────────┐
│  Fine-Tuning                            │
│  SFT → PEFT (LoRA/QLoRA)               │
│  Knowledge Distillation                 │
└─────────────────────────┬───────────────┘
                          ↓
┌─────────────────────────────────────────┐
│  Alignment (NeMo Aligner)               │
│  RLHF / DPO / SteerLM                  │
└─────────────────────────┬───────────────┘
                          ↓
┌─────────────────────────────────────────┐
│  Evaluation                             │
│  MMLU, HumanEval, TruthfulQA, MT-Bench │
└─────────────────────────┬───────────────┘
                          ↓
┌─────────────────────────────────────────┐
│  Quantization (modelopt)                │
│  PTQ / QAT / AWQ / SmoothQuant / FP8   │
└─────────────────────────┬───────────────┘
                          ↓
┌─────────────────────────────────────────┐
│  Checkpoint Translation                 │
│  Megatron Bridge (Megatron ↔ HuggingFace)│
└─────────────────────────┬───────────────┘
                          ↓
┌─────────────────────────────────────────┐
│  Inference Optimization                 │
│  TensorRT-LLM compilation              │
│  Kernel fusion + KV cache optimization  │
│  In-flight batching + Paged Attention   │
└─────────────────────────┬───────────────┘
                          ↓
┌─────────────────────────────────────────┐
│  Deployment                             │
│  Option A: Triton Inference Server      │
│  Option B: NVIDIA NIM (containerized)   │
│  + NeMo Guardrails (runtime safety)     │
└─────────────────────────┬───────────────┘
                          ↓
┌─────────────────────────────────────────┐
│  Applications                           │
│  Direct inference │ RAG pipelines       │
│  Multimodal systems │ AI agents         │
└─────────────────────────────────────────┘
```

Each component is modular — you can use NeMo Curator without NeMo 2.0, or TensorRT-LLM without Triton — but the stack is designed to work together, with NVIDIA GPU acceleration at every stage.

For engineers entering this space, the most productive path is to start with the fundamentals (attention, tokenization, embeddings), understand the inference memory profile (prefill vs. decode, KV cache sizing), and then work through the parallelism strategies that make large-scale training possible. From there, the post-training pipeline (SFT → alignment → evaluation → quantization) and the serving stack (TensorRT-LLM → Triton/NIM) become the path to production. The NVIDIA tooling becomes intuitive once you understand the problems it solves.

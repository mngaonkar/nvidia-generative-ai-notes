# Nemo Curator
NeMo Data Curator is an open-source, GPU-accelerated data curation pipeline developed by NVIDIA as part of the NeMo Framework. Its primary purpose is to help researchers and ML engineers prepare high-quality text datasets for pretraining large language models (LLMs). It is designed to handle datasets at massive scale — from hundreds of gigabytes to petabytes — with efficiency and reproducibility.

## Key Components and Capabilities

### Data Download & Format Handling
The pipeline can ingest data from Common Crawl (WARC/WET files), HuggingFace datasets, local files, and custom sources. It handles multiple formats including JSON, JSONL, Parquet, and plain text, and converts them into a unified format for downstream processing.

### Language Identitication
Using libraries like fastText or langdetect, Data Curator can classify and filter documents by language.

### Text Extraction and Cleaning
For HTML or WARC content, it extracts clean text by removing boilerplate (navigation bars, ads, etc.) using tools like JusText or Trafilatura. It also performs Unicode normalization and basic cleaning like fixing encoding artifacts.

### Heuristic Quality Filtering
- Word count and character count thresholds — removing very short or very long documents
- Symbol-to-word ratio — filtering out documents dominated by special characters or code-like noise
- Bullet point ratio — flagging documents that are more list-heavy than prose
- Perplexity filtering — using an n-gram language model (like a KenLM model) to score how "natural" text is and filter out low-perplexity junk
- Stop word presence — ensuring the text reads like natural language

### Deduplication
- Exact deduplication using MD5/SHA hashing on document or paragraph level
- Fuzzy deduplication using MinHash + LSH (Locality Sensitive Hashing), which identifies near-duplicate documents that share most of their content but may differ slightly.

### PII (Personally Identifiable Information) Redaction
Data Curator includes tools to detect and redact PII such as names, email addresses, phone numbers, IP addresses, and more. This uses rule-based regex patterns as well as NER (Named Entity Recognition) models, helping teams comply with privacy regulations like GDPR.

### Classifier-Based Filtering
Beyond heuristics, you can plug in custom ML classifiers to score and filter documents. NVIDIA provides pre-trained classifiers for things like:

- Toxicity / NSFW content detection
- Quality scoring
- Domain tagging

### Output Format
Output clean, shuffled JSONL or Parquet for training.
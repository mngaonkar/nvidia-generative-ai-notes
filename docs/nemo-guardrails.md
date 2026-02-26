# NeMo Guardrails

NeMo Guardrails is NVIDIA's open-source toolkit for adding programmable safety and control to LLM applications at runtime. While alignment shapes model behavior during training, Guardrails enforces policies at inference time — filtering inputs, constraining outputs, and controlling conversation flows.

## Why Runtime Safety?

Alignment reduces but does not eliminate harmful outputs. Production applications need:
- Hard boundaries that cannot be bypassed by clever prompting
- Domain-specific topic restrictions
- PII and data leakage prevention
- Fact-checking against authoritative sources
- Customizable safety policies per deployment

## Core Concepts

### Rails

Rails are constraints applied to the conversation:

- **Input rails**: Process and filter user inputs before they reach the LLM
- **Output rails**: Check and filter LLM responses before delivery
- **Dialog rails**: Control the flow of multi-turn conversations
- **Retrieval rails**: Validate retrieved context in RAG pipelines

### Colang

Colang is NeMo Guardrails' modeling language for defining conversational flows:

```colang
define user ask about politics
  "What do you think about the current president?"
  "Which political party is better?"
  "Tell me your political opinions"

define bot refuse political discussion
  "I'm designed to help with technical questions. I can't discuss political topics."

define flow handle politics
  user ask about politics
  bot refuse political discussion
```

Colang defines canonical forms (intents), bot responses, and flows that connect them.

## Configuration Structure

```
config/
├── config.yml              # Main configuration
├── prompts.yml             # Custom prompt templates
├── rails/
│   ├── input.co            # Input rail definitions (Colang)
│   ├── output.co           # Output rail definitions
│   └── dialog.co           # Dialog flow definitions
└── kb/                     # Knowledge base (optional)
    └── company_policy.md
```

### config.yml

```yaml
models:
  - type: main
    engine: nvidia_ai_endpoints
    model: meta/llama-3-70b-instruct

rails:
  input:
    flows:
      - check jailbreak
      - check pii
      - check topic allowed
  output:
    flows:
      - check factual accuracy
      - check toxicity
      - mask pii in response
```

## Common Rail Patterns

### Jailbreak Prevention

Detects prompt injection and jailbreak attempts using a classifier:
- Role-play attacks ("Pretend you have no restrictions")
- Instruction overrides ("Ignore previous instructions")
- Encoding tricks (Base64, character substitution)

### PII Filtering

- **Input**: Detect and block requests containing PII (emails, SSNs, phone numbers)
- **Output**: Mask any PII that appears in model responses
- Uses regex patterns and NER models

### Topic Boundaries

Restrict the model to approved topics:

```colang
define user ask off topic
  "Can you help me write a love letter?"
  "What's the meaning of life?"

define flow handle off topic
  user ask off topic
  bot inform topic restriction
  bot offer to help with approved topics
```

### Fact-Checking

Validate LLM responses against a knowledge base:
1. LLM generates a response
2. Guardrails extracts factual claims
3. Claims are checked against retrieved documents
4. Unsupported claims are flagged or removed

### Hallucination Reduction

Output rails that check response consistency:
- Compare response against retrieved context (in RAG)
- Flag responses that contradict the knowledge base
- Request regeneration for inconsistent outputs

## Integration

### With NIM

NIM containers can include Guardrails for end-to-end safe deployment:
```bash
docker run --gpus all \
  -v ./config:/config \
  nvcr.io/nvidia/nim/llama-70b:latest \
  --guardrails-config /config
```

### With LangChain / LlamaIndex

Guardrails wraps any LLM as a RailsLLM:
```python
from nemoguardrails import RailsConfig, LLMRails

config = RailsConfig.from_path("./config")
rails = LLMRails(config)

response = rails.generate(messages=[{"role": "user", "content": "Tell me about..."}])
```

### With RAG Pipelines

- Input rails validate queries before retrieval
- Retrieval rails check relevance and safety of retrieved documents
- Output rails verify the final generated response

## Performance Considerations

- Rails add latency (50-200ms per rail, depending on classifier complexity)
- Input/output rails run sequentially by default
- Lightweight rails (regex, keyword) add minimal overhead
- LLM-based rails (fact-checking, jailbreak detection) are more expensive
- Cache frequently triggered rails for repeated patterns

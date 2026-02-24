# Nemo 2.0

NVIDIA’s main deep learning framework for NLP, Speech, Audio, and Vision. It supports end-to-end development and deployment of large language models on-premnises, data center or public cloud. Nemo supports execution on Slurm or Kubernetes environment.

It provides:
- Model definitions (BERT, GPT, T5, Whisper, etc.)
- Trainer + optimizer + metrics
- Checkpointing
- Evaluation
- Exporters (TorchScript, ONNX, TensorRT)

```
pip install nemo-toolkit[all]
```

## Recommended Workflow
### Step 1 - Data Preparation

Use NeMo-Run or Python script to:
1. Download your dataset (e.g., Pile, Wikipedia dumps, custom data)
2. Tokenize (WordPiece / SentencePiece / Custom)
3. Create Megatron-Core binary dataset formats (.bin + .idx)
(needed if training large models like GPT / T5)

You can use NeMo’s built-in preprocessors or write a simple script — NeMo provides utilities.

[Nemo Data Curator](nemo-curator.md)

### Step 2 - Model Training

With NeMo 2.0 + Megatron-Bridge:
Example with NeMo’s T5 training recipe:

```
nemo-run \
  --config-path /path/to/configs \
  --config-name nemo_t5_training_config.yaml \
  model.train_ds.data_prefix=/path/to/bin_prefix \
  model.validation_ds.data_prefix=/path/to/bin_prefix
```
NeMo-Run wraps torchrun behind the scenes.

### Step 3 - Model Evaluation

After training completes:
```
nemo-run \
  --config-path eval_configs \
  --config-name t5_eval.yaml \
  model.restore_path=/path/to/checkpoint.ckpt
```

You get BLEU / accuracy / perplexity metrics.

### Step 4 - Inference
You can run:
```
from nemo.collections.nlp.models import TextGenerationModel

model = TextGenerationModel.restore_from("my_model.nemo")
response = model.generate(["translate Akkadian to English: ana ilu"])
```

Or export to ONNX / TensorRT for deployment.


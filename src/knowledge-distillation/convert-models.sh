#!/bin/sh
# Teacher
python examples/conversion/convert_checkpoints.py import \
  --hf-model google/gemma-2-9b \
  --megatron-path /models/megatron/gemma2-9b

# Student
python examples/conversion/convert_checkpoints.py import \
  --hf-model google/gemma-2-2b \
  --megatron-path /models/megatron/gemma2-2b
#!/bin/sh

torchrun \
  --nnodes=2 --nproc_per_node=2 \
  --node_rank=0 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=<NODE0_IP>:29500 \
  mcore_kd_gemma2.py \
  --teacher_size 9B --student_size 2B \
  --teacher_core_ckpt /models/megatron/gemma2-9b \
  --student_core_ckpt /models/megatron/gemma2-2b \
  --save_student_ckpt /models/megatron/gemma2-2b-kd \
  --tp 2 --pp 1 \
  --temperature 2.0 --alpha_lm 0.2 --beta_kd 1.0 \
  --steps 200 --micro_batch_size 1

  torchrun \
  --nnodes=2 --nproc_per_node=2 \
  --node_rank=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=<NODE0_IP>:29500 \
  mcore_kd_gemma2.py \
  --teacher_size 9B --student_size 2B \
  --teacher_core_ckpt /models/megatron/gemma2-9b \
  --student_core_ckpt /models/megatron/gemma2-2b \
  --save_student_ckpt /models/megatron/gemma2-2b-kd \
  --tp 2 --pp 1 \
  --temperature 2.0 --alpha_lm 0.2 --beta_kd 1.0 \
  --steps 200 --micro_batch_size 1
import os
import argparse
from functools import partial
from pathlib import Path
from typing import Dict, Iterator, Tuple, Callable, Any

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from megatron.core import parallel_state, dist_checkpointing
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads

# Gemma2 providers (Megatron Bridge)
from bridge.models.gemma.gemma2_provider import (
    Gemma2ModelProvider2B,
    Gemma2ModelProvider9B,
    Gemma2ModelProvider27B,
)

SEQ_LEN = 2048  # set <= provider.seq_length (Gemma2 default seq_length is 8192 in provider)  :contentReference[oaicite:3]{index=3}


# ---- toy dataset (replace with real tokenized dataset) ----
class RandomLMDataset(torch.utils.data.Dataset):
    def __init__(self, vocab_size: int, n: int = 20000):
        self.vocab_size = vocab_size
        self.n = n

    def __len__(self): return self.n

    def __getitem__(self, idx):
        tokens = torch.randint(0, self.vocab_size, (SEQ_LEN,), dtype=torch.long)
        labels = tokens.clone()
        # Minimal placeholders. For real training you should build correct causal mask & positions.
        attention_mask = torch.ones((1, 1, SEQ_LEN, SEQ_LEN), dtype=torch.bool)
        position_ids = torch.arange(SEQ_LEN, dtype=torch.long)
        return {
            "tokens": tokens,
            "labels": labels,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }


def init_dist(tp: int, pp: int):
    parallel_state.destroy_model_parallel()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world)

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
    )


def load_core_ckpt(ckpt_dir: str, model: torch.nn.Module) -> torch.nn.Module:
    underlying = model.module if hasattr(model, "module") else model
    sharded = underlying.sharded_state_dict(prefix="")
    state = dist_checkpointing.load(sharded_state_dict=sharded, checkpoint_dir=ckpt_dir)
    underlying.load_state_dict(state)
    return model


def kd_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float) -> torch.Tensor:
    # KL(teacher || student) using softened probabilities
    s_logp = F.log_softmax(student_logits / T, dim=-1)
    t_p = F.softmax(teacher_logits / T, dim=-1)
    return F.kl_div(s_logp, t_p, reduction="batchmean") * (T * T)


def forward_step_kd(
    data_iter: Iterator[Dict[str, torch.Tensor]],
    student_ddp: torch.nn.Module,
    teacher: torch.nn.Module,
    alpha_lm: float,
    beta_kd: float,
    temperature: float,
    device: torch.device,
) -> Tuple[Any, Callable]:

    batch = next(data_iter)
    tokens = batch["tokens"].to(device)
    labels = batch["labels"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    position_ids = batch["position_ids"].to(device)

    # ---- Teacher forward (no grad) ----
    teacher.eval()
    with torch.no_grad():
        teacher_logits = teacher(tokens, position_ids, attention_mask, labels=None)

    # ---- Student forward ----
    student_logits = student_ddp(tokens, position_ids, attention_mask, labels=None)
    lm_losses = student_ddp(tokens, position_ids, attention_mask, labels=labels)

    def loss_func(payload):
        lm_losses_, s_logits_, t_logits_ = payload
        lm_loss = lm_losses_.float().mean()
        kd_loss = kd_kl(s_logits_, t_logits_, float(temperature))
        total = alpha_lm * lm_loss + beta_kd * kd_loss
        return total, {
            "lm_loss": lm_loss.detach(),
            "kd_loss": kd_loss.detach(),
            "total_loss": total.detach(),
        }

    return (lm_losses, student_logits, teacher_logits), loss_func


def pick_gemma2_provider(name: str):
    name = name.lower()
    if name in ("2b", "gemma2-2b", "gemma-2-2b"):
        return Gemma2ModelProvider2B()
    if name in ("9b", "gemma2-9b", "gemma-2-9b"):
        return Gemma2ModelProvider9B()
    if name in ("27b", "gemma2-27b", "gemma-2-27b"):
        return Gemma2ModelProvider27B()
    raise ValueError("Unknown Gemma2 size. Use: 2B, 9B, 27B")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_core_ckpt", type=str, required=True)
    ap.add_argument("--student_core_ckpt", type=str, required=True)
    ap.add_argument("--save_student_ckpt", type=str, default="./student_kd_out")

    ap.add_argument("--teacher_size", type=str, required=True, help="2B|9B|27B")
    ap.add_argument("--student_size", type=str, required=True, help="2B|9B|27B")

    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--pp", type=int, default=1)

    # KD hyperparams
    ap.add_argument("--alpha_lm", type=float, default=0.2)
    ap.add_argument("--beta_kd", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=2.0)

    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--micro_batch_size", type=int, default=1)
    args = ap.parse_args()

    init_dist(args.tp, args.pp)
    model_parallel_cuda_manual_seed(123)
    device = torch.device("cuda")

    # ---- Build Gemma2 teacher/student via Megatron Bridge providers ----
    teacher_provider = pick_gemma2_provider(args.teacher_size)
    student_provider = pick_gemma2_provider(args.student_size)

    # Provider returns a Megatron-Core GPT model configured for Gemma2 (RMSNorm, softcapping, SWA, etc.) :contentReference[oaicite:4]{index=4}
    teacher = teacher_provider.provide().to(device).bfloat16()
    student = student_provider.provide().to(device).bfloat16()

    # Load converted Megatron-Core checkpoints
    teacher = load_core_ckpt(args.teacher_core_ckpt, teacher)
    student = load_core_ckpt(args.student_core_ckpt, student)

    # Freeze teacher
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Wrap student in Megatron-Core DDP
    ddp_cfg = DistributedDataParallelConfig(
        grad_reduce_in_fp32=False,
        overlap_grad_reduce=False,
        use_distributed_optimizer=False,
    )
    student_ddp = DistributedDataParallel(config=student.config, ddp_config=ddp_cfg, module=student)

    optim = Adam(student_ddp.parameters(), lr=2e-5)

    # IMPORTANT: use provider vocab_size so your data tokens match Gemma vocab (Gemma2 provider vocab_size=256000). :contentReference[oaicite:5]{index=5}
    vocab_size = int(student_provider.vocab_size)
    ds = RandomLMDataset(vocab_size=vocab_size, n=50000)
    dl = DataLoader(ds, batch_size=args.micro_batch_size, shuffle=True, num_workers=2, pin_memory=True)
    data_iter = iter(dl)

    fwd_bwd = get_forward_backward_func()

    for step in range(args.steps):
        optim.zero_grad(set_to_none=True)

        fwd = partial(
            forward_step_kd,
            teacher=teacher,
            alpha_lm=args.alpha_lm,
            beta_kd=args.beta_kd,
            temperature=args.temperature,
            device=device,
        )

        losses = fwd_bwd(
            forward_step_func=fwd,
            data_iterator=data_iter,
            model=student_ddp,
            num_microbatches=1,
            seq_length=SEQ_LEN,
            micro_batch_size=args.micro_batch_size,
            decoder_seq_length=SEQ_LEN,
            forward_only=False,
        )

        finalize_model_grads([student_ddp])
        optim.step()

        if torch.distributed.get_rank() == 0 and step % 10 == 0:
            print(f"step={step} losses={losses}")

    # Save distilled student
    outdir = Path(args.save_student_ckpt)
    outdir.mkdir(parents=True, exist_ok=True)
    underlying = student_ddp.module if hasattr(student_ddp, "module") else student_ddp
    dist_checkpointing.save(underlying.sharded_state_dict(prefix=""), checkpoint_dir=str(outdir))

    if torch.distributed.get_rank() == 0:
        print(f"Saved distilled student checkpoint to: {outdir}")


if __name__ == "__main__":
    main()
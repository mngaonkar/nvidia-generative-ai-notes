import nemo_run as run
from nemo.collections import llm
from nemo.collections.llm.gpt.data import FineTuningDataModule  # generic seq2seq data module

RUN_LOCAL = True  # set to False to run on cluster (e.g., Slurm)

def main():
    # NeMo 2.0 provides T5 finetune recipes (Megatron-T5 variants like 220M/3B/11B).
    # This example sets 2 nodes x 2 GPUs per node.
    recipe = llm.t5_220m.finetune_recipe(
        name="t5_220m_finetune_2n2g",
        ckpt_dir="/mnt/checkpoints",   # where NeMo will write checkpoints/logs
        num_nodes=2,
        num_gpus_per_node=2,
        peft_scheme="lora",            # 'lora' or 'none'
    )

    # By default, the recipe uses a SQuAD datamodule; replace recipe.data with your own datamodule
    # when you’re ready. (NeMo docs show this override pattern.)
    recipe.data = FineTuningDataModule(
        train_path="/mnt/data/train_sample.json",
        validation_path="/mnt/data/val_sample.json",
        input_key="src",
        target_key="tgt",
        max_seq_length=512,
        micro_batch_size=4,
        global_batch_size=16, # gradient accumulation steps = global_batch_size // (micro_batch_size * num_gpus)
        num_workers=4, # no. of CPU threads for data loading
    )

    # Run locally (single machine) would be LocalExecutor;
    # for multi-node, you typically use a cluster executor (e.g., Slurm).
    if not RUN_LOCAL:
        run.run(recipe, executor=run.SlurmExecutor(
            account="YOUR_ACCOUNT",
            partition="YOUR_PARTITION",
            nodes=2,
            gpus_per_node=2,
            time="02:00:00",
        # You can set launcher="torchrun" in some setups; NeMo-Run supports torchrun launchers.
        launcher="torchrun",
        ))
    else:
        # Run locally (single machine)
        run.run(recipe, executor=run.LocalExecutor(
        ntasks_per_node=2,  # number of GPUs to use
        launcher="torchrun",
        ))

if __name__ == "__main__":
    main()
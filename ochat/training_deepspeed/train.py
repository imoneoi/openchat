import argparse
import os
import math
from functools import partial

import torch
import torch.distributed

import transformers
import deepspeed
import tqdm
import wandb
import numpy as np
import pyarrow

from ochat.config.model_config import MODEL_CONFIG_MAP
from ochat.training_deepspeed.multipack_dataloader import MultipackDistributedDataloader
from ochat.training_deepspeed.parquet_dataset import ParquetDataset


LOCAL_RANK      = None

PAD_ID          = 0
IGNORE_LABEL_ID = -100


def _find_multiple(a, b):
    return (-(a // -b)) * b


def _rank0_print(*args):
    global LOCAL_RANK

    if LOCAL_RANK == 0:
        tqdm.tqdm.write(*args)


def parse_args():
    parser = argparse.ArgumentParser()
    # Distributed
    parser.add_argument("--local_rank", type=int, required=True)

    # Model type and data
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path",  type=str, required=True)
    parser.add_argument("--save_path",  type=str, required=True)
    parser.add_argument("--save_every", type=int, default=None)

    # Hyperparameters
    parser.add_argument("--batch_size_per_gpu", type=int,   default=16)
    parser.add_argument("--epochs",             type=int,   default=5)

    # Set lr to None to automatically estimate from LLaMA pretraining parameters (e.g. lr ~ sqrt(batch_size))
    parser.add_argument("--lr",                 type=float, default=None)
    parser.add_argument("--lr_min_ratio",       type=float, default=0.1)
    parser.add_argument("--lr_warmup_ratio",    type=int,   default=0.05)

    parser.add_argument("--weight_decay",       type=float, default=0.1)

    parser.add_argument("--beta1",              type=float, default=0.9)
    parser.add_argument("--beta2",              type=float, default=0.95)
    parser.add_argument("--eps",                type=float, default=1e-5)

    # DeepSpeed parameters
    parser = deepspeed.add_config_arguments(parser)

    # Parse known args
    args, unknown = parser.parse_known_args()
    return args


def create_dataset(args, split_name):
    # Load data
    filename = f"{args.data_path}.{split_name}.parquet"
    if not os.path.isfile(filename):
        _rank0_print (f"Skipping loading {split_name}")
        return None

    return ParquetDataset(filename)


def batch_to_tensor(batch, int_dtype=torch.long, loss_dtype=torch.bfloat16):
    # Pad an unused item to reach multiple of 64, for faster GEMM
    total_seqlen = pyarrow.compute.sum(batch.column("total_length")).as_py()
    pad_len      = _find_multiple(total_seqlen, 64) - total_seqlen

    if pad_len > 0:
        assert pad_len < 64

        # total length
        batch = pyarrow.concat_tables((batch, pyarrow.Table.from_pydict({
            "total_length": [pad_len],
            "num_seqs": [0],

            "seqlens": [[pad_len]],
            "nz_input_ids": [[PAD_ID] * pad_len],
            "nz_position_ids": [[0] * pad_len],
            "nz_shifted_label_ids": [[IGNORE_LABEL_ID] * pad_len],
            "nz_shifted_loss_weights": [[0.0] * pad_len],
        }, schema=batch.schema)))

    # concatenate
    batch_tensor = {}
    keys = {
        "seqlens": int_dtype, "nz_input_ids": int_dtype, "nz_position_ids": int_dtype, "nz_shifted_label_ids": int_dtype,
        "nz_shifted_loss_weights": loss_dtype
    }

    for k, dtype in keys.items():
        batch_tensor[k] = torch.from_numpy(np.concatenate(batch.column(k).to_numpy())).to(dtype)

    # cu seqlens
    batch_tensor["max_seqlen"] = torch.max(batch_tensor["seqlens"])
    batch_tensor["cu_seqlens"] = torch.nn.functional.pad(batch_tensor["seqlens"].cumsum(-1, dtype=torch.int32), (1, 0))

    del batch_tensor["seqlens"]

    # inputs
    return batch_tensor


def create_distributed_dataloader(args, data):
    # Check data
    assert data.metadata["model_type"] == args.model_type, \
        f"The dataset is for {data.metadata['model_type']}, but you specified {args.model_type} for training."

    # Multipack dataloader
    args.batch_max_len = args.batch_size_per_gpu * MODEL_CONFIG_MAP[args.model_type].model_max_context

    return MultipackDistributedDataloader(
        dataset=data,
        lengths=data["total_length"],
        numseqs=data["num_seqs"],

        batch_max_length=args.batch_max_len,
        collate_fn=batch_to_tensor,

        seed=0
    )


def create_model(args):
    global LOCAL_RANK

    # Create model + optimizer + lr scheduler
    model = MODEL_CONFIG_MAP[args.model_type].model_create(args.model_path)
    # Model to assigned cuda device
    model = model.to(LOCAL_RANK)
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  betas=(args.beta1, args.beta2),
                                  eps=args.eps,
                                  fused=True)

    # DeepSpeed model
    model_engine, optimizer, _, _ = deepspeed.initialize(args=args,
                                                         model=model,
                                                         model_parameters=model.parameters(),
                                                         optimizer=optimizer)

    # Put deepspeed arguments
    args.device                         = model_engine.device

    return model_engine, optimizer


def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


def create_lr_scheduler(args, train_total_steps):
    lr_scheduler = partial(
        cosine_schedule_with_warmup_lr_lambda,

        num_warmup_steps=round(args.lr_warmup_ratio * train_total_steps),
        num_training_steps=train_total_steps,
        min_ratio=args.lr_min_ratio
    )

    return lr_scheduler


def save_tokenizer(args, save_path):
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    tokenizer.eos_token = MODEL_CONFIG_MAP[args.model_type].eot_token
    tokenizer.save_pretrained(save_path)


def calculate_auto_lr(lr, batch_max_len, train_dataset):
    if lr is not None:
        return lr
    
    # Llama hyperparameters
    # FIXME: Only 7B/13B is supported
    base_lr = 3e-4
    base_bs = 4_000_000

    label_ids = np.concatenate(train_dataset["nz_shifted_label_ids"])
    supervised_ratio = np.sum(label_ids != IGNORE_LABEL_ID) / len(label_ids)

    supervised_tokens = batch_max_len * torch.distributed.get_world_size() * supervised_ratio
    lr = base_lr * math.sqrt(supervised_tokens / base_bs)

    _rank0_print(f"Use automatic learning rate {lr} (estimated from supervised ratio {supervised_ratio} effective batch size {supervised_tokens})")
    return lr


def train():
    global LOCAL_RANK

    deepspeed.init_distributed(dist_backend="nccl")

    # Args
    args       = parse_args()
    LOCAL_RANK = args.local_rank

    # Dataset
    _rank0_print("Loading data...")
    train_dataset = create_dataset(args, "train")
    eval_dataset  = create_dataset(args, "eval")

    # Data Loader
    train_loader      = create_distributed_dataloader(args, train_dataset)
    train_total_steps = args.epochs * train_loader.num_batches()

    eval_loader = None
    if eval_dataset is not None:
        eval_loader = create_distributed_dataloader(args, eval_dataset)

    # Hyperparams
    args.lr = calculate_auto_lr(args.lr, args.batch_max_len, train_dataset)

    # Model
    _rank0_print("Loading model...")
    model_engine, optimizer = create_model(args)

    # LR Scheduler
    lr_scheduler = create_lr_scheduler(args, train_total_steps)

    # Progress bar and logger
    progress_bar = None
    if LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=train_total_steps)

        wandb.init(project=os.path.basename(args.model_path), config=args)

    # Training Loop
    step = 0
    for epoch in range(args.epochs):
        _rank0_print(f"Epoch {epoch}")

        ############ Train Epoch
        model_engine.train()

        train_loader.set_epoch(epoch)
        for batch, all_numseq, cur_numseq in train_loader:
            step += 1
            if step > train_total_steps:  # At most train_total_steps
                break

            # To device
            batch = {k: (v.to(args.device) if v is not None else None) for k, v in batch.items()}

            # Update
            loss = (1 / all_numseq) * model_engine(**batch).loss

            model_engine.backward(loss)

            if model_engine.is_gradient_accumulation_boundary():
                # Set LR
                lr_this_step = args.lr * lr_scheduler(step)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step

                # Log
                if LOCAL_RANK == 0:
                    wandb.log({"loss": loss.item() * (all_numseq / cur_numseq), "lr": lr_this_step}, step=step)
                    progress_bar.update()

            model_engine.step()

        # Log batch efficiency
        if LOCAL_RANK == 0:
            wandb.log({"batch_efficiency": train_loader.efficiency()}, step=step)

        ############ Eval Epoch
        if eval_loader is not None:
            model_engine.eval()

            eval_total_loss = torch.zeros((), dtype=torch.float32, device=args.device)
            eval_total_steps = 0

            eval_loader.set_epoch(epoch)
            with torch.inference_mode():
                for batch, all_numseq, cur_numseq in eval_loader:
                    # To device
                    batch = {k: (v.to(args.device) if v is not None else None) for k, v in batch.items()}

                    # Eval
                    eval_loss = (1 / all_numseq) * model_engine(**batch).loss
                    
                    # Accumulate eval loss
                    eval_total_loss.add_(eval_loss)
                    eval_total_steps += 1

            # Gather eval loss (reduce sum)
            eval_total_loss.div_(eval_total_steps)
            torch.distributed.reduce(eval_total_loss, 0)

            if LOCAL_RANK == 0:
                wandb.log({"eval_loss": eval_total_loss.item()}, step=step)

        ############ Save Checkpoint
        # Save model with lean state dict
        # https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html
        if (epoch + 1 == args.epochs) or (args.save_every and ((epoch + 1) % args.save_every == 0)):
            torch.distributed.barrier()

            if LOCAL_RANK == 0:
                save_path = os.path.join(args.save_path, f"ep_{epoch}")

                model_engine.module.save_pretrained(save_path,
                                                    state_dict=deepspeed.checkpoint.utils.clone_tensors_for_torch_save(model_engine.module.state_dict()))

                # Also save tokenizer from base model
                save_tokenizer(args, save_path)


if __name__ == "__main__":
    train()

import argparse
import os
import json
import math
from functools import partial

import torch
import torch.distributed

import transformers
import deepspeed
import tqdm
import wandb
import numpy as np

from torch.utils.data import DataLoader
from transformers.optimization import _get_cosine_schedule_with_warmup_lr_lambda

from ochat.config.model_config import MODEL_CONFIG_MAP
from ochat.training_deepspeed.ffd_sampler import FFDDistributedBatchSampler


LOCAL_RANK      = None

PAD_ID          = 0
IGNORE_LABEL_ID = -100   # Defined in torch CrossEntropyLoss


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

    # Hyperparameters
    parser.add_argument("--batch_size_per_gpu", type=int,   default=14)
    parser.add_argument("--epochs",             type=int,   default=5)

    parser.add_argument("--lr",                 type=float, default=2e-5)
    parser.add_argument("--warmup_ratio",       type=float, default=0.03)
    parser.add_argument("--weight_decay",       type=float, default=0.)

    # DeepSpeed parameters
    parser = deepspeed.add_config_arguments(parser)

    # Parse known args
    args, unknown = parser.parse_known_args()
    return args


def create_dataset(args, split_name):
    # Load data
    with open(os.path.join(args.data_path, f"{args.model_type}.{split_name}.json"), "r") as f:
        data = json.load(f)

    return data


def batch_to_tensor(batch, dtype=torch.long):
    # Pad an unused item to reach multiple of 64, for faster GEMM
    pad_cur_len = sum([len(token_list) for (token_list, mask_list) in batch])
    pad_len     = _find_multiple(pad_cur_len, 64) - pad_cur_len

    if pad_len > 0:
        assert pad_len < 64
        batch.append([
            [PAD_ID] * pad_len,
            [False] * pad_len
        ])

    # seqlen
    batch_lengths = torch.tensor([len(token_list) for (token_list, mask_list) in batch], dtype=torch.int32, device="cpu")

    max_seqlen    = torch.max(batch_lengths)
    cu_seqlens    = torch.nn.functional.pad(batch_lengths.cumsum(-1, dtype=torch.int32), (1, 0))

    # nz elements
    nz_num               = cu_seqlens[-1]
    nz_input_ids         = torch.zeros((nz_num, ), dtype=dtype, pin_memory=True, device="cpu")
    nz_position_ids      = torch.zeros((nz_num, ), dtype=dtype, pin_memory=True, device="cpu")
    nz_shifted_label_ids = torch.zeros((nz_num, ), dtype=dtype, pin_memory=True, device="cpu")

    index = 0
    for token_list, mask_list in batch:
        length = len(token_list)

        tokens       = torch.tensor(token_list, dtype=dtype,      device="cpu")
        masks        = torch.tensor(mask_list,  dtype=torch.bool, device="cpu")
        position_ids = torch.arange(length,     dtype=dtype,      device="cpu")

        shifted_label_ids = torch.where(masks, tokens, IGNORE_LABEL_ID)
        shifted_label_ids = torch.nn.functional.pad(shifted_label_ids[1:], (0, 1), "constant", IGNORE_LABEL_ID)

        nz_input_ids[index: index + length]         = tokens
        nz_position_ids[index: index + length]      = position_ids
        nz_shifted_label_ids[index: index + length] = shifted_label_ids

        index += length

    # inputs
    return dict(max_seqlen=max_seqlen,
                cu_seqlens=cu_seqlens,
                nz_input_ids=nz_input_ids,
                nz_position_ids=nz_position_ids, 
                nz_shifted_label_ids=nz_shifted_label_ids)


def create_distributed_dataloader(args, data):
    # Sampler
    # Get length
    lengths = np.array([len(tokens) for (tokens, masks) in data])

    # FFD distributed sampler
    batch_max_len = args.batch_size_per_gpu * MODEL_CONFIG_MAP[args.model_type].model_max_context

    sampler = FFDDistributedBatchSampler(
        batch_max_length=batch_max_len,
        lengths=lengths,
        seed=0
    )

    return DataLoader(data, 
                      batch_sampler=sampler,
                      drop_last=False,
                      collate_fn=batch_to_tensor), sampler.num_batches()


def create_model(args):
    global LOCAL_RANK

    # Create model + optimizer + lr scheduler
    model = MODEL_CONFIG_MAP[args.model_type].model_create(args.model_path)
    # Model to assigned cuda device
    model = model.to(LOCAL_RANK)
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=True)

    # DeepSpeed model
    model_engine, optimizer, _, _ = deepspeed.initialize(args=args,
                                                         model=model,
                                                         model_parameters=model.parameters(),
                                                         optimizer=optimizer)

    # Put deepspeed arguments
    args.device                         = model_engine.device

    return model_engine, optimizer


def create_lr_scheduler(args, train_total_steps):
    lr_scheduler = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=math.ceil(train_total_steps * args.warmup_ratio),
        num_training_steps=train_total_steps,
        num_cycles=0.5,
    )

    return lr_scheduler


def train():
    global LOCAL_RANK

    deepspeed.init_distributed(dist_backend="nccl")

    # Args
    args       = parse_args()
    LOCAL_RANK = args.local_rank

    # Data
    _rank0_print("Loading data...")
    train_dataset = create_dataset(args, "train")
    eval_dataset  = create_dataset(args, "eval")

    # Model
    _rank0_print("Loading model...")
    model_engine, optimizer = create_model(args)

    # Data Loader
    train_loader, train_num_batches = create_distributed_dataloader(args, train_dataset)
    eval_loader,  eval_num_batches  = create_distributed_dataloader(args, eval_dataset)
    train_total_steps               = args.epochs * train_num_batches

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

        train_loader.batch_sampler.set_epoch(epoch)
        for batch in train_loader:
            step += 1
            if step > train_total_steps:  # At most train_total_steps
                break

            # To device
            batch = {k: v.to(args.device) for k, v in batch.items()}

            # Update
            loss = model_engine(**batch).loss

            model_engine.backward(loss)

            if model_engine.is_gradient_accumulation_boundary():
                # Set LR
                lr_this_step = args.lr * lr_scheduler(step)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step

                # Log
                if LOCAL_RANK == 0:
                    wandb.log({"loss": loss.item(), "lr": lr_this_step}, step=step)
                    progress_bar.update()

            model_engine.step()

        # Log batch efficiency
        if LOCAL_RANK == 0:
            wandb.log({"batch_efficiency": train_loader.batch_sampler.efficiency()}, step=step)

        ############ Eval Epoch
        model_engine.eval()

        eval_total_loss = torch.zeros((), dtype=torch.float32, device=args.device)
        eval_total_steps = 0

        eval_loader.batch_sampler.set_epoch(epoch)
        with torch.inference_mode():
            for batch in eval_loader:
                # To device
                batch = {k: v.to(args.device) for k, v in batch.items()}

                # Eval
                eval_loss = model_engine(**batch).loss
                
                # Accumulate eval loss
                eval_total_loss.add_(eval_loss)
                eval_total_steps += 1

        # Gather eval loss
        eval_total_loss.div_(eval_total_steps)
        torch.distributed.reduce(eval_total_loss, 0)

        if LOCAL_RANK == 0:
            wandb.log({"eval_loss": eval_total_loss.item() / torch.distributed.get_world_size()}, step=step)

    # Save model with lean state dict
    # https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html

    lean_state_dict = deepspeed.checkpoint.utils.clone_tensors_for_torch_save(model_engine.module.state_dict())
    model_engine.module.save_pretrained(args.save_path, state_dict=lean_state_dict)

    # Also save tokenizer from base model
    transformers.AutoTokenizer.from_pretrained(args.model_path, use_fast=False).save_pretrained(args.save_path)


if __name__ == "__main__":
    train()

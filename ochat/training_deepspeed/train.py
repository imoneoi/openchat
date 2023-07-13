import argparse
import os
import math
from functools import partial

import torch
import torch.distributed

import transformers
import datasets
import deepspeed
import tqdm
import wandb
import numpy as np

from transformers.optimization import _get_cosine_schedule_with_warmup_lr_lambda

from ochat.config.model_config import MODEL_CONFIG_MAP
from ochat.training_deepspeed.multipack_dataloader import MultipackDistributedDataloader


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
    parser.add_argument("--loss_balancing",     action="store_true", default=False)

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
    filename = f"{args.data_path}_{split_name}"
    if not os.path.isdir(filename):
        return None

    return datasets.load_dataset(filename, split="train", keep_in_memory=True)


def batch_to_tensor(batch, group_loss_weights, dtype=torch.long, loss_dtype=torch.bfloat16):
    # Pad an unused item to reach multiple of 64, for faster GEMM
    pad_cur_len = sum([item for item in batch["length"]])
    pad_len     = _find_multiple(pad_cur_len, 64) - pad_cur_len

    if pad_len > 0:
        assert pad_len < 64

        batch["tokens"].append([PAD_ID] * pad_len)
        batch["masks"].append([False] * pad_len)
        batch["group"].append(0)
        batch["length"].append(pad_len)

    # seqlen
    batch_lengths = torch.tensor([item for item in batch["length"]], dtype=torch.int32, device="cpu")

    max_seqlen    = torch.max(batch_lengths)
    cu_seqlens    = torch.nn.functional.pad(batch_lengths.cumsum(-1, dtype=torch.int32), (1, 0))

    # nz elements
    nz_num                  = cu_seqlens[-1]
    nz_input_ids            = torch.zeros((nz_num, ), dtype=dtype,      pin_memory=True, device="cpu")
    nz_position_ids         = torch.zeros((nz_num, ), dtype=dtype,      pin_memory=True, device="cpu")
    nz_shifted_label_ids    = torch.zeros((nz_num, ), dtype=dtype,      pin_memory=True, device="cpu")
    nz_shifted_loss_weights = torch.zeros((nz_num, ), dtype=loss_dtype, pin_memory=True, device="cpu")

    index = 0
    for token_list, mask_list, length, group in zip(batch["tokens"], batch["masks"], batch["length"], batch["group"]):
        tokens       = torch.tensor(token_list, dtype=dtype,      device="cpu")
        masks        = torch.tensor(mask_list,  dtype=torch.bool, device="cpu")
        position_ids = torch.arange(length,     dtype=dtype,      device="cpu")

        # Input IDs & shifted labels
        shifted_label_ids = torch.where(masks, tokens, IGNORE_LABEL_ID)
        shifted_label_ids = torch.nn.functional.pad(shifted_label_ids[1:], (0, 1), "constant", IGNORE_LABEL_ID)

        nz_input_ids[index: index + length]         = tokens
        nz_position_ids[index: index + length]      = position_ids
        nz_shifted_label_ids[index: index + length] = shifted_label_ids

        # Loss weights
        mask_count = sum(mask_list[1:])
        loss_weight = 1 / mask_count if mask_count > 0 else 0  # Avoid division by zero for paddings

        if group_loss_weights is not None:
            loss_weight *= group_loss_weights[group]

        nz_shifted_loss_weights[index: index + length] = loss_weight

        index += length

    # inputs
    return dict(max_seqlen=max_seqlen,
                cu_seqlens=cu_seqlens,
                nz_input_ids=nz_input_ids,
                nz_position_ids=nz_position_ids, 
                nz_shifted_label_ids=nz_shifted_label_ids,
                nz_shifted_loss_weights=nz_shifted_loss_weights)


def create_distributed_dataloader(args, data):
    # Sampler
    # Get length
    lengths = np.array(data["length"])

    # Loss balancing
    group_loss_weights = None
    if args.loss_balancing:
        groups = np.array(data["group"])

        unique, unique_counts = np.unique(groups, return_counts=True)
        total_count           = np.sum(unique_counts)
        group_loss_weights    = {k: total_count / c for k, c in zip(unique, unique_counts)}

        _rank0_print(f"Loss balancing enabled. Weights: {args.loss_balancing}")

    # Multipack dataloader
    batch_max_len = args.batch_size_per_gpu * MODEL_CONFIG_MAP[args.model_type].model_max_context

    collate_fn = partial(batch_to_tensor, group_loss_weights=group_loss_weights)

    return MultipackDistributedDataloader(
        dataset=data,
        lengths=lengths,

        batch_max_length=batch_max_len,
        collate_fn=collate_fn,

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
    train_loader      = create_distributed_dataloader(args, train_dataset)
    train_total_steps = args.epochs * train_loader.num_batches()

    eval_loader = None
    if eval_dataset is not None:
        eval_loader, _              = create_distributed_dataloader(args, eval_dataset)

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
        for batch, all_numseq in train_loader:
            step += 1
            if step > train_total_steps:  # At most train_total_steps
                break

            # To device
            batch = {k: v.to(args.device) for k, v in batch.items()}

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
                    wandb.log({"loss": loss.item(), "lr": lr_this_step}, step=step)
                    progress_bar.update()

            model_engine.step()

        # Log batch efficiency
        if LOCAL_RANK == 0:
            wandb.log({"batch_efficiency": train_loader.batch_sampler.efficiency()}, step=step)

        ############ Eval Epoch
        if eval_loader is not None:
            model_engine.eval()

            eval_total_loss = torch.zeros((), dtype=torch.float32, device=args.device)
            eval_total_steps = 0

            eval_loader.set_epoch(epoch)
            with torch.inference_mode():
                for batch, all_numseq in eval_loader:
                    # To device
                    batch = {k: v.to(args.device) for k, v in batch.items()}

                    # Eval
                    eval_loss = (1 / all_numseq) * model_engine(**batch).loss
                    
                    # Accumulate eval loss
                    eval_total_loss.add_(eval_loss)
                    eval_total_steps += 1

            # Gather eval loss
            eval_total_loss.div_(eval_total_steps)
            torch.distributed.reduce(eval_total_loss, 0)

            if LOCAL_RANK == 0:
                wandb.log({"eval_loss": eval_total_loss.item() / torch.distributed.get_world_size()}, step=step)

        ############ Save Checkpoint
        # Save model with lean state dict
        # https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html
        save_path = os.path.join(args.save_path, f"ep_{epoch}")

        model_engine.module.save_pretrained(save_path,
                                            state_dict=deepspeed.checkpoint.utils.clone_tensors_for_torch_save(model_engine.module.state_dict()))

        # Also save tokenizer from base model
        transformers.AutoTokenizer.from_pretrained(args.model_path, use_fast=False).save_pretrained(save_path)


if __name__ == "__main__":
    train()

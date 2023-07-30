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

from ochat.config.model_config import MODEL_CONFIG_MAP
from ochat.training_deepspeed.multipack_dataloader import MultipackDistributedDataloader
from ochat.training_deepspeed.parquet_dataset import ParquetDataset


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
    parser.add_argument("--save_every", type=int, default=None)

    # Hyperparameters
    parser.add_argument("--loss_balancing",      action="store_true", default=False)
    parser.add_argument("--no_weighted_average", action="store_true", default=False)

    parser.add_argument("--batch_size_per_gpu", type=int,   default=16)
    parser.add_argument("--epochs",             type=int,   default=5)

    # Estimated using LLaMA pretraining parameters (e.g. lr ~ sqrt(batch_size))
    parser.add_argument("--lr",                 type=float, default=4e-5)
    parser.add_argument("--lr_min_ratio",       type=float, default=0.1)
    parser.add_argument("--lr_warmup_steps",    type=int,   default=2000)

    parser.add_argument("--weight_decay",       type=float, default=0.1)

    parser.add_argument("--beta1",              type=float, default=0.9)
    parser.add_argument("--beta2",              type=float, default=0.95)
    parser.add_argument("--eps",                type=float, default=1e-8)

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


def batch_to_tensor(batch, num_groups, group_loss_weights, dtype=torch.long, loss_dtype=torch.bfloat16):
    # Pad an unused item to reach multiple of 64, for faster GEMM
    total_seqlen = sum([item for item in batch["total_length"]])
    pad_len      = _find_multiple(total_seqlen, 64) - total_seqlen

    if pad_len > 0:
        assert pad_len < 64

        # total length
        batch["total_length"].append(pad_len)

        # populate pad tokens & masks
        for group in range(num_groups - 1):
            batch[f"{group}_tokens"].append([])
            batch[f"{group}_masks"].append([])

        batch[f"{num_groups - 1}_tokens"].append([PAD_ID] * pad_len)
        batch[f"{num_groups - 1}_masks"].append([False] * pad_len)

    # nz elements
    nz_num                  = total_seqlen + pad_len
    nz_input_ids            = torch.zeros((nz_num, ), dtype=dtype,      pin_memory=True, device="cpu")
    nz_position_ids         = torch.zeros((nz_num, ), dtype=dtype,      pin_memory=True, device="cpu")
    nz_shifted_label_ids    = torch.zeros((nz_num, ), dtype=dtype,      pin_memory=True, device="cpu")
    nz_shifted_loss_weights = torch.zeros((nz_num, ), dtype=loss_dtype, pin_memory=True, device="cpu")

    seqlens                 = []

    index = 0
    for group in range(num_groups):
        for token_list, mask_list in zip(batch[f"{group}_tokens"], batch[f"{group}_masks"]):
            # calc length & skip empty
            length = len(token_list)
            if not length:
                continue

            # buffers
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

            # increment index
            seqlens.append(length)

            index += length

    # cu seqlens
    seqlens = torch.tensor(seqlens, dtype=torch.int32, device="cpu")

    max_seqlen    = torch.max(seqlens)
    cu_seqlens    = torch.nn.functional.pad(seqlens.cumsum(-1, dtype=torch.int32), (1, 0))

    # inputs
    return dict(max_seqlen=max_seqlen,
                cu_seqlens=cu_seqlens,
                nz_input_ids=nz_input_ids,
                nz_position_ids=nz_position_ids, 
                nz_shifted_label_ids=nz_shifted_label_ids,
                nz_shifted_loss_weights=nz_shifted_loss_weights)


def create_distributed_dataloader(args, data):
    # Check data
    assert data.metadata["model_type"] == args.model_type, \
        f"The dataset is for {data.metadata['model_type']}, but you specified {args.model_type} for training."

    # Sampler
    # Get length
    lengths = np.array(data["total_length"])
    numseqs = np.array(data["num_seqs"])
    num_groups = data.metadata["num_groups"]

    # Loss balancing
    group_loss_weights = None
    if args.loss_balancing:
        group_loss_weights = data.metadata["group_loss_weights"]
        if args.no_weighted_average:
            numseqs *= num_groups
        else:
            numseqs = np.array(data["total_loss_weight"])

        _rank0_print(f"Loss balancing enabled. Weights: {group_loss_weights}. No weighted average: {args.no_weighted_average}")

    # Multipack dataloader
    batch_max_len = args.batch_size_per_gpu * MODEL_CONFIG_MAP[args.model_type].model_max_context

    collate_fn = partial(batch_to_tensor,
                         num_groups=num_groups,
                         group_loss_weights=group_loss_weights)

    return MultipackDistributedDataloader(
        dataset=data,
        lengths=lengths,
        numseqs=numseqs,

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

        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=train_total_steps,
        min_ratio=args.lr_min_ratio
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
                transformers.AutoTokenizer.from_pretrained(args.model_path, use_fast=False).save_pretrained(save_path)


if __name__ == "__main__":
    train()

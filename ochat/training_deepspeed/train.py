import argparse
import os
import math
import json
from functools import partial

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

import tqdm
import wandb
import numpy as np

from ochat.config import MODEL_CONFIG_MAP
from ochat.training_deepspeed.openchat_dataset import OpenchatDataset

try:
    import deepspeed
except ImportError:
    raise ImportError("Please install deepspeed to train models.")


def parse_args():
    parser = argparse.ArgumentParser()
    # Distributed
    parser.add_argument("--local_rank", type=int, required=True)

    # Model type and data
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_prefix", type=str, required=True)
    parser.add_argument("--save_path",  type=str, required=True)
    parser.add_argument("--save_every", type=int, default=None)

    # Hyperparameters
    parser.add_argument("--batch_max_len",      type=int, default=81920)
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


def create_dataset_and_dataloader(args, split_name):
    # Find data
    filename = f"{args.data_prefix}.{split_name}.parquet"

    # Create dataset and dataloader
    print(f"Loading {split_name} data from {filename}...")

    dataset = OpenchatDataset(
        dataset_filename=filename,

        batch_max_length=args.batch_max_len,
        rank=dist.get_rank(),
        num_replicas=dist.get_world_size()
    )
    dataloader = DataLoader(
        dataset,
        batch_size=None,

        num_workers=1,
        prefetch_factor=8,

        pin_memory=True,
        persistent_workers=True
    )
    return dataset, dataloader


def create_model(args):
    print(f"Loading model {args.model_type} from {args.model_path}...")

    # Create model + optimizer + lr scheduler
    model = MODEL_CONFIG_MAP[args.model_type].model_create_for_training(args.model_path)
    # Model to assigned cuda device
    model = model.to(args.local_rank)
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(
        use_reentrant=False
    ))

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
    MODEL_CONFIG_MAP[args.model_type].model_tokenizer_create(args.model_path).save_pretrained(save_path)


def save_openchat_metadata(args, epoch, save_path):
    metadata = vars(args)
    metadata["epoch"] = epoch

    with open(os.path.join(save_path, "openchat.json"), "w") as f:
        json.dump(metadata, f, default=lambda o: "<non-serializable>")


def calculate_auto_lr(lr, batch_max_len, model_type, train_dataset):
    if lr is not None:
        return lr
    
    # Llama hyperparameters
    # FIXME: Only 7B/13B is supported
    base_lr = 3e-4
    base_bs = 4_000_000
    if "mistral" in model_type.lower():
        base_lr /= 6.0

    loss_weights = np.concatenate(train_dataset.dataset["nz_shifted_loss_weights"])
    supervised_ratio = np.sum(loss_weights != 0) / len(loss_weights)

    supervised_tokens = batch_max_len * dist.get_world_size() * supervised_ratio
    lr = base_lr * math.sqrt(supervised_tokens / base_bs)

    print(f"Use automatic learning rate {lr} (estimated from supervised ratio {supervised_ratio} effective batch size {supervised_tokens})")
    return lr


def state_dict_to_cpu(item, device=torch.device('cpu')):
    # Move all tensors to CPU
    if torch.is_tensor(item):
        return item.detach().to(device)
    elif isinstance(item, list):
        return [state_dict_to_cpu(v, device) for v in item]
    elif isinstance(item, tuple):
        return tuple([state_dict_to_cpu(v, device) for v in item])
    elif isinstance(item, dict):
        return type(item)({k: state_dict_to_cpu(v, device) for k, v in item.items()})
    else:
        return item


def train():
    deepspeed.init_distributed(dist_backend="nccl")
    RANK = dist.get_rank()

    # Args
    args       = parse_args()

    # Dataset
    train_dataset, train_loader = create_dataset_and_dataloader(args, "train")

    if train_dataset is None:
        raise RuntimeError("Training data not found.")

    # Load model type
    args.model_type = train_dataset.metadata["model_type"]

    train_total_steps = args.epochs * train_dataset.estimate_num_batches()

    # Hyperparams
    args.lr = calculate_auto_lr(args.lr, args.batch_max_len, args.model_type, train_dataset)

    # Model
    model_engine, optimizer = create_model(args)

    # LR Scheduler
    lr_scheduler = create_lr_scheduler(args, train_total_steps)

    # Progress bar and logger
    progress_bar = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_total_steps)

        wandb.init(project=os.path.basename(args.model_path), config=args)

    # Training Loop
    step = 0
    lr_this_step = None
    for epoch in range(args.epochs):
        print (f"[rank {RANK}]: Epoch {epoch}")

        ############ Train Epoch
        model_engine.train()
        for (batch_tensor, batch_info), all_numseq, cur_numseq in train_loader:
            step += 1
            if step > train_total_steps:  # At most train_total_steps
                break

            # To device
            batch_tensor = {k: (v.to(args.device) if v is not None else None) for k, v in batch_tensor.items()}

            # Update
            loss, acc = model_engine(**batch_tensor, **batch_info).loss
            loss = (1 / all_numseq) * loss
            acc  = (1 / all_numseq) * acc

            model_engine.backward(loss)

            if model_engine.is_gradient_accumulation_boundary():
                # Set LR
                lr_this_step = args.lr * lr_scheduler(step)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step

            model_engine.step()

            # Logging
            if RANK == 0:
                wandb.log({
                    "train/loss": loss.item() * (all_numseq / cur_numseq),
                    "train/acc":  acc.item()  * (all_numseq / cur_numseq),
                    "train/lr": lr_this_step
                }, step=step)
                progress_bar.update()  # type: ignore

        ############ Save Checkpoint
        # Save model with lean state dict
        # https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html
        if (epoch + 1 == args.epochs) or (args.save_every and ((epoch + 1) % args.save_every == 0)):
            if RANK == 0:
                save_path = os.path.join(args.save_path, f"ep_{epoch}")

                model_engine.module.save_pretrained(save_path,
                                                    state_dict=state_dict_to_cpu(model_engine.module.state_dict()))  # type: ignore

                # Also save tokenizer from base model
                save_tokenizer(args, save_path)

                # Write metadata
                save_openchat_metadata(args, epoch, save_path)


if __name__ == "__main__":
    train()

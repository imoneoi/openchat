import ochat.patches.apply  # Apply attention patches

import argparse
import os
import json
import math

import torch
import torch.distributed

import transformers
import deepspeed
import tqdm
import wandb

from torch.utils.data import DataLoader
from transformers.trainer_pt_utils import DistributedLengthGroupedSampler, DistributedSampler
from transformers.optimization import get_cosine_schedule_with_warmup


LOCAL_RANK      = None

PAD_TOKEN_ID    = 0      # <unk> in LLaMA
IGNORE_LABEL_ID = -100   # Defined in torch CrossEntropyLoss


def _ceildiv(a, b):
    return -(a // -b)


def _rank0_print(*args):
    global LOCAL_RANK

    if LOCAL_RANK == 0:
        print(*args)


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
    parser.add_argument("--length_grouping",  default=False, action="store_true")

    parser.add_argument("--epochs",           type=int,   default=10)

    parser.add_argument("--lr",               type=float, default=2e-5)
    parser.add_argument("--warmup_ratio",     type=float, default=0.03)
    parser.add_argument("--weight_decay",     type=float, default=0.)

    # DeepSpeed parameters
    parser = deepspeed.add_config_arguments(parser)

    # Parse known args
    args, unknown = parser.parse_known_args()
    return args


def create_dataset(args, split_name):
    # Load data
    with open(os.path.join(args.data_path, f"{args.model_type}.{split_name}.json"), "r") as f:
        data = json.load(f)

    return data, len(data)


def batch_to_tensor(batch, dtype=torch.long):
    batch_size = len(batch)
    max_length = max([len(tokens) for (tokens, masks) in batch])

    # create batch
    input_ids      = torch.full((batch_size, max_length), PAD_TOKEN_ID, dtype=dtype,      pin_memory=True, device="cpu")
    label_mask     = torch.full((batch_size, max_length), False,        dtype=torch.bool, pin_memory=True, device="cpu")
    attention_mask = torch.full((batch_size, max_length), False,        dtype=torch.bool, pin_memory=True, device="cpu")

    for idx, (tokens, masks) in enumerate(batch):
        length = len(tokens)

        input_ids[idx, :length]      = torch.tensor(tokens, dtype=dtype,     device="cpu")
        label_mask[idx, :length]     = torch.tensor(masks, dtype=torch.bool, device="cpu")
        attention_mask[idx, :length] = True

    # create labels
    labels = torch.where(label_mask, input_ids, IGNORE_LABEL_ID).pin_memory()

    return dict(input_ids=input_ids, labels=labels, attention_mask=attention_mask)


def create_distributed_dataloader(args, data, length_grouping):
    # Sampler
    if length_grouping:
        _rank0_print ("Length grouping enabled !")

        # Get length
        lengths = [len(tokens) for (tokens, masks) in data]
        # Length grouped sampler
        sampler = DistributedLengthGroupedSampler(
            batch_size=args.train_micro_batch_size_per_gpu,
            dataset=data,
            lengths=lengths,
            drop_last=False,
            seed=0
        )
    else:
        sampler = DistributedSampler(
            dataset=data,
            drop_last=False,
            seed=0
        )

    return DataLoader(data, 
                      batch_size=args.train_micro_batch_size_per_gpu,
                      sampler=sampler,
                      drop_last=False,
                      collate_fn=batch_to_tensor)


def create_model(args, train_dataset_size):
    global LOCAL_RANK

    # Get train total steps
    with open(args.deepspeed_config, "r") as f:
        train_batch_size = json.load(f)["train_batch_size"]

    train_total_steps = _ceildiv(train_dataset_size, train_batch_size) * args.epochs

    # Create model + optimizer + lr scheduler
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model_path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
    # Model to assigned cuda device
    model = model.to(LOCAL_RANK)
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=True)

    # LR Scheduler
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                   num_warmup_steps=math.ceil(train_total_steps * args.warmup_ratio),
                                                   num_training_steps=train_total_steps)

    # DeepSpeed model
    model_engine, _, _, lr_scheduler = deepspeed.initialize(args=args,
                                                            model=model,
                                                            model_parameters=model.parameters(),
                                                            optimizer=optimizer,
                                                            lr_scheduler=lr_scheduler)

    # Put deepspeed arguments
    args.train_batch_size               = model_engine.train_batch_size()
    args.train_micro_batch_size_per_gpu = model_engine.train_micro_batch_size_per_gpu()
    args.device                         = model_engine.device

    args.train_total_steps              = train_total_steps

    return model_engine, lr_scheduler


def train():
    global LOCAL_RANK

    deepspeed.init_distributed(dist_backend="nccl")

    # Args
    args       = parse_args()
    LOCAL_RANK = args.local_rank

    # Data
    _rank0_print("Loading data...")
    train_dataset, train_dataset_size = create_dataset(args, "train")
    eval_dataset,  eval_dataset_size  = create_dataset(args, "eval")

    # Model
    _rank0_print("Loading model...")
    model_engine, lr_scheduler = create_model(args, train_dataset_size)

    # Data Loader
    train_loader = create_distributed_dataloader(args, train_dataset, args.length_grouping)
    eval_loader  = create_distributed_dataloader(args, eval_dataset, args.length_grouping)

    # Progress bar and logger
    progress_bar = None
    if LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=args.train_total_steps)

        wandb.init(project=os.path.basename(args.model_path), config=args)

    # Training Loop
    step = 0
    for epoch in range(args.epochs):
        _rank0_print(f"Epoch {epoch}")

        ############ Train Epoch
        model_engine.train()

        train_loader.sampler.set_epoch(epoch)
        for batch in train_loader:
            # To device
            batch = {k: v.to(args.device) for k, v in batch.items()}

            # Update
            loss = model_engine(**batch,
                                use_cache=False, output_attentions=False, output_hidden_states=False).loss

            model_engine.backward(loss)

            # Log
            if model_engine.is_gradient_accumulation_boundary() and (LOCAL_RANK == 0):
                wandb.log({
                    "loss": loss.item(),
                    "lr":   lr_scheduler.get_last_lr()[0]
                }, step=step)

                step += 1
                progress_bar.update()

            # Step optimizer
            model_engine.step()

        ############ Eval Epoch
        model_engine.eval()

        eval_total_loss = torch.zeros((), dtype=torch.float32, device=args.device)
        eval_total_steps = 0

        eval_loader.sampler.set_epoch(epoch)
        with torch.no_grad():
            for batch in eval_loader:
                # To device
                batch = {k: v.to(args.device) for k, v in batch.items()}

                # Eval
                eval_loss = model_engine(**batch,
                                         use_cache=False, output_attentions=False, output_hidden_states=False).loss
                
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


if __name__ == "__main__":
    train()

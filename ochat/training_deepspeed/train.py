import argparse
import os
import json

import torch

import transformers
import deepspeed
import tqdm
import wandb

from torch.utils.data import DataLoader
from transformers.trainer_pt_utils import DistributedLengthGroupedSampler
from transformers.optimization import get_cosine_schedule_with_warmup

from torch.distributed import reduce, ReduceOp


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
    # Model type and data
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path",  type=str, required=True)

    parser.add_argument("--epochs",     type=int, default=10)

    parser.add_argument("--total_batch_size", type=int,   default=128)
    parser.add_argument("--lr",               type=float, default=2e-5)
    parser.add_argument("--warmup_ratio",     type=float, default=0.03)
    parser.add_argument("--weight_decay",     type=float, default=0.)

    # DeepSpeed parameters
    parser = deepspeed.add_config_arguments(parser)

    return parser.parse_args()


def batch_to_tensor(batch, dtype=torch.int32):
    batch_size = len(batch)
    max_length = max([len(tokens) for (tokens, masks) in batch])

    # create batch
    input_ids       = torch.full((batch_size, max_length), PAD_TOKEN_ID, dtype=dtype,      pin_memory=True, device="cpu")
    label_masks     = torch.full((batch_size, max_length), False,        dtype=torch.bool, pin_memory=True, device="cpu")
    attention_masks = torch.full((batch_size, max_length), False,        dtype=torch.bool, pin_memory=True, device="cpu")

    for idx, (tokens, masks) in enumerate(batch):
        length = len(tokens)

        input_ids[idx, :length]       = torch.tensor(tokens, dtype=dtype,     device="cpu")
        label_masks[idx, :length]     = torch.tensor(masks, dtype=torch.bool, device="cpu")
        attention_masks[idx, :length] = True

    # create labels
    labels = torch.where(label_masks, input_ids, IGNORE_LABEL_ID).pin_memory()

    return dict(input_ids=input_ids, labels=labels, attention_masks=attention_masks)


def create_distributed_dataloader(local_rank, world_size, total_batch_size, data_path, split_name):
    # Load data
    with open(os.path.join(data_path, f"ochat.{split_name}.npz"), "r") as f:
        data = json.load(f)

    # Get length
    lengths = [len(tokens) for (tokens, masks) in data]
    # Sampler
    sampler = DistributedLengthGroupedSampler(
        batch_size=total_batch_size,
        dataset=data,
        num_replicas=world_size,
        rank=local_rank,
        lengths=lengths,
        drop_last=False
    )

    # Total steps
    total_steps = _ceildiv(len(data), total_batch_size)

    return DataLoader(data, 
                      batch_size=total_batch_size,
                      sampler=sampler,
                      drop_last=False,
                      collate_fn=batch_to_tensor), total_steps


def create_model(args, train_total_steps):
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model_path)

    optimizer = deepspeed.ops.adam.FusedAdam(model.parameters(),
                                             lr=args.lr,
                                             weight_decay=args.weight_decay)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                   num_warmup_steps=round(train_total_steps * args.warmup_ratio),
                                                   num_training_steps=train_total_steps)

    model_engine, _, _, lr_scheduler = deepspeed.initialize(args=args,
                                                            model=model,
                                                            model_parameters=model.parameters(),
                                                            optimizer=optimizer,
                                                            lr_scheduler=lr_scheduler)

    return model_engine, lr_scheduler


def train():
    global LOCAL_RANK

    # Args
    args       = parse_args()
    LOCAL_RANK = args.local_rank

    # Data
    _rank0_print("Loading data...")
    train_loader, train_total_steps = create_distributed_dataloader(args.local_rank, args.world_size, args.total_batch_size, args.data_path, "train")
    eval_loader,  eval_total_steps  = create_distributed_dataloader(args.local_rank, args.world_size, args.total_batch_size, args.data_path, "eval")

    # Model
    _rank0_print("Loading model...")
    model_engine, lr_scheduler = create_model(args, train_total_steps)

    # Progress bar and logger
    progress_bar = None
    if LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=train_total_steps)

        wandb.init(project=os.path.basename(args.model_path), config=args)

    # Training Loop
    model_engine.train()

    step = 0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch}")

        # Epoch
        train_loader.sampler.set_epoch(epoch)
        for batch in train_loader:
            # To device
            batch = {k: v.cuda() for k, v in batch.items()}

            # Update
            loss = model_engine(**batch,
                                use_cache=False, output_attentions=False, output_hidden_states=False).loss

            model_engine.backward(loss)
            model_engine.step()

            # Gather loss
            loss_cpu = loss.detach().cpu()
            reduce(loss_cpu, dst=0, op=ReduceOp.SUM)

            # Log
            if progress_bar is not None:
                step += 1
                progress_bar.update()

                wandb.log({
                    "loss": loss_cpu / args.world_size,
                    "lr":   lr_scheduler.get_lr()
                }, step=step)



if __name__ == "__main__":
    train()

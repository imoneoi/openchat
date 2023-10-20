"""
Generate training data based on conversations

Usage: python -m ochat.data.generate_data --in-file sharegpt_gpt4.jsonl --tokenizer-name HF_REPO_NAME --out-dir .
"""

from typing import Optional
import argparse
import os
import random

import ray
import orjson
import pyarrow
from pyarrow import parquet


PAD_TOKEN_ID = 0


def _split(a, n):
    # Split list a to n chunks
    # https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m): (i+1)*k+min(i+1, m)] for i in range(n)]


def truncate_trailing_zero_weighted(tokens, weights):
    non_zero_index = len(weights) - 1
    while non_zero_index >= 0 and weights[non_zero_index] == 0:
        non_zero_index -= 1

    return tokens[:non_zero_index + 1], weights[:non_zero_index + 1]


def add_single_conv(output, tokens, weights):
    # truncate trailing zero weighted tokens
    tokens, weights = truncate_trailing_zero_weighted(tokens, weights)
    if not tokens:
        return

    # labels
    length = len(tokens)
    labels = [(t if w != 0 else PAD_TOKEN_ID) for t, w in zip(tokens, weights)]

    # populate results
    results = {
        "total_length": length,

        "seqlens": [length],
        "nz_input_ids": tokens,
        "nz_position_ids": list(range(length)),

        "nz_shifted_label_ids":    labels[1:]  + [PAD_TOKEN_ID],
        "nz_shifted_loss_weights": weights[1:] + [0.0]
    }
    results["num_seqs"] = sum(results["nz_shifted_loss_weights"])

    for k, v in results.items():
        output[k].append(v)


@ray.remote
def convert_conversation_batch(model_type: str, model_path: str, batch: list, schema: pyarrow.Schema, per_sequence_loss: bool):
    from ochat.config import MODEL_CONFIG_MAP, Conversation

    # Tokenization
    model_config = MODEL_CONFIG_MAP[model_type]
    tokenizer = model_config.model_tokenizer_create(model_path)
    conv_template = model_config.conversation_template(tokenizer=tokenizer)

    # Decode data
    print ("Decoding JSON ...")
    batch = [Conversation(**orjson.loads(json_line)) for json_line in batch]

    # Tokenize
    print ("Tokenizing ...")
    tokens_list, weights_list = conv_template.tokenize_conversations(batch, inference=False, seq_level_weight=per_sequence_loss)

    # Generate data
    print ("Generating ...")
    max_context = model_config.model_max_context

    outputs = {k: [] for k in schema.names}
    for tokens, weights in zip(tokens_list, weights_list):
        assert len(tokens) == len(weights)

        # Truncate to specified tokens
        tokens  = tokens[:max_context]
        weights = weights[:max_context]

        # Add to results
        add_single_conv(outputs, tokens, weights)

    print ("Chunk finish")

    return pyarrow.Table.from_pydict(outputs, schema=schema)


def generate_split(model_type: str, model_path: str, conversations: list, split_name: str, out_prefix: str, per_sequence_loss: bool):
    # schema
    metadata = {
        "model_type": model_type
    }
    schema = [
        pyarrow.field("total_length", pyarrow.int32()),
        pyarrow.field("num_seqs", pyarrow.float32()),

        pyarrow.field(f"seqlens", pyarrow.list_(pyarrow.int32())),
        pyarrow.field(f"nz_input_ids", pyarrow.list_(pyarrow.int32())),
        pyarrow.field(f"nz_position_ids", pyarrow.list_(pyarrow.int32())),
        pyarrow.field(f"nz_shifted_label_ids", pyarrow.list_(pyarrow.int32())),
        pyarrow.field(f"nz_shifted_loss_weights", pyarrow.list_(pyarrow.float32()))
    ]

    schema = pyarrow.schema(schema, metadata={"metadata_json": orjson.dumps(metadata)})

    # launch remote workers
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=os.cpu_count())

    handles = [convert_conversation_batch.remote(
        model_type=model_type,  # type: ignore
        model_path=model_path,
        batch=batch,
        schema=schema,
        per_sequence_loss=per_sequence_loss
    ) for batch in _split(conversations, int(ray.available_resources()["CPU"]))]

    # write
    parquet.write_table(pyarrow.concat_tables([ray.get(handle) for handle in handles]), f"{out_prefix}.{split_name}.parquet")


def generate_dataset(model_type, model_path, in_files, out_prefix, per_sequence_loss, seed, eval_ratio):
    # Load conversations
    conversations = []
    for filename in in_files:
        with open(filename, "rt") as f:
            conversations.extend(f.readlines())

    # Train-test split
    random.seed(seed)
    random.shuffle(conversations)
    eval_num = int(eval_ratio * len(conversations))

    train_conversations = conversations[eval_num:]
    eval_conversations  = conversations[:eval_num]

    generate_split(model_type, model_path, train_conversations, "train", out_prefix, per_sequence_loss)
    if eval_num > 0:
        generate_split(model_type, model_path, eval_conversations, "eval", out_prefix, per_sequence_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)

    parser.add_argument("--in-files", type=str, nargs="+", required=True)
    parser.add_argument("--out-prefix", type=str, required=True)

    parser.add_argument("--per-sequence-loss", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-ratio", type=float, default=0.005)
    args = parser.parse_args()

    generate_dataset(**vars(args))

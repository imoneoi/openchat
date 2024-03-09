"""
Generate training data based on conversations

Usage: python -m ochat.data.generate_data --in-file sharegpt_gpt4.jsonl --tokenizer-name HF_REPO_NAME --out-dir .
"""

import argparse
import os
import gc
import random

import orjson
import pyarrow
import ray
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

    del batch
    gc.collect()

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

    del tokens_list, weights_list
    gc.collect()

    print ("To table ...")
    table = pyarrow.Table.from_pydict(outputs, schema=schema)

    del outputs
    gc.collect()

    print ("Chunk finish")
    return table


def generate_epoch(seed: int, model_type: str, model_path: str, in_filename: str, out_filename: str, per_sequence_loss: bool):
    # schema
    metadata = {
        "model_type": model_type
    }
    schema = [
        pyarrow.field("total_length", pyarrow.int32()),
        pyarrow.field("num_seqs", pyarrow.float32()),

        pyarrow.field("seqlens", pyarrow.list_(pyarrow.int32())),
        pyarrow.field("nz_input_ids", pyarrow.list_(pyarrow.int32())),
        pyarrow.field("nz_position_ids", pyarrow.list_(pyarrow.int32())),
        pyarrow.field("nz_shifted_label_ids", pyarrow.list_(pyarrow.int32())),
        pyarrow.field("nz_shifted_loss_weights", pyarrow.list_(pyarrow.float32()))
    ]

    schema = pyarrow.schema(schema, metadata={"metadata_json": orjson.dumps(metadata)})

    # Load data
    with open(in_filename, "rb") as f:
        batches = f.readlines()

        random.seed(seed)  # Randomized load balancing
        random.shuffle(batches)

        batches = _split(batches, int(ray.available_resources()["CPU"]))

    # launch remote workers
    handles = [convert_conversation_batch.remote(
        model_type=model_type,  # type: ignore
        model_path=model_path,
        batch=batch,
        schema=schema,
        per_sequence_loss=per_sequence_loss
    ) for batch in batches]

    # write
    parquet.write_table(pyarrow.concat_tables([ray.get(handle) for handle in handles]), out_filename)


def generate_dataset(model_type, model_path, in_prefix, out_prefix, per_sequence_loss, seed):
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=os.cpu_count())

    # Load epochs and tokenize
    epoch = 0
    while True:
        in_filename = f"{in_prefix}.{epoch}.jsonl"
        if not os.path.exists(in_filename):
            break

        out_filename = f"{out_prefix}.{epoch}.parquet"
        generate_epoch(
            seed=seed + epoch,
            model_type=model_type,
            model_path=model_path,
            in_filename=in_filename,
            out_filename=out_filename,
            per_sequence_loss=per_sequence_loss
        )
        gc.collect()

        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)

    parser.add_argument("--in-prefix", type=str, required=True)
    parser.add_argument("--out-prefix", type=str, required=True)

    parser.add_argument("--per-sequence-loss", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_dataset(**vars(args))

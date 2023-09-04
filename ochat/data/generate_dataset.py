"""
Generate training data based on conversations

Usage: python -m ochat.data.generate_data --in-file sharegpt_gpt4.json --tokenizer-name HF_REPO_NAME --out-dir .
"""

import argparse
import os
import random

import orjson
import ray
import pyarrow
from pyarrow import parquet

from ochat.data.unwanted_words import contains_unwanted_words


IGNORE_TOKEN_ID = -100


def _split(a, n):
    # Split list a to n chunks
    # https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m): (i+1)*k+min(i+1, m)] for i in range(n)]


def conversation_properties(c, gpt3_weight):
    is_gpt4 = c.get("model", "") == "Model: GPT-4"
    return {
        "is_gpt4": is_gpt4,
        "weight": 1.0 if is_gpt4 else gpt3_weight
    }


def add_single_conv(output, tokens, masks, weights):
    # sanity check
    for m, w in zip(masks, weights):
        assert m == (w != 0)

    # length
    length = len(tokens)
    assert len(masks) == length
    assert len(weights) == length

    if sum(masks) == 0:
        return

    # labels
    labels = [(t if m else IGNORE_TOKEN_ID) for t, m in zip(tokens, masks)]

    # populate results
    results = {
        "total_length": length,

        "seqlens": [length],
        "nz_input_ids": tokens,
        "nz_position_ids": list(range(length)),

        "nz_shifted_label_ids":    labels[1:]  + [IGNORE_TOKEN_ID],
        "nz_shifted_loss_weights": weights[1:] + [0.0]
    }
    results["num_seqs"] = sum(results["nz_shifted_loss_weights"])

    for k, v in results.items():
        output[k].append(v)


@ray.remote
def convert_conversation_batch(model_type: str, model_path: str, batch: list, field_names: list, gpt3_weight: float):
    from ochat.config.model_config import MODEL_CONFIG_MAP

    # Tokenization
    model_config = MODEL_CONFIG_MAP[model_type]
    tokenizer = model_config.model_tokenizer_create(model_path)

    def _tokenize(text):
        """Tokenize text-only, ignoring all special tokens."""
        return tokenizer.convert_tokens_to_ids(tokenizer._tokenize(text))

    def _tokenize_special(special_name):
        return tokenizer.convert_tokens_to_ids(special_name)

    # Generate data
    outputs = {k: [] for k in field_names}
    for c in batch:
        props = conversation_properties(c, gpt3_weight)
        message_list = c["items"]
        for msg in message_list:
            assert msg["from"] in {"human", "gpt"}

            if msg["from"] == "gpt":
                msg["use_loss"] = not contains_unwanted_words(msg["value"])

        # Generate template
        tokens, masks, weights = model_config.generate_conversation_template(_tokenize, _tokenize_special,
                                                                             system_prompt=c.get("system_message", ""),
                                                                             message_list=message_list,
                                                                             message_props=props)

        # Truncate to specified tokens
        max_context = model_config.model_max_context
        if max_context is not None:
            tokens  = tokens[:max_context]
            masks   = masks[:max_context]
            weights = weights[:max_context]

        # Add to results
        add_single_conv(outputs, tokens, masks, weights)

    return outputs


def generate_split(model_type: str, model_path: str, conversations: list, split_name: str, out_prefix: str, gpt3_weight: float, num_cpus: int = os.cpu_count()):
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
    ray.init(num_cpus=num_cpus)

    handles = [convert_conversation_batch.remote(
        model_type=model_type,
        model_path=model_path,
        batch=batch,
        field_names=schema.names,
        gpt3_weight=gpt3_weight
    ) for batch in _split(conversations, num_cpus)]

    # aggegrate results
    results = {k: [] for k in schema.names}
    for handle in handles:
        batch_result = ray.get(handle)

        for k, v in batch_result.items():
            results[k].extend(v)

    # write
    parquet.write_table(pyarrow.Table.from_pydict(results, schema=schema), f"{out_prefix}.{split_name}.parquet")

    ray.shutdown()


def generate_dataset(model_type, model_path, in_files, out_prefix, gpt3_weight, seed, eval_ratio):
    # Load conversations
    conversations = []
    for filename in in_files:
        with open(filename, "rb") as f:
            conversations.extend(orjson.loads(f.read()))

    # Train-test split
    random.seed(seed)
    random.shuffle(conversations)
    eval_num = int(eval_ratio * len(conversations))

    train_conversations = conversations[eval_num:]
    eval_conversations  = conversations[:eval_num]

    generate_split(model_type, model_path, train_conversations, "train", out_prefix, gpt3_weight)
    if eval_num > 0:
        generate_split(model_type, model_path, eval_conversations, "eval", out_prefix, gpt3_weight)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)

    parser.add_argument("--in-files", type=str, nargs="+", required=True)
    parser.add_argument("--out-prefix", type=str,required=True)

    parser.add_argument("--gpt3_weight", type=float, default=0.1)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-ratio", type=float, default=0.0)
    args = parser.parse_args()

    generate_dataset(**vars(args))

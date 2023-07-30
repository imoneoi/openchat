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


def _split(a, n):
    # Split list a to n chunks
    # https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m): (i+1)*k+min(i+1, m)] for i in range(n)]


def conversation_properties(c):
    return {
        "is_gpt4": c.get("model", "") == "Model: GPT-4"
    }


@ray.remote
def convert_conversation_batch(model_type: str, model_path: str, batch: list, field_names: list):
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
    results = {k: [] for k in field_names}
    for c in batch:
        props = conversation_properties(c)

        # Generate template
        tokens, masks, group = model_config.generate_conversation_template(_tokenize, _tokenize_special,
                                                                           system_prompt="",
                                                                           message_list=c["items"],
                                                                           message_props=props)

        # Truncate to specified tokens
        max_context = model_config.model_max_context
        if max_context is not None:
            tokens = tokens[:max_context]
            masks  = masks[:max_context]

        # Add to results
        if sum(masks) > 0:
            item = {
                f"{group}_tokens": tokens,
                f"{group}_masks": masks,
                "total_length": len(tokens),
                "num_seqs": 1
            }

            for k in field_names:
                results[k].append(item.get(k, []))

    return results


def calculate_weights(results: dict, num_groups: int):
    n = len(results["total_length"])

    # Group loss weight
    group_freq = [sum([bool(item) for item in results[f"{group}_tokens"]])
                  for group in range(num_groups)]
    group_loss_weights = [n / freq if freq > 0 else 0
                          for freq in group_freq]

    # Total loss weight
    total_loss_weight = [sum([(group_loss_weights[group] if results[f"{group}_tokens"][idx] else 0.) for group in range(num_groups)])
                         for idx in range(n)]

    return group_loss_weights, total_loss_weight


def generate_split(model_type: str, model_path: str, conversations: list, split_name: str, out_prefix: str, num_cpus: int = os.cpu_count()):
    from ochat.config.model_config import MODEL_CONFIG_MAP

    # schema
    num_groups = MODEL_CONFIG_MAP[model_type].num_groups

    schema = [
        pyarrow.field("total_length", pyarrow.int32()),
        pyarrow.field("num_seqs", pyarrow.int32()),
    ]
    for group in range(num_groups):
        schema.extend([
            pyarrow.field(f"{group}_tokens", pyarrow.list_(pyarrow.int32())),
            pyarrow.field(f"{group}_masks",  pyarrow.list_(pyarrow.bool_())),
        ])

    schema = pyarrow.schema(schema)

    # launch remote workers
    ray.init(num_cpus=num_cpus)

    handles = [convert_conversation_batch.remote(
        model_type=model_type,
        model_path=model_path,
        batch=batch,
        field_names=schema.names
    ) for batch in _split(conversations, num_cpus)]

    # aggegrate results
    results = {k: [] for k in schema.names}
    for handle in handles:
        batch_result = ray.get(handle)

        for k, v in batch_result.items():
            results[k].extend(v)

    # weights
    group_loss_weights, results["total_loss_weight"] = calculate_weights(results, num_groups)
    schema = schema.append(pyarrow.field(f"total_loss_weight", pyarrow.float32()))

    # metadata & write
    metadata = {
        "model_type": model_type,
        "group_loss_weights": group_loss_weights,
        "num_groups": num_groups,
    }
    schema = schema.with_metadata({"metadata_json": orjson.dumps(metadata)})

    parquet.write_table(pyarrow.Table.from_pydict(results, schema=schema), f"{out_prefix}.{split_name}.parquet")

    ray.shutdown()


def generate_dataset(model_type, model_path, in_file, out_prefix, seed, eval_ratio):
    # Load conversations
    with open(in_file, "r") as f:
        conversations = orjson.loads(f.read())

    # Train-test split
    random.seed(seed)
    random.shuffle(conversations)
    eval_num = int(eval_ratio * len(conversations))

    train_conversations = conversations[eval_num:]
    eval_conversations  = conversations[:eval_num]

    generate_split(model_type, model_path, train_conversations, "train", out_prefix)
    if eval_num > 0:
        generate_split(model_type, model_path, eval_conversations, "eval", out_prefix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)

    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--out-prefix", type=str,required=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-ratio", type=float, default=0.0)
    args = parser.parse_args()

    generate_dataset(**vars(args))

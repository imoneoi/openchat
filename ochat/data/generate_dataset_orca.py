import argparse
import os
import random
import json

import ray
import pyarrow
from pyarrow import parquet


GROUPS = {
    "answer_gpt35": {"id": 0, "props": {"is_gpt4": False}},
    "answer_gpt4":  {"id": 1, "props": {"is_gpt4": True}}
}


def _split(a, n):
    # Split list a to n chunks
    # https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m): (i+1)*k+min(i+1, m)] for i in range(n)]


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
    for item in batch:
        # Convert item
        item_converted = {"total_length": 0, "num_seqs": 0}

        for group_name, group_info in GROUPS.items():
            group_id = group_info["id"]
            props    = group_info["props"]
            answer   = item[group_name]

            if not len(answer):
                continue

            conversation = [
                {"from": "human", "value": item["question"]},
                {"from": "gpt",   "value": answer}
            ]

            # Generate template
            tokens, masks, _     = model_config.generate_conversation_template(_tokenize, _tokenize_special,
                                                                               system_prompt=item["system_prompt"],
                                                                               message_list=conversation,
                                                                               message_props=props)

            # Truncate to specified tokens
            max_context = model_config.model_max_context
            if max_context is not None:
                tokens = tokens[:max_context]
                masks  = masks[:max_context]

            if sum(masks) > 0:
                item_converted[f"{group_id}_tokens"] = tokens
                item_converted[f"{group_id}_masks"] = masks

                item_converted["total_length"] += len(tokens)
                item_converted["num_seqs"] += 1

        # Append to result
        if item_converted["num_seqs"] >= 1:
            for k in field_names:
                results[k].append(item_converted.get(k, []))

    return results


def calculate_weights(results: dict):
    n = len(results["total_length"])

    # Group loss weight
    group_freq = [sum([bool(item) for item in results[f"{group}_tokens"]])
                  for group in range(len(GROUPS))]
    group_loss_weights = [n / freq if freq > 0 else 0
                          for freq in group_freq]

    # Total loss weight
    total_loss_weight = [sum([(group_loss_weights[group] if results[f"{group}_tokens"][idx] else 0.) for group in range(len(GROUPS))])
                         for idx in range(n)]

    return group_loss_weights, total_loss_weight


def generate_split(model_type: str, model_path: str, conversations: list, split_name: str, out_prefix: str, num_cpus: int = os.cpu_count()):
    # schema
    schema = [
        pyarrow.field("total_length", pyarrow.int32()),
        pyarrow.field("num_seqs", pyarrow.int32()),
    ]
    for group in range(len(GROUPS)):
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
    group_loss_weights, results["total_loss_weight"] = calculate_weights(results)
    schema = schema.append(pyarrow.field(f"total_loss_weight", pyarrow.float32()))

    # metadata & write
    metadata = {
        "model_type": model_type,
        "group_loss_weights": group_loss_weights,
        "num_groups": len(GROUPS),
    }
    schema = schema.with_metadata({"metadata_json": json.dumps(metadata)})

    parquet.write_table(pyarrow.Table.from_pydict(results, schema=schema), f"{out_prefix}.{split_name}.parquet")

    ray.shutdown()


def generate_dataset(model_type, model_path, in_file, out_prefix, seed, eval_ratio):
    # Load conversations
    conversations = parquet.read_table(in_file).to_pylist()

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
    parser.add_argument("--out-prefix", type=str, default=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-ratio", type=float, default=0.0)
    args = parser.parse_args()

    generate_dataset(**vars(args))

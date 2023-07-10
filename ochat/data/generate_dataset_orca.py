import argparse
import os
import random

import ray
import pyarrow
from pyarrow import parquet
import datasets


def _split(a, n):
    # Split list a to n chunks
    # https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m): (i+1)*k+min(i+1, m)] for i in range(n)]


def conversation_properties(model):
    return {
        "is_gpt4": model == "gpt4"
    }


@ray.remote
def convert_conversation_batch(model_type: str, model_path: str, batch: list):
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
    results = {"tokens": [], "masks": [], "group": [], "length": []}
    for item in batch:
        conversation = [
            {"from": "human", "value": item["question"]},
            {"from": "gpt",   "value": item["response"]}
        ]
        props = conversation_properties(item["model"])

        # Generate template
        tokens, masks, group = model_config.generate_conversation_template(_tokenize, _tokenize_special, item["system_prompt"], conversation, props)

        # Truncate to specified tokens
        max_context = model_config.model_max_context
        if max_context is not None:
            tokens = tokens[:max_context]
            masks  = masks[:max_context]

        results["tokens"].append(tokens)
        results["masks"].append(masks)
        results["group"].append(group)
        results["length"].append(len(tokens))

    return results


def generate_split(model_type: str, model_path: str, conversations: list, split_name: str, out_prefix: str, num_cpus: int = os.cpu_count()):
    # launch remote workers
    ray.init(num_cpus=num_cpus)

    handles = [convert_conversation_batch.remote(
        model_type=model_type,
        model_path=model_path,
        batch=batch
    ) for batch in _split(conversations, num_cpus)]

    # aggegrate results
    results = {"tokens": [], "masks": [], "group": [], "length": []}
    for handle in handles:
        batch_result = ray.get(handle)

        for k, v in batch_result.items():
            results[k].extend(v)

    schema = pyarrow.schema([
        pyarrow.field("tokens", pyarrow.list_(pyarrow.int32())),
        pyarrow.field("masks", pyarrow.list_(pyarrow.bool_())),
        pyarrow.field("group", pyarrow.int32()),
        pyarrow.field("length", pyarrow.int32()),
    ])
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

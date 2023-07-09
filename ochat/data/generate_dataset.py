"""
Generate training data based on conversations

Usage: python -m ochat.data.generate_data --in-file sharegpt_gpt4.json --tokenizer-name HF_REPO_NAME --out-dir .
"""

import argparse
import json
import os
import random

import ray


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
    results = []
    texts = []
    for c in batch:
        props = conversation_properties(c)

        # Generate template
        tokens, masks, group = model_config.generate_conversation_template(_tokenize, _tokenize_special, c["items"], props)

        # Truncate to specified tokens
        max_context = model_config.model_max_context
        if max_context is not None:
            tokens = tokens[:max_context]
            masks  = masks[:max_context]

        results.append((tokens, masks, group))
        texts.append(tokenizer.decode(tokens, spaces_between_special_tokens=False))

    return results, texts


def generate_split(model_type: str, model_path: str, conversations: list, split_name: str, out_file: str, num_cpus: int = os.cpu_count()):
    # launch remote workers
    ray.init(num_cpus=num_cpus)

    handles = [convert_conversation_batch.remote(
        model_type=model_type,
        model_path=model_path,
        batch=batch
    ) for batch in _split(conversations, num_cpus)]

    # aggegrate results
    results = []
    texts = []
    for handle in handles:
        batch_result, batch_text = ray.get(handle)

        results.extend(batch_result)
        texts.extend(batch_text)

    with open(f"{out_file}.{split_name}.json", "w") as f:
        json.dump(results, f)
    with open(f"{out_file}.{split_name}.text.json", "w") as f:
        json.dump(texts, f, indent="\t")

    ray.shutdown()


def generate_dataset(model_type, model_path, in_file, out_file, seed, eval_ratio):
    # Load conversations
    with open(in_file, "r") as f:
        conversations = json.load(f)

    # Train-test split
    random.seed(seed)
    random.shuffle(conversations)
    eval_num = int(eval_ratio * len(conversations))

    train_conversations = conversations[eval_num:]
    eval_conversations  = conversations[:eval_num]

    generate_split(model_type, model_path, train_conversations, "train", out_file)
    if eval_num > 0:
        generate_split(model_type, model_path, eval_conversations, "eval", out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)

    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--out-file", type=str,required=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-ratio", type=float, default=0.01)
    args = parser.parse_args()

    generate_dataset(**vars(args))

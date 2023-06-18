"""
Generate training data based on conversations

Usage: python -m ochat.data.generate_data --in-file sharegpt_gpt4.json --tokenizer-name HF_REPO_NAME --out-dir .
"""

import argparse
import json
import os
import random

import transformers
from ochat.config.model_config import OCHAT_CONFIG


TOKENIZER: transformers.AutoTokenizer = None


def convert_single_conversation(c):
    def _tokenize(text):
        """Tokenize text-only, ignoring all special tokens."""
        return TOKENIZER.convert_tokens_to_ids(TOKENIZER._tokenize(text))

    def _tokenize_special(special_name):
        return TOKENIZER.convert_tokens_to_ids(special_name)

    return OCHAT_CONFIG.generate_conversation_template(_tokenize, _tokenize_special, c["items"])


def generate_split(conversations: list, split_name: str, out_dir: str):
    # FIXME: Tokenizer have GIL, build faster multiprocessing
    converted = list(map(convert_single_conversation, conversations))

    # Output dataset
    with open(os.path.join(out_dir, f"ochat.{split_name}.json"), "w") as f:
        json.dump(converted, f)

    # Output plain texts
    all_plain_texts = TOKENIZER.batch_decode([tokens for (tokens, masks) in converted], spaces_between_special_tokens=False)

    with open(os.path.join(out_dir, f"ochat.{split_name}.text.json"), "w") as f:
        json.dump(all_plain_texts, f, indent="\t")


def generate_dataset(seed, in_file, tokenizer_name, out_dir, eval_ratio):
    # Load tokenizer
    global TOKENIZER
    TOKENIZER = transformers.AutoTokenizer.from_pretrained(tokenizer_name, use_auth_token=True, use_fast=False)

    # Load conversations
    with open(in_file, "r") as f:
        conversations = json.load(f)

    # Train-test split
    random.seed(seed)
    random.shuffle(conversations)
    eval_num = int(eval_ratio * len(conversations))

    train_conversations = conversations[eval_num:]
    eval_conversations  = conversations[:eval_num]

    generate_split(train_conversations, "train", out_dir)
    generate_split(eval_conversations, "eval", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--tokenizer-name", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument("--eval-ratio", type=float, default=0.01)
    args = parser.parse_args()

    generate_dataset(**vars(args))

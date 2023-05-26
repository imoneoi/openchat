"""
- Embed text documents using OpenAI embedding API.

Usage:
python -m ochat.visualization.openai_embedding --in-file dataset_processed/ochat_text.json --out-file dataset_processed/ochat_text_embeddings.json
"""


import tiktoken
import openai
import json
import argparse
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm import tqdm


# Text preprocessing

TEXT_BOS = "<s>"
TEXT_PROMPT = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
TEXT_REPLACE_TABLE = {"<|end_of_turn|>": "\n\n"}

# API

MAX_TOKENS = 8191
BATCH_SIZE = 64

MODEL_TYPE = "text-embedding-ada-002"
MODEL_TOKENIZER = tiktoken.encoding_for_model(MODEL_TYPE)
OPENAI_API_KEY = ""


############


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def embedding_with_backoff(**kwargs):
    return openai.Embedding.create(**kwargs)


def preprocess_text(text: str):
    # Preprocess text, remove bos, add prompt and replace
    if text.startswith(TEXT_BOS):
        text = text[len(TEXT_BOS):]

    for src, dst in TEXT_REPLACE_TABLE.items():
        text = text.replace(src, dst)

    text = TEXT_PROMPT + text

    # Tokenize and truncate
    tokens = MODEL_TOKENIZER.encode(text, disallowed_special=())
    tokens = tokens[:MAX_TOKENS]
    return tokens


def calculate_embeddings(samples):
    embeddings = []

    for start_idx in tqdm(range(0, len(samples), BATCH_SIZE)):
        # Obtain a chunk
        sample_chunk = samples[start_idx: start_idx + BATCH_SIZE]

        # To tokens
        tokens_chunk = list(map(preprocess_text, sample_chunk))

        # Call API
        # Reference: https://github.com/openai/openai-cookbook/blob/main/examples/Embedding_Wikipedia_articles_for_search.ipynb

        response = embedding_with_backoff(model=MODEL_TYPE, input=tokens_chunk)
        for i, be in enumerate(response["data"]):
            assert i == be["index"]  # double check embeddings are in same order as input

        embeddings_chunk = [e["embedding"] for e in response["data"]]

        embeddings.extend(embeddings_chunk)

    return embeddings


def main(args):
    samples = json.load(open(args["in_file"], "r"))
    embeddings = calculate_embeddings(samples)

    json.dump(embeddings, open(args["out_file"], "w"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, default="ochat.train.text.json")
    parser.add_argument("--out-file", type=str, default="ochat.train.embeddings.json")
    args = parser.parse_args()

    main(vars(args))

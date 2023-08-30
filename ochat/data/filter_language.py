"""
Filter languages using fasttext

Usage: python -m ochat.data.filter_language --fasttext-model lid.176.bin --in-file sharegpt_clean.json --out-file sharegpt_en.json --lang en
"""

import argparse
import json
import os
from pprint import pprint

import ray
import fasttext


def _split(a, n):
    # Split list a to n chunks
    # https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m): (i+1)*k+min(i+1, m)] for i in range(n)]


@ray.remote
def filter_conversation_batch(fasttext_model: str, keep_lang: list, skip_lang: list, batch: list):
    model = fasttext.load_model(fasttext_model)

    result = []
    lang_freq = {}
    for conversation in batch:
        # Extract pure text content
        text = ""
        for turn in conversation["items"]:
            text += turn["value"]

        # Predict language
        lang = model.predict(text.replace("\n", ""))[0][0].replace("__label__", "")

        # Count & Filter
        lang_freq.setdefault(lang, 0)
        lang_freq[lang] += 1

        if lang in skip_lang:
            continue
        if keep_lang and (lang not in keep_lang):
            continue

        result.append(conversation)

    return result, lang_freq


def filter_lang(fasttext_model: str, keep_lang: list, skip_lang: list, in_file: str, out_file: str, num_cpus: int = os.cpu_count()):
    # load conversations
    with open(in_file, "r") as f:
        conversations = json.load(f)

    # launch remote workers
    ray.init(num_cpus=num_cpus)

    handles = [filter_conversation_batch.remote(
        fasttext_model=fasttext_model,
        keep_lang=keep_lang,
        skip_lang=skip_lang,
        batch=batch
    ) for batch in _split(conversations, num_cpus)]

    # aggegrate results
    results = []
    total_lang_freq = {}
    for handle in handles:
        batch_result, lang_freq = ray.get(handle)

        results.extend(batch_result)
        for k, v in lang_freq.items():
            total_lang_freq.setdefault(k, 0)
            total_lang_freq[k] += v

    with open(out_file, "w") as f:
        json.dump(results, f)

    # show statistics
    print(f"Total {len(conversations)} Keep {len(results)}")
    pprint(total_lang_freq)

    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasttext-model", type=str, required=True)
    parser.add_argument("--keep-lang",  type=str, nargs='+', default=["en"])
    parser.add_argument("--skip-lang",  type=str, nargs='+', default=[])

    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--out-file", type=str, required=True)

    args = parser.parse_args()

    filter_lang(**vars(args))

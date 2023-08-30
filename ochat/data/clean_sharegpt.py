"""
- Convert html to markdown with basic data cleaning.
- Deduplication.

Usage:
python -m ochat.data.clean_sharegpt --in-dir dataset/ShareGPT --out-file dataset_processed/sharegpt_clean.json
"""

import argparse
import json
import re
import os
import glob
import pprint

import bs4
import markdownify  # == 0.11.6
from ray.util.multiprocessing import Pool


class DataPipelineError(Exception):
    pass


############### [1] Load data


def sample_load(filename):
    with open(filename) as f:
        parser = bs4.BeautifulSoup(f, "html.parser", from_encoding="utf-8")

    sample = json.loads(parser.find("script", {"id": "__NEXT_DATA__"}).decode_contents())
    sample = sample["props"]["pageProps"]

    if "content" not in sample:
        raise DataPipelineError("Unable to extract content")

    return {**sample["content"],
            "id": sample.get("id", ""),
            "views": sample.get("views", 0)}


############### [2] HTML cleaning


# blocked_words = ["openai"]

div_pattern = re.compile("<div.*?>")
span_pattern = re.compile("<span.*?>")
code_lang_pattern = re.compile(
    "```\s*" + "(.*?)" + "(?:Copy code)+" + "(.+?)" + "\s*?```", re.DOTALL
)
code_lang_format = "```\g<1>\n\g<2>\n```"
regenerate_pattern = re.compile("\d+ / \d+")
copy_chars_pattern = re.compile("Copy\d+ chars / \d+ words")
copy_code_pattern = re.compile("```(.*?)Copy code\s*```")


def reformat_code(val: str) -> str:
    # Input code format is:
    # ```
    # $<language>Copy code$<exact_code_here>
    #
    # ```
    # This function convert it into the correct markdown format
    return re.sub(code_lang_pattern, code_lang_format, val)


def html_to_markdown(val: str) -> str:
    # Remove all <div>. This is required to make intent work in code blocks.
    val = re.sub(div_pattern, "", val)
    # Remove all <span>. This is required to make underscores work in code blocks.
    val = re.sub(span_pattern, "", val)
    # Markdown to html
    val = markdownify.markdownify(val).strip()
    # Reformat code
    val = reformat_code(val)

    # Remove noisy "[number] / [number]" at the beginning
    noise = re.search(regenerate_pattern, val)
    if noise and noise.start() == 0:
        val = val[noise.end() :]
    # Remove noisy "Copy[number] chars / [number] words"
    val = re.sub(copy_chars_pattern, "", val)
    # Remove empty code block ```\nCopy code\n```
    val = re.sub(copy_code_pattern, "", val)

    # Strip
    val = val.replace("\n\n\n", "\n").strip()

    return val


# def contain_blocked_words(val: str) -> bool:
#     for w in blocked_words:
#         if w in val.lower():
#             return True
#     return False


def remove_whitespace_and_non_printable(s: str) -> str:
    return "".join([c for c in s if c.isprintable() and not c.isspace()])


def sample_clean_html(sample):
    roles = ["human", "gpt"]

    if len(sample["items"]) <= 1:
        raise DataPipelineError("Conversation too short")

    # Adjust the offset for cases like https://sharegpt.com/c/VyaZlh4
    if sample["items"][0]["from"] != "human":
        sample["items"] = sample["items"][1:]
    if len(sample["items"]) <= 1:
        raise DataPipelineError("Conversation too short")

    if sample["items"][-1]["from"] == "human":
        sample["items"] = sample["items"][:-1]
    if len(sample["items"]) <= 1:
        raise DataPipelineError("Conversation too short")

    char_count = 0
    for i, c in enumerate(sample["items"]):
        if c["from"] != roles[i % 2]:
            raise DataPipelineError("Wrong format")

        # if contain_blocked_words(c["value"]):
        #     raise DataPipelineError("Contain blocked words")

        try:
            new_val = html_to_markdown(c["value"])
        except (bs4.builder.ParserRejectedMarkup, AssertionError):
            raise DataPipelineError("Parser error")

        # Filter empty answers like https://sharegpt.com/c/mrllZ6u
        if not len(remove_whitespace_and_non_printable(new_val)):
            raise DataPipelineError("Empty answer")

        char_count += len(new_val)
        c["value"] = new_val

    if char_count < 16:
        raise DataPipelineError("Conversation too short")

    return sample


############### [3] Hash


def sample_add_hash(sample):
    values = ""
    for c in sample["items"]:
        values += c["value"]

    sample["hash"] = (hash(str(values)), len(sample["items"]))
    return sample


############### Full pipeline


def sample_pipeline(filename):
    try:
        sample = sample_load(filename)
        sample = sample_clean_html(sample)
        sample = sample_add_hash(sample)
        
        return True, sample
    except DataPipelineError as e:
        return False, str(e)


def main(in_dir, out_file):
    file_list = glob.glob(os.path.join(in_dir, "*.html"))

    # Process samples
    samples = Pool().map(sample_pipeline, file_list)

    # Hash-based deduplication
    error_count = {}
    num_duplicates = 0
    results = []

    visited = set()
    for success, sample in samples:
        if not success:
            error_count.setdefault(sample, 0)
            error_count[sample] += 1

            continue

        if sample["hash"] in visited:
            num_duplicates += 1
            continue

        visited.add(sample["hash"])
        results.append(sample)

    # Dump result
    with open(out_file, "wt") as f:
        json.dump(results, f, indent="\t")

    # Print errors
    pprint.pprint(error_count)
    print(f"Number of duplicates: {num_duplicates}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir",   type=str, required=True)
    parser.add_argument("--out-file", type=str, default="sharegpt_clean.json")

    main(**vars(parser.parse_args()))

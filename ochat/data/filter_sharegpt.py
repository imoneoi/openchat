"""
Filter content based on model

Usage: python -m ochat.data.filter_sharegpt --in-file sharegpt_clean.json --out-file sharegpt_gpt4.json
"""

import argparse
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--out-file", type=str, default="sharegpt_gpt4.json")
    parser.add_argument("--model", type=str, default="Model: GPT-4")

    args = parser.parse_args()

    with open(args.in_file, "r") as f:
        input_samples = json.load(f)

    # Filter samples based on model
    filtered_samples = []
    for sample in input_samples:
        model = sample.get("model", "")
        if model == args.model:
            filtered_samples.append(sample)

    with open(args.out_file, "w") as f:
        json.dump(filtered_samples, f, indent="\t")

    # Print
    print(f"Picked {len(filtered_samples)} samples from total {len(input_samples)} samples")

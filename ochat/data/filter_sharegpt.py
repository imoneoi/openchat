"""
Filter content based on model

Usage: python -m ochat.data.filter_sharegpt --in-file sharegpt_clean.json --out-file sharegpt_gpt4.json
"""

import argparse
import json

import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--out-file", type=str, default="sharegpt_gpt4.json")
    parser.add_argument("--model", type=str, default="Model: GPT-4")
    parser.add_argument("--inverse", action="store_true", default=False)

    parser.add_argument("--subsample", type=float, default=1.0)
    parser.add_argument("--subsample-seed", type=int, default=0)

    args = parser.parse_args()

    with open(args.in_file, "r") as f:
        input_samples = json.load(f)

    # Filter samples based on model
    filtered_samples = []
    for sample in input_samples:
        model = sample.get("model", "")
        if (model == args.model) ^ args.inverse:
            filtered_samples.append(sample)

    # Print
    print(f"Picked {len(filtered_samples)} samples from total {len(input_samples)} samples")

    # Subsampling
    if args.subsample < 1.0:
        keep = np.random.default_rng(seed=args.subsample_seed).random(len(filtered_samples)) < args.subsample
        filtered_samples = [s for s, k in zip(filtered_samples, keep) if k]

        # Print
        print(f"Subsampled {len(filtered_samples)} samples")

    with open(args.out_file, "w") as f:
        json.dump(filtered_samples, f, indent="\t")


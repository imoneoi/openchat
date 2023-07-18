import argparse
import os
from pathlib import Path

import orjson
import pandas as pd


def view_results(result_file: str):
    with open(result_file, "rb") as f:
        eval_results = orjson.loads(f.read())

    report = {}
    for name in sorted(eval_results["accuracy"].keys()):
        eval_set  = os.path.dirname(name)
        eval_name = Path(name).stem

        if eval_set not in report:
            report[eval_set] = {"name": [], "accuracy": [], "unmatched": []}
        
        report[eval_set]["name"].append(eval_name)
        report[eval_set]["accuracy"].append(eval_results["accuracy"][name])
        report[eval_set]["unmatched"].append(eval_results["unmatched"][name])

    for eval_set, result_df in report.items():
        result_df = pd.DataFrame.from_dict(result_df)
        result_df.loc[len(result_df)] = {
            "name": "Average",
            "accuracy": result_df["accuracy"].mean(),
            "unmatched": result_df["unmatched"].mean()
        }

        print(f"{eval_set}\n\n")
        print(result_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()

    # Input / output
    parser.add_argument("result_file", type=str, default="mcq_eval_result.json")

    args = parser.parse_args()

    view_results(**vars(args))

if __name__ == "__main__":
    main()

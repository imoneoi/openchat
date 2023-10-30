import argparse
import os
from pathlib import Path

import orjson
import pandas as pd
from glob import glob


def view_results(result_path: str):
    # Read results
    eval_results = []
    for filename in glob(os.path.join(result_path, "*.json")):
        with open(filename, "rb") as f:
            questions = orjson.loads(f.read())

            eval_results.extend([{
                "model": Path(filename).stem,
                "task_type": q["task_type"],
                "task_name": os.path.relpath(q["task_name"], q["task_type"]),
                "accuracy": q["is_correct"],
                "unmatched": not q["is_matched"],
            } for q in questions])

    df = pd.DataFrame.from_records(eval_results)

    # Overall metrics table
    df_overall = df.pivot_table(index=["model"], columns=["task_type"], values=["accuracy", "unmatched"], aggfunc="mean")
    print(df_overall.to_string(float_format=lambda x: f"{x * 100:.1f}", na_rep="-"))

    # Print tables for each task
    for task_type in df["task_type"].unique():
        df_task = df[df["task_type"] == task_type].pivot_table(index=["task_name"], columns=["model"], values=["accuracy", "unmatched"], aggfunc="mean")

        print(f"\n### {task_type}\n")
        print(df_task.to_string(float_format=lambda x: f"{x * 100:.1f}", na_rep="-"))


def main():
    parser = argparse.ArgumentParser()

    # Input / output
    parser.add_argument("--result_path", type=str, default="ochat/evaluation/eval_results")

    args = parser.parse_args()

    view_results(**vars(args))


if __name__ == "__main__":
    main()

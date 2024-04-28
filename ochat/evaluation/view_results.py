import argparse
import os
from pathlib import Path

import orjson
import pandas as pd
from glob import glob

def save_results(dfs, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with pd.ExcelWriter(save_path) as writer:
        for task_type, df_task in dfs.items():
            df_task.to_excel(writer, sheet_name=task_type)
                
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
    all_tables = dict()    
    # Overall metrics table
    df_overall = df.pivot_table(index=["model"], columns=["task_type"], values=["accuracy", "unmatched"], aggfunc="mean")
    all_tables["overall"] = df_overall
    print(df_overall.to_string(float_format=lambda x: f"{x * 100:.1f}", na_rep="-"))
    # Print tables for each task
    for task_type in df["task_type"].unique():
        df_task = df[df["task_type"] == task_type].pivot_table(index=["task_name"], columns=["model"], values=["accuracy", "unmatched"], aggfunc="mean")
        all_tables[task_type.replace("/", "_")] = df_task
        print(f"\n### {task_type}\n")
        print(df_task.to_string(float_format=lambda x: f"{x * 100:.1f}", na_rep="-"))
    return all_tables


def main():
    parser = argparse.ArgumentParser()

    # Input / output
    parser.add_argument("--result_path", type=str, default="ochat/evaluation/eval_results")
    parser.add_argument("--save_path", type=str, default="ochat/evaluation/eval_results/summary.xlsx")
    parser.add_argument("--save", "-s", action="store_true", help="Save the results to a file")
    args = parser.parse_args()
    
    all_tables = view_results(args.result_path)
    if args.save:
        save_results(all_tables, args.save_path)


if __name__ == "__main__":
    main()

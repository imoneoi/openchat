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
            data = orjson.loads(f.read())

        for template_name, template_questions in data.items():
            eval_results.extend([{
                "model": Path(filename).stem,
                "template": template_name,

                "task_type": q["task_type"],
                "task_name": os.path.relpath(q["task_name"], q["task_type"]),

                "accuracy": q["is_correct"],
                "unmatched": not q["is_matched"],
            } for q in template_questions])

    df = pd.DataFrame.from_records(eval_results)
    df = df.groupby(["model", "template", "task_type", "task_name"]).mean()
    print (df)

    report = {}
    # for name in sorted(eval_results["accuracy"].keys()):
    #     name_split = name.split("___", 1)
    #     if len(name_split) == 2:
    #         config_set, task_filename = name_split
    #     else:
    #         config_set = "default"
    #         task_filename = name

    #     eval_set  = os.path.dirname(task_filename)
    #     eval_name = Path(task_filename).stem

    #     report.setdefault(config_set, {})
    #     report[config_set].setdefault(eval_set, {"name": [], "accuracy": [], "unmatched": []})
        
    #     report[config_set][eval_set]["name"].append(eval_name)
    #     report[config_set][eval_set]["accuracy"].append(eval_results["accuracy"][name])
    #     report[config_set][eval_set]["unmatched"].append(eval_results["unmatched"][name])

    # for config_set, eval_sets_report in report.items():
    #     print(f"\n{config_set}\n==========")

    #     # eval set
    #     for eval_set, result_df in eval_sets_report.items():
    #         result_df = pd.DataFrame.from_dict(result_df)
    #         result_df.loc[len(result_df)] = {
    #             "name": "Average",
    #             "accuracy": result_df["accuracy"].mean(),
    #             "unmatched": result_df["unmatched"].mean()
    #         }

    #         print(f"\n## {eval_set}\n")
    #         print(result_df.to_markdown(index=False, floatfmt=".3f"))


def main():
    parser = argparse.ArgumentParser()

    # Input / output
    parser.add_argument("--result_path", type=str, default="ochat/evaluation/eval_results")

    args = parser.parse_args()

    view_results(**vars(args))

if __name__ == "__main__":
    main()

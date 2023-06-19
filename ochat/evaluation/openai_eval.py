import os
import json
import argparse
import time
import glob
import pathlib
import asyncio
from copy import deepcopy

import openai
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_random_exponential


def read_jsonl(filename):
    with open(filename, "r") as f:
        result = list(map(json.loads, f.readlines()))

    return result


def write_jsonl(filename, content):
    content = list(map(lambda x: json.dumps(x) + "\n", content))

    with open(filename, "w") as f:
        f.writelines(content)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
async def chat_completion_with_backoff(**kwargs):
    return await openai.ChatCompletion.acreate(**kwargs)


async def chat_completion_and_parse_score(**kwargs):
    review = await chat_completion_with_backoff(**kwargs)
    review = review["choices"][0]["message"]["content"].strip("\n")

    # Try to find the score pair in the beginning line
    try:
        sp = review.split("\n")[0].replace(",", " ").split(" ")
        if len(sp) == 2:
            return float(sp[0]), float(sp[1]), review
    except:
        pass

    # Try to find the score pair in the last line
    try:
        sp = review.split("\n")[-1].replace(",", " ").split(" ")
        if len(sp) == 2:
            return float(sp[0]), float(sp[1]), review
    except:
        pass

    return 0., 0., review


async def openai_eval(reviewer_list: list, prompt_list: list, answer1_list: list, answer2_list: list, continue_from: list):
    result = deepcopy(continue_from)
    if result is None:
        result = deepcopy(answer1_list)

    # maps
    reviewer_category_map = {reviewer["category"]: reviewer for reviewer in reviewer_list}
    prompt_id_map = {prompt["prompt_id"]: prompt for prompt in prompt_list}

    default_reviewer = reviewer_category_map["general"]

    # eval all answers
    for answer1, answer2, eval in tqdm(zip(answer1_list, answer2_list, result)):
        if eval.get("score", None) is not None:
            continue

        # eval answer
        assert answer1["text"] == answer2["text"]

        question = answer1["text"]
        category = answer1["category"]

        reviewer = reviewer_category_map.get(category, default_reviewer)
        prompt   = prompt_id_map[reviewer["prompt_id"]]

        try:
            # Average of two directions
            a_score1, a_score2, a_review = await chat_completion_and_parse_score(
                messages=[
                    {"role": "system", "content": prompt["system_prompt"]},
                    {"role": "user", "content": prompt["prompt_template"].format(
                        question=question, answer_1=answer1["answer"], answer_2=answer2["answer"], **prompt["defaults"]
                    )}
                ],
                **reviewer["kwargs"]
            )
            b_score2, b_score1, b_review = await chat_completion_and_parse_score(
                messages=[
                    {"role": "system", "content": prompt["system_prompt"]},
                    {"role": "user", "content": prompt["prompt_template"].format(
                        question=question, answer_1=answer2["answer"], answer_2=answer1["answer"], **prompt["defaults"]
                    )}
                ],
                **reviewer["kwargs"]
            )

            eval["score"] = {
                "score": ((a_score1 + b_score1) / 2, (a_score2 + b_score2) / 2), 
                "raw_a": [(a_score1, a_score2), a_review],
                "raw_b": [(b_score1, b_score2), b_review],
            }

        except Exception as e:
            tqdm.write (str(e))

    return result


async def main():
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",     type=str, required=True)
    parser.add_argument("--input_path",    type=str, required=True)
    parser.add_argument("--baseline_path", type=str, required=True)

    parser.add_argument("--do_continue",      default=False, action="store_true")

    args = parser.parse_args()

    # Load reviewer and prompt list
    reviewer_list = read_jsonl(os.path.join(args.data_path, "reviewer.jsonl"))
    prompt_list   = read_jsonl(os.path.join(args.data_path, "prompt.jsonl"))

    # Load baseline answers
    baseline_answer_list = read_jsonl(args.baseline_path)

    # Eval models w.r.t baselines
    cur_date = time.strftime("%Y%m%d")

    results_output_filename = []
    results_eval_list       = []
    for f in glob.glob(os.path.join(args.input_path, "*.jsonl")):
        # load answers
        model_answer_list = read_jsonl(f)

        # load (continue) file name
        output_filename = os.path.join(args.input_path,
                                       f"eval_result_{cur_date}",
                                       f"{pathlib.Path(f).stem}_VS_{pathlib.Path(args.baseline_path).stem}.jsonl")
        
        continue_from = None
        if args.do_continue and os.path.exists(output_filename):
            print (f"Continuing from {output_filename}.")

            continue_from = read_jsonl(output_filename)

        # eval
        results_output_filename.append(output_filename)
        results_eval_list.append(asyncio.create_task(
            openai_eval(reviewer_list, prompt_list, model_answer_list, baseline_answer_list, continue_from)
        ))

    await asyncio.wait(results_eval_list)

    # Write results
    for output_filename, eval_list in zip(results_output_filename, results_eval_list):
        eval_list = eval_list.result()
        
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        write_jsonl(output_filename, eval_list)


if __name__ == "__main__":
    asyncio.run(main())

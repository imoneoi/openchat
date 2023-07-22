"""Evaluate OpenAI API models on multiple-choice questions (MCQ)"""

from typing import Optional
import argparse
import os
import asyncio
from glob import glob

import orjson
import openai
from tqdm import tqdm
from openai.error import RateLimitError, ServiceUnavailableError
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type


def _match_answer(response: str, letter_set: list):
    for c in response:
        if c in letter_set:
            return c

    return ""


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(20), retry=retry_if_exception_type((RateLimitError, ServiceUnavailableError, )))
async def chat_completion_with_backoff(sem, **kwargs):
    async with sem:
        return await openai.ChatCompletion.acreate(**kwargs)


async def run_mcq(
    model_type: str,
    data_path: str,
    continue_from: Optional[str],
    output_path: str,
    parallel: int
):
    # Load mcq files
    mcq_questions = []

    mcq_filenames = glob(os.path.join(data_path, "**", "*.jsonl"), recursive=True)
    for filename in mcq_filenames:
        task_name = filename[len(data_path):]
        with open(filename, "r") as f:
            task_data = list(map(orjson.loads, f.readlines()))

        mcq_questions.extend([{**item, "task_name": task_name, "response": ""} for item in task_data])

    # Load continue
    if continue_from is not None:
        print (f"Continuing from {continue_from}...")
        with open(continue_from, "r") as f:
            continue_data = orjson.loads(f.read())
        
        # map results
        continue_ans_map = {}
        for results in continue_data["results"].values():
            for item in results:
                continue_ans_map[item["prompt"]] = item["response"]

        for q in mcq_questions:
            q["response"] = continue_ans_map.get(q["question"], "")
        
    # run openai api
    sem = asyncio.Semaphore(parallel)
    for q in mcq_questions:
        if q["response"]:
            continue

        messages = [{"role": "user", "content": q["question"]}]
        q["response"] = asyncio.create_task(chat_completion_with_backoff(
            sem,
            model=model_type,
            messages=messages,

            temperature=0
        ))

    # Group answers
    results = {}
    for q in tqdm(mcq_questions):
        response = q["response"]
        if not isinstance(response, str):
            try:
                await response
                response = response.result()["choices"][0]["message"]["content"]
            except Exception as e:
                response = ""

                if hasattr(e, "last_attempt"):
                    e = e.last_attempt
                if hasattr(e, "_exception"):
                    e = e._exception

                print(str(e))

        task_name = q["task_name"]

        results.setdefault(task_name, [])
        results[task_name].append({
            "prompt": q["question"],
            "response": response,

            "answer": _match_answer(response, set(q["options"])),
            "label": q["label"],
            "options": q["options"]
        })

    # Calculate accuracy
    accuracy = {}
    unmatched = {}
    for task_name, qs in results.items():
        accuracy[task_name]  = sum([q["answer"]     in q["label"]   for q in qs]) / len(qs)
        unmatched[task_name] = sum([q["answer"] not in q["options"] for q in qs]) / len(qs)

    with open(output_path, "wb") as f:
        f.write(orjson.dumps({"accuracy": accuracy, "unmatched": unmatched, "results": results}, option=orjson.OPT_INDENT_2))


async def main():
    parser = argparse.ArgumentParser()

    # Input / output
    parser.add_argument("--model_type",  type=str, default="gpt-3.5-turbo")
    parser.add_argument("--data_path",   type=str, default="ochat/evaluation/mcq_set")
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--parallel",    type=int, default=16)

    args = parser.parse_args()

    await run_mcq(**vars(args))

if __name__ == "__main__":
    asyncio.run(main())

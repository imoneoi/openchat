import os
import json
import argparse
import time
import asyncio

import openai
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_random_exponential


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
async def chat_completion_with_backoff(sem, **kwargs):
    async with sem:
        return await openai.ChatCompletion.acreate(**kwargs)


async def main():
    parser = argparse.ArgumentParser()

    # Input / output
    parser.add_argument("--data_path",    type=str, required=True)
    parser.add_argument("--output_path",  type=str, required=True)
    parser.add_argument("--model_types",  type=str, nargs='+', default=["gpt-3.5-turbo"])

    parser.add_argument("--api_base",     type=str, default=None)
    parser.add_argument("--parallel_req", type=int, default=32)

    # Temperature
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p",       type=float, default=0.9)

    args = parser.parse_args()

    # Load questions
    with open(os.path.join(args.data_path, "question.jsonl"), "r") as f:
        question_list = list(map(json.loads, f.readlines()))

    # Get API answers
    sem = asyncio.Semaphore(args.parallel_req)

    if args.api_base is not None:
        openai.api_base = args.api_base

    cur_date = time.strftime("%Y%m%d")
    for model_type in args.model_types:
        output_filename = os.path.join(args.output_path, f"{os.path.basename(args.data_path)}_{model_type}.jsonl")

        # Async API call
        tasks = []
        for question in question_list:
            tasks.append(asyncio.create_task(chat_completion_with_backoff(
                sem,
                model=model_type,
                messages=[
                    {"role": "user", "content": question["text"]}
                ],

                temperature=args.temperature,
                top_p=args.top_p
            )))

        # Write answers
        answer_list = []
        for question, task in zip(question_list, tqdm(tasks)):
            await task
            answer = task.result()
            answer = answer["choices"][0]["message"]["content"]

            answer_list.append({
                "answer_model": f"{model_type}_{cur_date}",
                "answer": answer,
                **question
            })

        # Write jsonl
        answer_list = list(map(lambda x: json.dumps(x) + "\n", answer_list))
        with open(output_filename, "w") as f:
            f.writelines(answer_list)


if __name__ == "__main__":
    asyncio.run(main())

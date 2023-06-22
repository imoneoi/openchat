import os
import json
import argparse
import time

import openai
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_random_exponential


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def chat_completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def main():
    parser = argparse.ArgumentParser()

    # Input / output
    parser.add_argument("--data_path",    type=str, required=True)
    parser.add_argument("--output_path",  type=str, required=True)
    parser.add_argument("--model_types",  type=str, nargs='+', default=["gpt-3.5-turbo", "gpt-4"])

    # Temperature
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p",       type=float, default=0.9)

    args = parser.parse_args()

    # Load questions
    with open(os.path.join(args.data_path, "question.jsonl"), "r") as f:
        question_list = list(map(json.loads, f.readlines()))

    # Get API answers
    cur_date = time.strftime("%Y%m%d")
    for model_type in args.model_types:
        output_filename = os.path.join(args.output_path, f"{os.path.basename(args.data_path)}_{model_type}.jsonl")

        # API call
        answer_list = []
        for question in tqdm(question_list):
            answer = chat_completion_with_backoff(
                model=model_type,
                messages=[
                    {"role": "user", "content": question["text"]}
                ],

                temperature=args.temperature,
                top_p=args.top_p
            )
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
    main()

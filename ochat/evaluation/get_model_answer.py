import json
import argparse
import random
import os

import torch
import numpy as np
from tqdm import tqdm

from ochat.config.model_config import ModelConfig, MODEL_CONFIG_MAP
from ochat.serving.inference import generate_stream


def get_model_answers(
    seed: int,
    # Input
    model_path: str,
    model_config: ModelConfig,
    question_list: list,
    # Settings
    temperature: float,
    top_p: float
):
    # Seed all RNGs
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    # Load tokenizer and model
    tokenizer = model_config.model_tokenizer_create(model_path)
    model     = model_config.model_create(model_path).cuda()

    # Generate
    answer_list = []

    model.eval()
    for question in tqdm(question_list):
        conversation = [{"from": "human", "value": question["text"]}]

        _, answer = next(generate_stream(
            model=model, tokenizer=tokenizer, model_config=model_config,
            conversation=conversation,
            temperature=temperature, top_p=top_p))

        tqdm.write(answer)

        answer_list.append(
            {
                "answer_model": model_path,
                "answer": answer,
                **question
            }
        )

    return answer_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)

    # Input / output
    parser.add_argument("--model_type",     type=str, required=True)
    parser.add_argument("--models_path",    type=str, required=True)
    parser.add_argument("--data_path",      type=str, required=True)
    parser.add_argument("--output_path",    type=str, required=True)

    # Temperature
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p",       type=float, default=0.9)

    args = parser.parse_args()

    # Load questions
    with open(os.path.join(args.data_path, "question.jsonl"), "r") as f:
        question_list = list(map(json.loads, f.readlines()))

    # Eval models
    eval_results = {}
    for f in os.scandir(args.models_path):
        if f.is_dir():
            output_filename = os.path.join(args.output_path, f"{os.path.basename(args.data_path)}_{f.name}_{args.seed}.jsonl")

            if not os.path.exists(output_filename):
                tqdm.write (f"Evaluating {f.path}...")
                eval_results[output_filename] = get_model_answers(
                    seed=args.seed,

                    model_path=f.path,
                    model_config=MODEL_CONFIG_MAP[args.model_type],
                    question_list=question_list,

                    temperature=args.temperature,
                    top_p=args.top_p
                )

    # Write results
    for output_filename, result in eval_results.items():
        # to jsonl
        result = list(map(lambda x: json.dumps(x) + "\n", result))

        with open(output_filename, "w") as f:
            f.writelines(result)


if __name__ == "__main__":
    main()

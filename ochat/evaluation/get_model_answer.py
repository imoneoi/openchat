import json
import argparse
import random
import os

import torch
import ray
import transformers
import numpy as np
from tqdm import tqdm

from ochat.config.model_config import ModelConfig, MODEL_CONFIG_MAP


@ray.remote(num_gpus=1)
def get_model_answers(
    seed: int,
    # Input
    tokenizer_path: str,
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
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path, use_auth_token=True, use_fast=False)
    model     = transformers.AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).cuda()

    # Tokenizer functions
    def _tokenize(text):
        return tokenizer.convert_tokens_to_ids(tokenizer._tokenize(text))

    def _tokenize_special(special_name):
        return tokenizer.convert_tokens_to_ids(special_name)

    # Generate
    answer_list = []

    model.eval()
    with torch.no_grad():
        for question in tqdm(question_list):
            prompt_tokens, _ = model_config.generate_conversation_template(
                _tokenize, _tokenize_special, [
                    {"from": "human", "value": question["text"]},
                    {"from": "gpt"}
                ]
            )

            answer_ids = model.generate(
                inputs=torch.as_tensor(prompt_tokens).unsqueeze(0).cuda(),
                generation_config=transformers.GenerationConfig(
                    max_length=model_config.max_tokens,
                    do_sample=True,
                    use_cache=True,
                    top_p=top_p,
                    temperature=temperature,
                    eos_token_id=_tokenize_special(model_config.eot_token),
                    pad_token_id=0  # FIXME: <unk> in LLaMA
                )
            )
            answer_ids = answer_ids[0][len(prompt_tokens):]
            answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

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
    parser.add_argument("--tokenizer_path", type=str, required=True)
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
    ray.init()

    eval_results = {}
    for f in os.scandir(args.models_path):
        if f.is_dir():
            output_filename = os.path.join(args.output_path, f"{os.path.basename(args.data_path)}_{f.name}_{args.seed}.jsonl")

            if not os.path.exists(output_filename):
                eval_results[output_filename] = get_model_answers.remote(
                    seed=args.seed,

                    tokenizer_path=args.tokenizer_path,
                    model_path=f.path,
                    model_config=MODEL_CONFIG_MAP[args.model_type],
                    question_list=question_list,

                    temperature=args.temperature,
                    top_p=args.top_p
                )

    # Write results
    for output_filename, result in eval_results.items():
        # to jsonl
        result = ray.get(result)
        result = list(map(lambda x: json.dumps(x) + "\n", result))

        with open(output_filename, "w") as f:
            f.writelines(result)


if __name__ == "__main__":
    main()

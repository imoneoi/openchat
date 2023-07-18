"""Evaluate OpenChat models on multiple-choice questions (MCQ)"""

import argparse
import os
from glob import glob

from vllm import LLM, SamplingParams
import ray
import orjson


def _split(a, n):
    # Split list a to n chunks
    # https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m): (i+1)*k+min(i+1, m)] for i in range(n)]


def _match_answer(response: str):
    for c in response:
        if c.isupper():
            return c

    return ""


@ray.remote
def convert_conversation_batch(model_type: str, model_path: str, batch: list):
    from ochat.config.model_config import MODEL_CONFIG_MAP

    # Tokenization
    model_config = MODEL_CONFIG_MAP[model_type]
    tokenizer = model_config.model_tokenizer_create(model_path)

    def _tokenize(text):
        """Tokenize text-only, ignoring all special tokens."""
        return tokenizer.convert_tokens_to_ids(tokenizer._tokenize(text))

    def _tokenize_special(special_name):
        return tokenizer.convert_tokens_to_ids(special_name)
    
    # Generate data
    results = []
    for item in batch:
        # Generate template
        tokens, _, _ = model_config.generate_conversation_template(_tokenize, _tokenize_special, system_prompt="", message_list=[
            {"from": "human", "value": item['question']},
            {"from": model_config.ai_role}
        ], message_props={"is_gpt4": False})
        # Alpaca
        # tokens = [_tokenize_special("<s>")] + _tokenize(f"### Instruction:\n{item['question']}\n\n### Response:")
        # Vicuna
        #tokens = [_tokenize_special("<s>")] + \
        #         _tokenize(f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {item['question']} ASSISTANT:")
        # Plain
        # tokens = [_tokenize_special("<s>")] + _tokenize(item['question'])
        # Orca paper
        # tokens = [_tokenize_special("<s>")] + _tokenize(f"### System:\n### Human:\n{item['question']}\n### Assistant:")

        # Truncate to specified tokens
        max_context = model_config.model_max_context
        if max_context is not None:
            tokens = tokens[-max_context:]

        results.append({
            "tokens": tokens,
            "text": tokenizer.decode(tokens, spaces_between_special_tokens=False),

            "label": item["label"],
            "options": item["options"],

            "task_name": item["task_name"]
        })

    return results


def run_mcq(
    model_path: str,
    model_type: str,
    data_path: str,
    output_path: str,
    num_cpus: int = os.cpu_count() - 1
):
    from ochat.config.model_config import MODEL_CONFIG_MAP

    # Load mcq files
    mcq_questions = []

    mcq_filenames = glob(os.path.join(data_path, "**", "*.jsonl"), recursive=True)
    for filename in mcq_filenames:
        task_name = filename[len(data_path):]
        with open(filename, "r") as f:
            task_data = list(map(orjson.loads, f.readlines()))

        mcq_questions.extend([{**item, "task_name": task_name} for item in task_data])
        
    # batch tokenize using ray
    ray.init(num_cpus=num_cpus)

    handles = [convert_conversation_batch.remote(
        model_type=model_type,
        model_path=model_path,
        batch=batch
    ) for batch in _split(mcq_questions, num_cpus)]

    converted_questions = []
    for handle in handles:
        converted_questions.extend(ray.get(handle))

    # vLLM inference
    engine = LLM(model_path)
    sampling_params = SamplingParams(temperature=0, stop=MODEL_CONFIG_MAP[model_type].eot_token)
    responses = engine.generate(prompt_token_ids=[q["tokens"] for q in converted_questions],
                                sampling_params=sampling_params)

    responses = sorted(responses, key=lambda x: int(x.request_id))
    responses = [x.outputs[0].text for x in responses]

    # Group answers
    results = {}
    for q, response in zip(converted_questions, responses):
        task_name = q["task_name"]

        results.setdefault(task_name, [])
        results[task_name].append({
            "prompt": q["text"],
            "response": response,

            "answer": _match_answer(response),
            "label": q["label"],
            "options": q["options"]
        })

    # Calculate accuracy
    accuracy = {}
    unmatched = {}
    for task_name, qs in results.items():
        accuracy[task_name]  = sum([q["answer"] == q["label"]       for q in qs]) / len(qs)
        unmatched[task_name] = sum([q["answer"] not in q["options"] for q in qs]) / len(qs)

    with open(output_path, "wb") as f:
        f.write(orjson.dumps({"accuracy": accuracy, "unmatched": unmatched, "results": results}, option=orjson.OPT_INDENT_2))


def main():
    parser = argparse.ArgumentParser()

    # Input / output
    parser.add_argument("--model_path",  type=str, required=True)
    parser.add_argument("--model_type",  type=str, required=True)
    parser.add_argument("--data_path",   type=str, default="ochat/evaluation/mcq_set")
    parser.add_argument("--output_path", type=str, default=None)

    args = parser.parse_args()

    run_mcq(**vars(args))

if __name__ == "__main__":
    main()

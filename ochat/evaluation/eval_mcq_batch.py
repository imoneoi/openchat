"""Evaluate OpenChat models on multiple-choice questions (MCQ)"""

import argparse
import os
from glob import glob

from vllm import LLM, SamplingParams
import ray
import orjson


PROMPT_TEMPLATES = {
    # Plain text
    "plain": lambda question, model_config, tokenize_fn, tokenize_special_fn: 
        [tokenize_special_fn("<s>")] + tokenize_fn(question),

    # GPT-4 condition
    "cond_gpt4": lambda question, model_config, tokenize_fn, tokenize_special_fn: 
        model_config.generate_conversation_template(tokenize_fn, tokenize_special_fn, system_prompt="", message_list=[
            {"from": "human", "value": question},
            {"from": model_config.ai_role}
        ])[0],

    # GPT-3.5 condition
    "cond_gpt35": lambda question, model_config, tokenize_fn, tokenize_special_fn: 
        model_config.generate_conversation_template(tokenize_fn, tokenize_special_fn, system_prompt="", message_list=[
            {"from": "human", "value": question},
            {"from": model_config.ai_role}
        ], message_props={"is_gpt4": False})[0],
}


def _split(a, n):
    # Split list a to n chunks
    # https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m): (i+1)*k+min(i+1, m)] for i in range(n)]


def _match_answer(response: str, letter_set: list):
    for c in response:
        if c in letter_set:
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
        for prompt_template_name, prompt_template_fn in PROMPT_TEMPLATES.items():
            # Generate template
            tokens = prompt_template_fn(item["question"],
                                        model_config=model_config,
                                        tokenize_fn=_tokenize,
                                        tokenize_special_fn=_tokenize_special)

            # Truncate to specified tokens
            max_context = model_config.model_max_context
            if max_context is not None:
                tokens = tokens[-max_context:]

            results.append({
                "tokens": tokens,
                "text": tokenizer.decode(tokens, spaces_between_special_tokens=False),

                "label": item["label"],
                "options": item["options"],

                "task_name": f"{prompt_template_name}___{item['task_name']}"
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
    sampling_params = SamplingParams(temperature=0,
                                     max_tokens=MODEL_CONFIG_MAP[model_type].model_max_context,
                                     stop=[MODEL_CONFIG_MAP[model_type].eot_token])
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

            "answer": _match_answer(response, set(q["options"])),
            "label": q["label"],
            "options": q["options"]
        })

    # Calculate accuracy
    accuracy = {}
    unmatched = {}
    for task_name, qs in results.items():
        accuracy[task_name]  = sum([q["answer"] in q["label"]       for q in qs]) / len(qs)
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

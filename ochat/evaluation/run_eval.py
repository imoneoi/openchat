from typing import Optional
import argparse
import os
import asyncio
from glob import glob
from copy import deepcopy

import orjson
import openai
from tqdm import tqdm
from openai.error import RateLimitError, ServiceUnavailableError
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from vllm import LLM, SamplingParams

from ochat.config.model_config import MODEL_CONFIG_MAP
from ochat.evaluation.match_answer import MATCH_ANSWER_FUNCTION
from ochat.evaluation.conversation_templates import CONVERSATION_TEMPLATES


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(20), retry=retry_if_exception_type((RateLimitError, ServiceUnavailableError, )))
async def chat_completion_with_backoff(sem, **kwargs):
    async with sem:
        return await openai.ChatCompletion.acreate(**kwargs)


async def get_openai_answers(
    questions: dict,
    model_type: str,
    parallel: int
):
    # Complete in retry cycles
    last_to_complete_num = None
    while True:
        # run openai api
        to_complete_num = 0
        sem = asyncio.Semaphore(parallel)
        for q in questions["default"]:
            if q["response"]:
                continue

            q["response"] = asyncio.create_task(chat_completion_with_backoff(
                sem,
                model=model_type,
                messages=[{"role": "user", "content": q["question"]}],

                temperature=0
            ))
            to_complete_num += 1

        # Fetch answers
        tqdm.write(f"New completion cycle. To complete {to_complete_num}, number of parallel calls {parallel}")
        for q in tqdm(questions["default"]):
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

                q["response"] = response

        # Next retry cycle
        # Break if cannot complete more
        if (to_complete_num == last_to_complete_num) or (to_complete_num == 0):
            break
        last_to_complete_num = to_complete_num

        # Reduce parallel calls
        parallel = max(1, parallel // 2)

    return questions


async def get_model_answers(
    questions: dict,
    model_type: str,
    model_path: str
):
    # Init vLLM engine
    model_config = MODEL_CONFIG_MAP[model_type]

    engine = LLM(model_path,
                 max_num_batched_tokens=model_config.model_max_context)
    sampling_params = SamplingParams(temperature=0,
                                     max_tokens=model_config.model_max_context,
                                     stop=[model_config.eot_token])

    # Init tokenizer
    tokenizer = model_config.model_tokenizer_create(model_path)

    def _tokenize(text):
        """Tokenize text-only, ignoring all special tokens."""
        return tokenizer.convert_tokens_to_ids(tokenizer._tokenize(text))

    def _tokenize_special(special_name):
        return tokenizer.convert_tokens_to_ids(special_name)

    # Complete
    prompts = []
    prompt_indices = []

    for template_name, template_questions in questions.items():
        prompt_template_fn = CONVERSATION_TEMPLATES[template_name]

        for idx, q in enumerate(template_questions):
            if q["response"]:
                continue

            tokens = prompt_template_fn(q["question"],
                                        model_config=model_config,
                                        tokenize_fn=_tokenize,
                                        tokenize_special_fn=_tokenize_special)

            # Truncate to specified tokens
            max_context = model_config.model_max_context
            if max_context is not None:
                tokens = tokens[-max_context:]

            prompt_indices.append((template_name, idx))
            prompts.append(tokens)

            q["prompt"] = tokenizer.decode(tokens)

    # calculate & fill in responses
    responses = engine.generate(prompt_token_ids=prompts, sampling_params=sampling_params)

    responses = sorted(responses, key=lambda x: int(x.request_id))
    responses = [x.outputs[0].text for x in responses]

    for (template_name, idx), resp in zip(prompt_indices, responses):
        questions[template_name][idx]["response"] = resp

    return questions


async def run_eval(
    model_type: str,
    model_path: str,
    conversation_templates: list,

    data_path: str,
    eval_sets: list,

    continue_from: Optional[str],
    output_file: str,

    parallel: int
):
    if continue_from is not None:
        # Load continue
        print (f"Continuing from {continue_from}...")

        with open(continue_from, "rb") as f:
            questions = orjson.loads(f.read())
    else:
        # Load questions
        questions = []

        for filename in glob(os.path.join(data_path, "**", "*.jsonl"), recursive=True):
            task_name = os.path.splitext(filename[len(data_path):])[0].strip("\\/")
            task_type = os.path.dirname(task_name)

            assert task_type in MATCH_ANSWER_FUNCTION

            # Filter eval sets
            if eval_sets and not sum([task_name.startswith(a) for a in eval_sets]):
                continue

            # Load task
            with open(filename, "r") as f:
                task_data = list(map(orjson.loads, f.readlines()))

            questions.extend([{**item, "task_name": task_name, "task_type": task_type, "response": ""} for item in task_data])

        # Add conversation templates
        if not conversation_templates:
            conversation_templates = CONVERSATION_TEMPLATES.keys()

        questions = {template: deepcopy(questions) for template in conversation_templates}

    # run completion
    if model_path is None:
        questions = await get_openai_answers(questions, model_type, parallel)
    else:
        questions = await get_model_answers(questions, model_type, model_path)

    # Calculate accuracy
    for template_name, template_questions in questions.items():
        for q in template_questions:
            q["is_matched"], q["answer"] = MATCH_ANSWER_FUNCTION[q["task_type"]](q, q["response"])
            q["is_correct"] = q["answer"] in q["label"]

    # Write results
    if output_file is None:
        output_file = os.path.join(os.path.dirname(data_path), "eval_results", f"{model_type}.json")

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "wb") as f:
        f.write(orjson.dumps(questions, option=orjson.OPT_INDENT_2))


async def main():
    parser = argparse.ArgumentParser()

    # Input / output
    parser.add_argument("--model_type",             type=str, default="gpt-3.5-turbo")
    parser.add_argument("--model_path",             type=str, default=None)
    parser.add_argument("--conversation_templates", type=str, nargs="+", default=["default"])

    parser.add_argument("--data_path", type=str, default="ochat/evaluation/eval_data")
    parser.add_argument("--eval_sets", type=str, nargs="+", default=[])

    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--output_file",   type=str, default=None)
    parser.add_argument("--parallel",      type=int, default=16)

    args = parser.parse_args()

    await run_eval(**vars(args))

if __name__ == "__main__":
    asyncio.run(main())

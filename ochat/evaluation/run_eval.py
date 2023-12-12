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
from vllm import LLM, SamplingParams

from transformers.utils.hub import cached_file

from ochat.evaluation.match_answer import MATCH_ANSWER_FUNCTION
from ochat.config import MODEL_CONFIG_MAP


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(20), retry=retry_if_exception_type((RateLimitError, ServiceUnavailableError, )))
async def _chat_completion_with_backoff(**kwargs):
    return await openai.ChatCompletion.acreate(**kwargs)


async def chat_completion_thread(model, progress_bar, queue):
    while True:
        # Fetch task
        try:
            task = queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        # Completion
        try:
            response = await _chat_completion_with_backoff(
                model=model,
                messages=[{"role": "user", "content": task["question"]}],

                temperature=0
            )
            task["response"] = response["choices"][0]["message"]["content"]  # type: ignore
        except Exception as e:
            if hasattr(e, "last_attempt"):
                e = e.last_attempt
            if hasattr(e, "_exception"):
                e = e._exception

            print(type(e), str(e))
        
        # Progress
        progress_bar.update()


async def get_openai_answers(
    model: str,
    questions: list,
    parallel: int
):
    # Complete in retry cycles
    last_to_complete_num = 0

    while True:
        # fill queue
        to_complete_num = 0
        queue = asyncio.Queue()
        for q in questions:
            if q["response"]:
                continue

            queue.put_nowait(q)
            to_complete_num += 1

        tqdm.write(f"New completion cycle. To complete {to_complete_num}, number of parallel calls {parallel}")

        # Create tasks
        progress_bar = tqdm(total=to_complete_num)
        async with asyncio.TaskGroup() as task_group:
            for _ in range(parallel):
                task_group.create_task(chat_completion_thread(model, progress_bar, queue))

        # Next retry cycle
        # Break if cannot complete more
        if (to_complete_num == last_to_complete_num) or (to_complete_num == 0):
            break
        last_to_complete_num = to_complete_num

        # Reduce parallel calls
        parallel = max(1, parallel // 2)

    return questions


def tokenize_questions(model_config: object, conv_template: object, questions: list, condition: str, system_msg: str):
    from ochat.config import Conversation, Message

    # Construct conversation
    prompt_indices = []
    conversations = []
    for idx, q in enumerate(questions):
        if q["response"]:
            continue

        conversations.append(Conversation(
            items=[
                Message(role="user", content=q["question"]),
                Message(role="assistant", content="")
            ],
            condition=condition,
            system=system_msg
        ))
        prompt_indices.append(idx)

    # Tokenize
    conversations, _ = conv_template.tokenize_conversations(conversations, inference=True)
    conversations    = [tokens[-model_config.model_max_context:] for tokens in conversations]

    return conversations, prompt_indices


def get_model_answers(
    model: str,
    questions: list,
    condition: str,
    system_msg: str,
    model_type: str
):
    # Load model config
    if model_type is None:
        with open(cached_file(path_or_repo_id=model, filename="openchat.json"), "r") as f:
            model_type = orjson.loads(f.read())["model_type"]

    model_config = MODEL_CONFIG_MAP[model_type]
    tokenizer = model_config.model_tokenizer_create(model)
    conv_template = model_config.conversation_template(tokenizer=tokenizer)

    # Init vLLM engine
    engine = LLM(model,
                 max_num_batched_tokens=model_config.model_max_context,
                 max_model_len=model_config.model_max_context)
    sampling_params = SamplingParams(temperature=0,
                                     max_tokens=model_config.model_max_context,
                                     stop_token_ids=conv_template.eot_tokens_,  # Override stop tokens
                                     ignore_eos=True)

    # Complete
    prompts, prompt_indices = tokenize_questions(model_config, conv_template, questions,
                                                 condition=condition, system_msg=system_msg)

    # calculate & fill in responses
    responses = engine.generate(prompt_token_ids=prompts, sampling_params=sampling_params)
    for idx, resp in zip(prompt_indices, responses):
        questions[idx]["response"] = resp.outputs[0].text


    return questions


async def run_eval(
    model: str,
    condition: str,
    system_msg: str,
    model_type: str,

    data_path: str,
    eval_sets: list,

    continue_from: Optional[str],
    output_file: str,

    parallel: int
):
    print (f"Evaluating ({model_type})...\n\nCondition: {condition}\nSystem Prompt: {system_msg}\n")

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

    # run completion
    if model.startswith("gpt-3.5-turbo") or model.startswith("gpt-4"):
        questions = await get_openai_answers(model, questions, parallel)
    else:
        questions = get_model_answers(model, questions, condition, system_msg, model_type)

    # Calculate accuracy
    for q in questions:
        q["is_matched"], q["answer"] = MATCH_ANSWER_FUNCTION[q["task_type"]](q, q["response"])
        try:
            q["is_correct"] = q["answer"] in q["label"]
        except:
            q["is_correct"] = False

    # Write results
    if output_file is None:
        output_file = os.path.join(os.path.dirname(data_path), "eval_results", f"{os.path.basename(model)}_{condition}.json")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "wb") as f:
        f.write(orjson.dumps(questions, option=orjson.OPT_INDENT_2))


async def main():
    parser = argparse.ArgumentParser()

    # Input / output
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--condition", type=str, default="")
    parser.add_argument("--system-msg", type=str, default="")
    parser.add_argument("--model-type", type=str, default=None)

    parser.add_argument("--data_path", type=str, default="ochat/evaluation/eval_data")
    parser.add_argument("--eval_sets", type=str, nargs="+", default=[])

    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--output_file",   type=str, default=None)
    parser.add_argument("--parallel",      type=int, default=16)

    args = parser.parse_args()

    await run_eval(**vars(args))

if __name__ == "__main__":
    asyncio.run(main())

from typing import Optional
import argparse
import os
import ast

import orjson
from vllm import LLM, SamplingParams

from transformers.utils.hub import cached_file
from evalplus.data import get_human_eval_plus, write_jsonl


def _function_exists(code, func_name):
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return True

    return False


def format_prompt(problem):
    return f"Provide a correct completion to the following code snippet.\n```\n{problem['prompt'].strip()}\n```"


def match_completions(problem, response):
    include_prefix = problem['prompt'].split('def')[0].strip() + "\n\n"

    for block in response.split("```"):
        # Sanitize block
        block = block.strip()
        if block.startswith("python"):
            block = block[len("python"):]

        # Check syntax
        try:
            code_completion = include_prefix + block
            if _function_exists(code_completion, problem["entry_point"]):
                return code_completion
        except SyntaxError:
            pass

    # Failure
    print (f"Cannot match:\n{response}")
    return response


def tokenize_questions(model_config: object, conv_template: object, questions: list, condition: Optional[str], system_msg: str):
    from ochat.config import Conversation, Message

    # Construct conversation
    prompt_indices = []
    conversations = []
    for idx, q in enumerate(questions):
        conversations.append(Conversation(
            items=[
                Message(role="user", content=q["prompt"]),
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


def run_codegen(
    model: str,
    condition: Optional[str],
    system_msg: str,

    output_file: Optional[str],
):
    from ochat.config import MODEL_CONFIG_MAP

    # Load model config
    with open(cached_file(path_or_repo_id=model, filename="openchat.json"), "r") as f:
        model_type = orjson.loads(f.read())["model_type"]

    model_config = MODEL_CONFIG_MAP[model_type]
    tokenizer = model_config.model_tokenizer_create(model)
    conv_template = model_config.conversation_template(tokenizer=tokenizer)

    # Init vLLM engine
    model_config = MODEL_CONFIG_MAP[model_type]
    engine = LLM(model,
                 max_num_batched_tokens=model_config.model_max_context)
    sampling_params = SamplingParams(temperature=0,
                                     max_tokens=model_config.model_max_context,
                                     stop_token_ids=conv_template.eot_tokens_,  # Override stop tokens
                                     ignore_eos=True)

    # Complete
    questions = [{"task_id": task_id, "problem": problem, "prompt": format_prompt(problem)}
                 for task_id, problem in get_human_eval_plus().items()]

    prompts, prompt_indices = tokenize_questions(model_config, conv_template, questions,
                                                 condition=condition, system_msg=system_msg)

    # calculate & fill in responses
    responses = engine.generate(prompt_token_ids=prompts, sampling_params=sampling_params)
    for idx, resp in zip(prompt_indices, responses):
        questions[idx]["completion"] = match_completions(questions[idx]["problem"], resp.outputs[0].text)

    # Write completions
    if output_file is None:
        output_file = f"ochat/evaluation/codegen_results/{os.path.basename(model)}_{condition}.jsonl"
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    write_jsonl(output_file, questions)


def main():
    parser = argparse.ArgumentParser()

    # Input / output
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--condition", type=str, default="")
    parser.add_argument("--system-msg", type=str, default="")

    parser.add_argument("--output_file", type=str, default=None)

    args = parser.parse_args()

    run_codegen(**vars(args))

if __name__ == "__main__":
    main()

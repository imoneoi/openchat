from typing import OrderedDict
import signal
import os
import json
import subprocess
import argparse
import time
import requests
import re
import coolname


MAX_CONTEXT = 4096


def find_models(path, prefix, ep_filter):
    run_name = '_'.join(coolname.generate(2))

    def generate_model_name(root, ep_number):
        return f"{prefix}{os.path.basename(root)}_ep{ep_number}_{run_name}"

    models = {}
    for root, dirs, _ in os.walk(path):
        for d in dirs:
            ep_match = re.match(r"ep_(\d+)", d)
            if not ep_match:
                continue

            if ep_filter and ep_match.group(1) != ep_filter:
                continue

            model_name = generate_model_name(root, ep_match.group(1))
            models[model_name] = os.path.join(root, d)

    # Sort and return the models dictionary as an OrderedDict
    return OrderedDict(sorted(models.items(), reverse=True, key=lambda x: x[0].split("_ep")[::-1]))


def run_mt_bench(mt_bench_path, model_name):
    working_dir = os.path.join(mt_bench_path, "fastchat", "llm_judge")

    # Skip if result exists
    if os.path.exists(os.path.join(working_dir, "data", "mt_bench", "model_answer", f"{model_name}.jsonl")):
        return

    # run mt bench
    commands = [
        f"python gen_api_answer.py --model {model_name} --max-tokens {MAX_CONTEXT} --parallel 128 --openai-api-base http://localhost:18888/v1",
        f"python gen_judgment.py --model-list {model_name} --parallel 8 --mode single",
        # f"python gen_judgment.py --model-list {model_name} --parallel 8 --mode pairwise-baseline",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = f'{env.get("PYTHONPATH", "")}:{mt_bench_path}'

    for command in commands:
        subprocess.run(command, shell=True, cwd=working_dir, env=env)


def run_vicuna_bench(mt_bench_path, model_name):
    working_dir = os.path.join(mt_bench_path, "fastchat", "llm_judge")

    # Skip if result exists
    if os.path.exists(os.path.join(working_dir, "data", "vicuna_bench", "model_answer", f"{model_name}.jsonl")):
        return

    # run mt bench
    commands = [
        f"python gen_api_answer.py --model {model_name} --max-tokens {MAX_CONTEXT} --parallel 128 --openai-api-base http://localhost:18888/v1 --bench-name vicuna_bench",
        f"python gen_judgment.py --model-list {model_name} --parallel 8 --mode pairwise-baseline --bench-name vicuna_bench",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = f'{env.get("PYTHONPATH", "")}:{mt_bench_path}'

    for command in commands:
        subprocess.run(command, shell=True, cwd=working_dir, env=env)


def create_alpaca_eval_config(alpacaeval_path, model_name):
    config_dir = os.path.join(alpacaeval_path, "src", "alpaca_eval", "models_configs", model_name.lower())
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, "configs.yaml")

    config_content = f"""{model_name.lower()}:
  prompt_template: "openchat-13b/prompt.txt"
  fn_completions: "openai_completions"
  completions_kwargs:
    openai_api_base: http://127.0.0.1:18888/v1
    requires_chatml: True
    sleep_time: 0

    model_name: "{model_name.lower()}"
    max_tokens: {MAX_CONTEXT}

    top_p: 1.0
    temperature: 0.7

    num_procs: 128

  pretty_name: "{model_name}"
  link: "https://github.com/imoneoi/openchat"
"""

    with open(config_path, "w") as f:
        f.write(config_content)


def run_alpaca_eval(alpacaeval_path, model_name):
    # Skip if result exists
    if os.path.exists(os.path.join(alpacaeval_path, "results", model_name.lower())):
        return

    # Create config
    create_alpaca_eval_config(alpacaeval_path, model_name)

    # Run
    command = f"python -m alpaca_eval.main evaluate_from_model --model_configs {model_name.lower()} --annotators_config alpaca_eval_gpt4"

    env = os.environ.copy()
    env["PYTHONPATH"] = f'{env.get("PYTHONPATH", "")}:{os.path.join(alpacaeval_path, "src")}'

    subprocess.run(command, shell=True, cwd=alpacaeval_path, env=env)


def wait_for_server(url):
    while True:
        try:
            response = requests.get(url)
            if response.status_code in [200, 404]:
                break
        except requests.exceptions.RequestException:
            pass

        time.sleep(1)


def main(path, prefix, ep_filter, mt_bench_path, alpacaeval_path):
    models = find_models(path, prefix, ep_filter)

    for i, (model_name, model_path) in enumerate(models.items()):
        print(f"Processing model {i + 1}/{len(models)}: {model_name}")

        print("Starting server...")
        server_command = f"python -m ochat.serving.openai_api_server --model {model_path} --engine-use-ray --worker-use-ray"
        server_process = subprocess.Popen(server_command, shell=True, preexec_fn=os.setsid)

        wait_for_server("http://127.0.0.1:18888/v1")
        print("Server is ready.")

        print("Running MT-bench...")
        run_mt_bench(mt_bench_path, model_name)

        print("Running AlpacaEval...")
        # run_alpaca_eval(alpacaeval_path, model_name)

        # print("Running Vicuna-bench")
        # run_vicuna_bench(mt_bench_path, model_name)

        print("Terminating server...")
        os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
        server_process.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="/ML-A100/home/csj/trained_models/openchat_mistral/1017", help="Path to the models directory")
    parser.add_argument("--prefix", default="gpt4correct_")
    parser.add_argument("--ep_filter", default=None, help="Filter epochs")

    parser.add_argument("--mt_bench_path", default="/ML-A100/home/csj/one_benchmarks/FastChat", help="Path to the MT-bench directory")
    parser.add_argument("--alpacaeval_path", default="/ML-A100/home/csj/one_benchmarks/alpaca_eval", help="Path to the AlpacaEval directory")
    args = parser.parse_args()

    main(**vars(args))

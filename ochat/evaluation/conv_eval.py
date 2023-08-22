import os
import json
import subprocess
import argparse
import time
import requests
import re
import os


MAX_CONTEXT = 4096


def find_models(path):
    models = {}
    for root, dirs, _ in os.walk(path):
        for d in dirs:
            if re.match(r"ep_\d+", d):
                model_name = os.path.basename(root)
                if model_name not in models:
                    models[model_name] = []
                models[model_name].append(os.path.join(root, d))

    for model_name in models:
        models[model_name] = max(models[model_name], key=lambda x: int(x.split("_")[-1]))
    return models


def modify_special_tokens_map(model_path):
    special_tokens_map_path = os.path.join(model_path, "special_tokens_map.json")
    with open(special_tokens_map_path, "r") as f:
        special_tokens_map = json.load(f)
        
    special_tokens_map["eos_token"] = "<|end_of_turn|>"

    with open(special_tokens_map_path, "w") as f:
        json.dump(special_tokens_map, f)


def run_mt_bench(mt_bench_path, model_name):
    # run mt bench
    commands = [
        f"python gen_api_answer.py --model {model_name} --max-tokens {MAX_CONTEXT} --parallel 128 --openai-api-base http://localhost:18888/v1",
        f"python gen_judgment.py --model-list {model_name} --parallel 3 --mode single",
        f"python gen_judgment.py --model-list {model_name} --parallel 3 --mode pairwise-baseline",
    ]
    for command in commands:
        subprocess.run(command, shell=True, cwd=mt_bench_path)


def create_alpaca_eval_config(alpacaeval_path, model_name, model_type):
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

    model_name: "{model_type}"
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
    command = f"python -m alpaca_eval.main evaluate_from_model --model_configs {model_name.lower()} --annotators_config alpaca_eval_gpt4"

    env = os.environ.copy()
    env["PYTHONPATH"] = f'{env.get("PYTHONPATH", "")}:{os.path.join(alpacaeval_path, "src")}'

    subprocess.run(command, shell=True, cwd=os.path.join(alpacaeval_path), env=env)


def wait_for_server(url):
    while True:
        try:
            response = requests.get(url)
            if response.status_code in [200, 404]:
                break
        except requests.exceptions.RequestException:
            pass

        time.sleep(1)


def main(model_type, path, mt_bench_path, alpacaeval_path):
    models = find_models(path)

    for i, (model_name, model_path) in enumerate(models.items()):
        print(f"Processing model {i + 1}/{len(models)}: {model_name}")
        
        modify_special_tokens_map(model_path)

        print("Starting server...")
        server_command = f"python -m ochat.serving.openai_api_server --model-type {model_type} --model {model_path} --engine-use-ray --worker-use-ray --max-num-batched-tokens 5120"
        server_process = subprocess.Popen(server_command, shell=True)

        wait_for_server("http://127.0.0.1:18888/v1")
        print("Server is ready.")

        print("Running MT-bench...")
        run_mt_bench(mt_bench_path, model_name)

        print("Running AlpacaEval...")
        create_alpaca_eval_config(alpacaeval_path, model_name, model_type)
        run_alpaca_eval(alpacaeval_path, model_name)

        print("Terminating server...")
        server_process.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="openchat_v3.2", help="Model type, e.g., 'gpt-4'")

    parser.add_argument("--path", default="/data2/one/trained_models/ablation", help="Path to the models directory")

    parser.add_argument("--mt_bench_path", default="/data/one/FastChat/fastchat/llm_judge", help="Path to the MT-bench directory")
    parser.add_argument("--alpacaeval_path", default="/data/one/alpaca_eval", help="Path to the AlpacaEval directory")
    args = parser.parse_args()

    main(**vars(args))

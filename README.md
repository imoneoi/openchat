# OpenChat: Advancing Open-source Language Models with Mixed-Quality Data</h1>

<div align="center">
  <img src="assets/logo_new.png" style="width: 65%">
</div>

<p align="center">
  <a href="https://openchat.team">Online Demo</a> •
  <a href="https://discord.gg/pQjnXvNKHY">Discord</a> •
  <a href="https://huggingface.co/openchat">Huggingface</a> •
  <a href="https://arxiv.org/pdf/2309.11235.pdf">Paper</a>
</p>

OpenChat is a collection of open-source language models, optimized and fine-tuned with [C-RLFT](https://arxiv.org/pdf/2309.11235.pdf), a strategy inspired by offline reinforcement learning, to learn from mixed-quality data without preference labels. We use approximately 80k ShareGPT conversations to deliver outstanding performance, despite our simple approach. Our ultimate goal is to develop a high-performance, commercially available, open-source large language model, and we are continuously making strides toward this vision.

**🤖 Ranked #1 among all open-source models on [AgentBench](https://github.com/THUDM/AgentBench)**

**🔥 Ranked #1 among 13B open-source models | 89.5% win-rate on [AlpacaEval](https://tatsu-lab.github.io/alpaca_eval/) | 7.19 score on [MT-bench](https://chat.lmsys.org/?leaderboard)**

**🕒 Exceptionally efficient padding-free fine-tuning, only requires 15 hours on 8xA100 80G**

**💲 FREE for commercial use under [Llama 2 Community License](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)**

[![DOI](https://zenodo.org/badge/645397533.svg)](https://zenodo.org/badge/latestdoi/645397533)

## News

- [2023/09/21] We released our paper [OpenChat: Advancing Open-source Language Models with Mixed-Quality Data](https://arxiv.org/pdf/2309.11235.pdf).

- [2023/09/03] We released the [OpenChat V3.2 SUPER](#models) model.

- [2023/08/04] We have launched an [Online Demo](https://openchat.team) featuring the latest version, OpenChat 3.2.

- [2023/07/30] We are thrilled to introduce the [OpenChat V3 model series](#models), based on Llama 2, and now available for free for commercial use!

- [2023/07/07] We released the [OpenChat V2 model series](#legacy-models).

- [2023/07/01] We released the [OpenChat V1 model series](#legacy-models).

## <a id="models"></a> Models

Our latest model is OpenChat 3.2 SUPER. We recommend using it for optimal conversational and instruction-following performance. This models is designed for English and have limited multilingual capabilities. They can be downloaded under the [Llama 2 Community License](https://ai.meta.com/resources/models-and-libraries/llama-downloads/).

To use these models, we highly recommend installing the OpenChat package by following the [installation guide](#installation) and using the OpenChat OpenAI-compatible API server by running the serving command from the table below. The server is optimized for high-throughput deployment using [vLLM](https://github.com/vllm-project/vllm) and can run on a GPU with at least 48GB RAM or two consumer GPUs with tensor parallelism. To enable tensor parallelism, append `--tensor-parallel-size 2` to the serving command.

Once started, the server listens at `localhost:18888` for requests and is compatible with the [OpenAI ChatCompletion API specifications](https://platform.openai.com/docs/api-reference/chat). Please refer to the example request below for reference. Additionally, you can use the [OpenChat Web UI](#web-ui) for a user-friendly experience.

If you want to deploy the server as an online service, you can use `--api-keys sk-KEY1 sk-KEY2 ...` to specify allowed API keys and `--disable-log-requests --disable-log-stats --log-file openchat.log` for logging only to a file. For security purposes, we recommend using an [HTTPS gateway](https://fastapi.tiangolo.com/es/deployment/concepts/#security-https) in front of the server.

<details>
  <summary>Example request (click to expand)</summary>

```bash
curl http://localhost:18888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openchat_v3.2",
    "messages": [{"role": "user", "content": "You are a large language model named OpenChat. Write a poem to describe yourself"}]
  }'
```

</details>

| Model              | Size | Context | Weights                                                            | Serving                                                                                                            |
|--------------------|------|---------|--------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| OpenChat 3.2 SUPER | 13B  | 4096    | [Huggingface](https://huggingface.co/openchat/openchat_v3.2_super) | `python -m ochat.serving.openai_api_server --model openchat/openchat_v3.2_super --engine-use-ray --worker-use-ray` |

For inference with Huggingface Transformers (slow and not recommended), follow the conversation template provided below:

<details>
  <summary>Conversation templates (click to expand)</summary>

```python
# Single-turn V3.2 (SUPER)
tokenize("GPT4 User: Hello<|end_of_turn|>GPT4 Assistant:")
# Result: [1, 402, 7982, 29946, 4911, 29901, 15043, 32000, 402, 7982, 29946, 4007, 22137, 29901]

# Multi-turn V3.2 (SUPER)
tokenize("GPT4 User: Hello<|end_of_turn|>GPT4 Assistant: Hi<|end_of_turn|>GPT4 User: How are you today?<|end_of_turn|>GPT4 Assistant:")
# Result: [1, 402, 7982, 29946, 4911, 29901, 15043, 32000, 402, 7982, 29946, 4007, 22137, 29901, 6324, 32000, 402, 7982, 29946, 4911, 29901, 1128, 526, 366, 9826, 29973, 32000, 402, 7982, 29946, 4007, 22137, 29901]
```

</details>

## <a id="benchmarks"></a> Benchmarks

We have evaluated our models using the two most popular evaluation benchmarks **, including AlpacaEval and MT-bench. Here we list the top models with our released versions, sorted by model size in descending order. The full version can be found on the [MT-bench](https://chat.lmsys.org/?leaderboard) and [AlpacaEval](https://tatsu-lab.github.io/alpaca_eval/) leaderboards.

To ensure consistency, we used the same routine as ChatGPT / GPT-4 to run these benchmarks. We started the OpenAI API-compatible server and set the `openai.api_base` to `http://localhost:18888/v1` in the benchmark program.

| **Model**                        | **Size** | **Context** | **Dataset Size** | **💲Free** | **AlpacaEval (win rate %)** | **MT-bench (win rate adjusted %)** | **MT-bench (score)** |
|----------------------------------|----------|-------------|------------------|-----------|-----------------------------|------------------------------------|----------------------|
|                                  |          |             |                  |           | **v.s. text-davinci-003**   | **v.s. ChatGPT**                   |                      |
| GPT-4                            | 1.8T*    | 8K          |                  | ❌         | 95.3                        | 82.5                               | 8.99                 |
| ChatGPT                          | 175B*    | 4K          |                  | ❌         | 89.4                        | 50.0                               | 7.94                 |
| Llama-2-70B-Chat                 | 70B      | 4K          | 2.9M             | ✅         | 92.7                        | 60.0                               | 6.86                 |
| **OpenChat 3.2 SUPER**           | **13B**  | **4K**      | **80K**          | ✅         | **89.5**                    | **57.5**                           | **7.19**             |
| Llama-2-13B-Chat                 | 13B      | 4K          | 2.9M             | ✅         | 81.1                        | 55.3                               | 6.65                 |
| WizardLM 1.2                     | 13B      | 4K          | 196K             | ✅         | 89.2                        | 53.1                               | 7.05                 |
| Vicuna 1.5                       | 13B      | 2K          | 125K             | ✅         | 78.8                        | 37.2                               | 6.57                 |

*: Estimated model size

**: The benchmark metrics represent a quantified measure of a subset of the model's capabilities. A win-rate greater than 50% does not necessarily indicate that the model is better than ChatGPT in all scenarios or for all use cases. It is essential to consider the specific tasks or applications for which the model was evaluated and compare the results accordingly.

## vLLM Eval

🚀 To ensure comprehensive evaluation of large language models (LLMs), we are working on developing a suite of accelerated standard benchmarks, including AGIEval, BBH, and Chain-of-Thought Hub, named vLLM Eval. This suite leverages the speedup provided by [vLLM](https://github.com/vllm-project/vllm) and allows us to finish the entire benchmark in just 5 minutes.

We will release the evaluation results as soon as they become available, so stay tuned!

## <a id="installation"></a> Installation

To use OpenChat, you need to install PyTorch, then you can install OpenChat via pip:

```bash
pip3 install ochat
```

If you encounter compatibility problems, you can try to create a new `conda` environment following the instructions below.

```bash
conda create -y --name openchat
conda activate openchat

conda install -y python=3.11
pip3 install torch torchvision torchaudio

pip3 install ochat
```

<details>
  <summary>In addition to pypi, you can also install from source (click to expand)</summary>

```bash
git clone https://github.com/imoneoi/openchat
cd openchat

pip3 install --upgrade pip  # enable PEP 660 support
pip3 install -e .
```
</details>

## <a id="web-ui"></a> Web UI

After lanuching the API server, you can interact with it using [OpenChat-UI](https://github.com/imoneoi/openchat-ui), which is a fork of Chatbot UI with support for OpenChat models.

To use OpenChat-UI, follow these steps:

1. Clone the OpenChat-UI repo:

```bash
git clone https://github.com/imoneoi/openchat-ui.git
```

2. Install Dependencies

```bash
npm i
```

3. Set the API host to the local server (or the address of the OpenChat server)

Create a `.env.local` file in the root of the OpenChat-UI repo with the following content:

```conf
OPENAI_API_HOST=http://localhost:18888
OPENAI_API_KEY=openchat-dummy-key
NEXT_PUBLIC_DEFAULT_TEMPERATURE=0.7
```

4. Run the App

```bash
npm run dev
```

## <a id="training"></a> OpenChat Model Training

The OpenChat training system utilizes padding-free training and the [Multipack Sampler](https://github.com/imoneoi/multipack_sampler), achieving a **3~10x** speedup compared to the conventional padded training. The V3 series can be trained in approximately 15 hours using eight A100 80GB GPUs.

## Installing DeepSpeed

First, ensure that the CUDA `nvcc` compiler is available in your environment. If it is not, install the CUDA toolkit that matches the version used by PyTorch.

Next, install DeepSpeed:

```bash
pip install deepspeed
```

### Preparing Your Data

To utilize the OpenChat trainer, prepare your SFT data into a JSON Lines format where each line corresponds to a `Conversation` object:

```python
class Message(BaseModel):
    role: str     # Must be "user" or "assistant"
    content: str  # Message content
    weight: Optional[float] = None  # Loss weight for this message. Typically 0 for user and 1 for assistant to supervise assistant's responses only


class Conversation(BaseModel):
    items: List[Message]  # All messages within the conversation
    condition: str = ""  # C-RLFT condition, can be any string or empty.
    system: str = ""  # System message for this conversation
```

For basic SFT, assign `weight` as `0` for human messages and `1` for assistant responses.

SFT example:

```json
{"items":[{"from":"user","content":"Hello","weight":0.0},{"from":"assistant","content":"Hi","weight":1.0},{"from":"user","content":"How are you today?","weight":0.0},{"from":"assistant","content":"I'm fine.","weight":1.0}],"system":""}
{"items":[{"from":"user","content":"Who are you?","weight":0.0},{"from":"assistant","content":"I'm OpenChat.","weight":1.0}],"system":"You are a helpful assistant named OpenChat."}
```

For C-RLFT, `condition` should be set as the class the conversation belongs to (e.g. `GPT3` or `GPT4`). The `weight` is assigned as `0` for human messages and `w` for assistant responses, where `w` is the weight of the class (e.g. `0.1` for `GPT3` and `1` for `GPT4`, as found in our C-RLFT paper).

C-RLFT example:

```json
{"items":[{"from":"user","content":"What is C-RLFT?","weight":0.0},{"from":"assistant","content":"C-RLFT is a method for improving open-source LLMs with mixed-quality data.","weight":1.0}],"condition":"GPT4","system":""}
{"items":[{"from":"user","content":"What is C-RLFT?","weight":0.0},{"from":"assistant","content":"I don't know.","weight":0.1}],"condition":"GPT3","system":""}
```

### Pre-tokenizing the Dataset

You'll then need to pre-tokenize the dataset using the command:

```bash
python -m ochat.data.generate_dataset --model-type openchat_v3.2 --model-path imone/Llama2_13B_with_EOT_token --in-files data.jsonl --out-prefix PRETOKENIZED_DATA_OUTPUT_PATH
```

We provide the pre-tokenized dataset of OpenChat 3.2 SUPER at the following location: [openchat/openchat_sharegpt_v3](https://huggingface.co/datasets/openchat/openchat_sharegpt_v3).

Note: The OpenChat conversation template requires an `<|end_of_turn|>` special token. The base model specified must include this token. We provide Llama 2 weights with this token added in the following HuggingFace repositories:

```
imone/Llama2_7B_with_EOT_token
imone/Llama2_13B_with_EOT_token
```

To add the end-of-turn token to a Llama base model, use the `convert_llama_weights_to_hf_add_tokens.py` in `scripts` directory:

```
python scripts/convert_llama_weights_to_hf_add_tokens.py --input_dir LLAMA_WEIGHT_DIR --model_size LLAMA_SIZE --output_dir OUTPUT_DIR --added_special_tokens \<\|end_of_turn\|\> \<\|PAD\|\>
```

### Launching the OpenChat Trainer

You can now launch the OpenChat trainer using the command below. Training a 13B model requires eight A/H100s with 80GB VRAM, while a 7B model can be trained with four A/H100s with 80GB VRAM or eight A/H100s with 40GB VRAM.

<details>

<summary>Training Commands (click to expand)</summary>

```bash
NUM_GPUS=8

deepspeed --num_gpus=$NUM_GPUS --module ochat.training_deepspeed.train \
          --model_path imone/Llama2_13B_with_EOT_token \
          --data_prefix PRETOKENIZED_DATA_OUTPUT_PATH \
          --save_path PATH_TO_SAVE_MODEL \
          --epochs 5 \
          --deepspeed \
          --deepspeed_config ochat/training_deepspeed/deepspeed_config.json
```

</details>

We recommend using the default hyperparameters as they have been carefully selected. Furthermore, the default learning rate is automatically determined based on the [inverse square-root rule](https://arxiv.org/abs/2006.09092).

The default hyperparameters utilized in the model training are as follows:

| **Hyperparameter** | Context | Batch size | Learning rate | AdamW betas | AdamW eps | Weight decay |
|--------------------|---------|------------|---------------|-------------|-----------|--------------|
| **Value**          | 4096    | 64         | Auto          | (0.9, 0.95) | 1e-5      | 0.1          |

## Limitations

**Foundation Model Limitations**
Despite its advanced capabilities, OpenChat is still bound by the limitations inherent in its foundation models. These limitations may impact the model's performance in areas such as:

 - Complex reasoning
 - Mathematical and arithmetic tasks
 - Programming and coding challenges

**Hallucination of Non-existent Information**
OpenChat may sometimes generate information that does not exist or is not accurate, also known as "hallucination". Users should be aware of this possibility and verify any critical information obtained from the model.

## License

Our OpenChat V3 models are licensed under the [Llama 2 Community License](https://ai.meta.com/resources/models-and-libraries/llama-downloads/). The code is distributed under the Apache License 2.0.

## Contact

💌 We are a student team from Tsinghua University, working on OpenChat, a project that requires additional computing power or LLMs API keys for further development. If you are interested in our project and would like to offer support, please feel free to reach out to us:

* Wang Guan (Project Leader) [imonenext at gmail dot com]
* Cheng Sijie [LeslieCheng0701 at outlook dot com]

We look forward to hearing from you and collaborating on this exciting project!

## Citation

```
@misc{wang2023openchat,
      title={OpenChat: Advancing Open-source Language Models with Mixed-Quality Data}, 
      author={Guan Wang and Sijie Cheng and Xianyuan Zhan and Xiangang Li and Sen Song and Yang Liu},
      year={2023},
      eprint={2309.11235},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@software{openchat,
  title = {{OpenChat: Advancing Open-source Language Models with Imperfect Data}},
  author = {Wang, Guan and Cheng, Sijie and Yu, Qiying and Liu, Changling},
  doi = {10.5281/zenodo.8105775},
  url = {https://github.com/imoneoi/openchat},
  version = {pre-release},
  year = {2023},
  month = {7},
}
```

## Acknowledgements

We would like to express our gratitude to GPT Desk Pte. Ltd., 01.AI company, and Tsinghua Laboratory of Brain and Intelligence (THBI) for their invaluable support.

We are also grateful to the developers of the following projects, which have contributed significantly to our research: [Llama 2](https://ai.meta.com/llama/), [self-instruct](https://arxiv.org/abs/2212.10560), [FastChat (Vicuna)](https://github.com/lm-sys/FastChat), [Alpaca](https://github.com/tatsu-lab/stanford_alpaca.git) and [StarCoder](https://github.com/bigcode-project/starcoder).

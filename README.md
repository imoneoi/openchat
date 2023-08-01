# OpenChat: Advancing Open-source Language Models with Imperfect Data</h1>

<div align="center">
  <img src="assets/logo_new.png" style="width: 65%">
</div>

OpenChat is a series of open-source language models based on supervised fine-tuning (SFT). We leverage the ~80k ShareGPT conversations with a conditioning strategy and weighted loss to achieve remarkable performance despite our simple methods. Our final vision is to develop a high-performance, open-source, and commercially available large language model, and we are continuously making progress.

**🔥 Rank #1 of 13B open-source models | 89.5% win-rate on [AlpacaEval](https://tatsu-lab.github.io/alpaca_eval/) | 7.01 score on [MT-bench](https://chat.lmsys.org/?leaderboard)**

**💲 FREE for commercial use under [Llama 2 Community License](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)**

**🕒 Super efficient padding-free finetuning for applications, only 10 hours on 8xA100 80G**

[![DOI](https://zenodo.org/badge/645397533.svg)](https://zenodo.org/badge/latestdoi/645397533)

## News

- [2023/07/30] We released the [OpenChat V3 model series](#models) based on Llama 2. Now free for commercial use!

- [2023/07/07] We released the [OpenChat V2 model series](#legacy-models).

- [2023/07/01] We released the [OpenChat V1 model series](#legacy-models).


## <a id="models"></a> Models

We recommend using our V3.2 model for optimal performance in multi-turn conversation scenarios since it utilizes multi-turn conditioning. However, if you require a model for single-turn instruction following, we suggest using V3.1, which uses a single condition schema and performs slightly better in this context. All models listed below are designed for English and have limited multilingual capabilities. They are available for download under the [Llama 2 Community License](https://ai.meta.com/resources/models-and-libraries/llama-downloads/).

To use these models, we highly recommend installing the OpenChat package by following the [installation guide](#installation) and using the OpenChat OpenAI-compatible API server by running the serving command from the table below. The server is optimized for high-throughput deployment using [vLLM](https://github.com/vllm-project/vllm) and can run on a GPU with at least 48GB RAM or two consumer GPUs with tensor parallelism. To enable tensor parallelism, append `--tensor-parallel-size 2` to the serving command.

Once started, the server listens at `localhost:18888` for requests and is compatible with the [OpenAI ChatCompletion API specifications](https://platform.openai.com/docs/api-reference/chat). Please refer to the example request below for reference. Additionally, you can use the [OpenChat Web UI](#web-ui) for a user-friendly experience.

If you want to deploy the server as an online service, you can use `--api-keys sk-KEY1 sk-KEY2 ...` to specify allowed API keys and `--disable-log-requests --disable-log-stats --log-file openchat.log` for logging only to a file. For security purposes, we recommend using an [HTTPS gateway](https://fastapi.tiangolo.com/es/deployment/concepts/#security-https) in front of the server.

*Note:* If you encounter IPv6 address errors, which is a known [vLLM issue](https://github.com/vllm-project/vllm/issues/570), please run `export NCCL_IGNORE_DISABLED_P2P=1` before starting the server.

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

| Model        | Size | Context | Weights                                                      | Serving                                                                                                                                                                      |
|--------------|------|---------|--------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| OpenChat 3.2 | 13B  | 4096    | [Huggingface](https://huggingface.co/openchat/openchat_v3.2) | `python -m ochat.serving.openai_api_server --model-type openchat_v3.2 --model openchat/openchat_v3.2 --engine-use-ray --worker-use-ray --max-num-batched-tokens 5120`        |
| OpenChat 3.1 | 13B  | 4096    | [Huggingface](https://huggingface.co/openchat/openchat_v3.1) | `python -m ochat.serving.openai_api_server --model-type openchat_v3.1_llama2 --model openchat/openchat_v3.1 --engine-use-ray --worker-use-ray --max-num-batched-tokens 5120` |

For inference with Huggingface Transformers (slow and not recommended), follow the conversation template provided below:

<details>
  <summary>Conversation templates (click to expand)</summary>

V3.2

```python
# Single-turn V3.2
tokenize("GPT4 User: Hello<|end_of_turn|>GPT4 Assistant:")
# Result: [1, 402, 7982, 29946, 4911, 29901, 15043, 32000, 402, 7982, 29946, 4007, 22137, 29901]

# Multi-turn V3.2
tokenize("GPT4 User: Hello<|end_of_turn|>GPT4 Assistant: Hi<|end_of_turn|>GPT4 User: How are you today?<|end_of_turn|>GPT4 Assistant:")
# Result: [1, 402, 7982, 29946, 4911, 29901, 15043, 32000, 402, 7982, 29946, 4007, 22137, 29901, 6324, 32000, 402, 7982, 29946, 4911, 29901, 1128, 526, 366, 9826, 29973, 32000, 402, 7982, 29946, 4007, 22137, 29901]
```

V3.1

```python
# Single-turn V3.1
tokenize("Assistant is GPT4<|end_of_turn|>User: Hello<|end_of_turn|>Assistant:")
# Result: [1, 4007, 22137, 338, 402, 7982, 29946, 32000, 4911, 29901, 15043, 32000, 4007, 22137, 29901]

# Multi-turn V3.1
tokenize("Assistant is GPT4<|end_of_turn|>User: Hello<|end_of_turn|>Assistant: Hi<|end_of_turn|>User: How are you today?<|end_of_turn|>Assistant:")
# Result: [1, 4007, 22137, 338, 402, 7982, 29946, 32000, 4911, 29901, 15043, 32000, 4007, 22137, 29901, 6324, 32000, 4911, 29901, 1128, 526, 366, 9826, 29973, 32000, 4007, 22137, 29901]
```

</details>

## <a id="benchmarks"></a> Benchmarks

We have evaluated our models using the two most popular evaluation benchmarks **, including AlpacaEval and MT-bench. Here we list the top models with our released versions, sorted by model size in descending order. The full version can be found on the [MT-bench](https://chat.lmsys.org/?leaderboard) and [AlpacaEval](https://tatsu-lab.github.io/alpaca_eval/) leaderboards.

To ensure consistency, we used the same routine as ChatGPT / GPT-4 to run these benchmarks. We started the OpenAI API-compatible server and set the `openai.api_base` to `http://localhost:18888/v1` in the benchmark program.

| **Model**        | **Size** | **Context** | **💲Free** | **AlpacaEval (win rate %)** | **MT-bench (win rate adjusted %)** | **MT-bench (score)** |
|------------------|----------|-------------|------------|-----------------------------|------------------------------------|----------------------|
|                  |          |             |            | **v.s. text-davinci-003**   | **v.s. ChatGPT**                   |                      |
| GPT-4            | 1.8T*    | 8K          | ❌         | 95.3                        | 82.5                               | 8.99                 |
| ChatGPT          | 175B*    | 4K          | ❌         | 89.4                        | 50.0                               | 7.94                 |
| Llama-2-70B-Chat | 70B      | 4K          | ✅         | 92.7                        |                                    | 6.86                 |
| **OpenChat 3.2** | **13B**  | **4K**      | ✅         | **89.1**                    | **51.6**                           | **7.01**             |
| **OpenChat 3.1** | **13B**  | **4K**      | ✅         | **89.5**                    | **50.0**                           | **6.65**             |
| Llama-2-13B-Chat | 13B      | 4K          | ✅         | 81.0                        |                                    | 6.65                 |
| Vicuna 1.3       | 13B      | 2K          | ❌         | 82.1                        | 37.5                               | 6.00                 |

*: Estimated model size

**: The benchmark metrics represent a quantified measure of a subset of the model's capabilities. A win-rate greater than 50% does not necessarily indicate that the model is better than ChatGPT in all scenarios or for all use cases. It is essential to consider the specific tasks or applications for which the model was evaluated and compare the results accordingly.

## vLLM Eval

🚀 To ensure comprehensive evaluation of large language models (LLMs), we are working on developing a suite of accelerated standard benchmarks, including AGIEval, BBH, and Chain-of-Thought Hub, named vLLM Eval. This suite leverages the speedup provided by [vLLM](https://github.com/vllm-project/vllm) and allows us to finish the entire benchmark in just 5 minutes.

We will release the evaluation results as soon as they become available, so stay tuned!

## <a id="installation"></a> Installation

To use OpenChat, you need to install CUDA and PyTorch, then you can install OpenChat via pip:

```bash
pip3 install ochat
```

If you want to train models, please also install FlashAttention 1.

```bash
pip3 install packaging ninja
pip3 install --no-build-isolation "flash-attn<2"
```

FlashAttention and vLLM may have compatibility issues. If you encounter these problems, you can try to create a new `conda` environment following the instructions below.

```bash
conda create -y --name openchat
conda activate openchat

conda install -y python
conda install -y cudatoolkit-dev -c conda-forge
pip3 install torch torchvision torchaudio

pip3 install packaging ninja
pip3 install --no-build-isolation "flash-attn<2"

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

## <a id="training"></a> Training

OpenChat leverages padding-free training and [Multipack Sampler](https://github.com/imoneoi/multipack_sampler), achieving a 3~6x speedup compared to commonly-used padded training. V3 series can be trained in 10 hours on 8x A100 80GB.

The hyperparameters used in training the models are listed as follows:

| **Hyperparameter** | Context | Batch size | Learning rate | AdamW betas | AdamW eps | Weight decay |
|--------------------|---------|------------|---------------|-------------|-----------|--------------|
| **Value**          | 4096    | 64         | 2e-5          | (0.9, 0.95) | 1e-5      | 0.1          |

To train using 8xA100 80GB, you should first clone the dataset for training:

```bash
git lfs install
git clone https://huggingface.co/datasets/openchat/openchat_sharegpt_v3
```

Then, run the following commands for V3.1 and V3.2 respectively:

<details>

<summary>Training commands (click to expand)</summary>

OpenChat V3.2:

```bash
NUM_GPUS=8

deepspeed --num_gpus=$NUM_GPUS --module ochat.training_deepspeed.train \
    --model-type openchat_v3.2 \
    --model_path imone/LLaMA2_13B_with_EOT_token \
    --data_path openchat_sharegpt_v3/sharegpt_v3.2 \
    --save_path PATH_TO_SAVE_MODEL \
    --epochs 5 \
    --lr 2e-5 \
    --eps 1e-5 \
    --batch_size_per_gpu 8 \
    --lr_warmup_steps 100 \
    --loss_balancing \
    --deepspeed \
    --deepspeed_config ochat/training_deepspeed/deepspeed_config.json
```

OpenChat V3.1:

```bash
NUM_GPUS=8

deepspeed --num_gpus=$NUM_GPUS --module ochat.training_deepspeed.train \
    --model-type openchat_v3.1_llama2 \
    --model_path imone/LLaMA2_13B_with_EOT_token \
    --data_path openchat_sharegpt_v3/sharegpt_v3.1_llama2 \
    --save_path PATH_TO_SAVE_MODEL \
    --epochs 3 \
    --lr 2e-5 \
    --eps 1e-5 \
    --batch_size_per_gpu 8 \
    --lr_warmup_steps 50 \
    --loss_balancing \
    --deepspeed \
    --deepspeed_config ochat/training_deepspeed/deepspeed_config.json
```

</details>

Please note that we added an EOT (end-of-turn) token to the Llama 2 base models. The embedding of the EOT token is initialized as the average of all existing token embeddings. The HF repo `imone/LLaMA2_13B_with_EOT_token` contains converted Llama weights with the aforementioned EOT token.

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

## TODO

**High-priority**

- [ ] Improving reasoning and math skills
- [ ] Training larger LLaMA models

**Low-priority**

- [ ] Mixing SFT data with pretraining data (e.g. RedPajama)
- [ ] Extending context by interpolating RoPE (requires mixing with pretraining data)
- [ ] Improving conversation splitting

## Citation

```
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

## <a id="section4"></a> Legacy Models (V1 and V2)

The following models are older versions of OpenChat and have inferior performance compared to the latest version. They are licensed under the same terms as the foundation model and will be deprecated in the next release.

 - OpenChat series: [LLaMA License](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md)
 - OpenCoder series: [BigCode OpenRAIL-M v1](https://huggingface.co/spaces/bigcode/bigcode-model-license-agreement)

To run the models on multiple GPUs with smaller VRAM, you can enable tensor parallelization, for example, using the `--tensor-parallel-size 2` flag.

<details>
  <summary>OpenChat V2 (click to expand)</summary>

| Model         | Size | Context | Weights                                                                 | Serve                                                                                                      |
|---------------|------|---------|-------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| openchat-v2-w | 13B  | 2K      | [Huggingface](https://huggingface.co/openchat/openchat_v2_w) | `python -m ochat.serving.openai_api_server --model-type openchat_v2 --model openchat/openchat_v2_w --engine-use-ray --worker-use-ray`      |
| openchat-v2   | 13B  | 2K      | [Huggingface](https://huggingface.co/openchat/openchat_v2)     | `python -m ochat.serving.openai_api_server --model-type openchat_v2 --model openchat/openchat_v2 --engine-use-ray --worker-use-ray`        |
</details>

<details>
  <summary>OpenChat V1 (click to expand)</summary>

| Model         | Size | Context | Weights                                                                 | Serve                                                                                                      |
|---------------|------|---------|-------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| openchat      | 13B  | 2K      | [Huggingface](https://huggingface.co/openchat/openchat)           | `python -m ochat.serving.openai_api_server --model-type openchat --model openchat/openchat --engine-use-ray --worker-use-ray`           |
| openchat8192  | 13B  | 8K      | [Huggingface](https://huggingface.co/openchat/openchat_8192) | `python -m ochat.serving.openai_api_server --model-type openchat_8192 --model openchat/openchat_8192 --engine-use-ray --worker-use-ray` |
| opencoderplus | 15B  | 8K      | [Huggingface](https://huggingface.co/openchat/opencoderplus) | `python -m ochat.serving.openai_api_server --model-type opencoder --model openchat/opencoderplus --engine-use-ray --worker-use-ray`     |

</details>

## Acknowledgements

We would like to express our gratitude to GPT Desk Pte. Ltd. and Tsinghua Laboratory of Brain and Intelligence (THBI) for their invaluable support.

We are also grateful to the developers of the following projects, which have contributed significantly to our research: [Llama 2](https://ai.meta.com/llama/), [self-instruct](https://arxiv.org/abs/2212.10560), [FastChat (Vicuna)](https://github.com/lm-sys/FastChat), [Alpaca](https://github.com/tatsu-lab/stanford_alpaca.git) and [StarCoder](https://github.com/bigcode-project/starcoder).

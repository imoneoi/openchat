# <img src="assets/logo.png" style="height:30pt; margin-right:5pt; display:inline-block; vertical-align:middle;">OpenChat: Advancing Open-source Language Models with Imperfect Data</h1>

OpenChat is a series of open-source language models based on supervised fine-tuning (SFT). We release two versions ([v1](#v1) and [v2](#v2)) models. Specifically, v1 uses only ~6K GPT-4 conversations directly filtered from the ~90K ShareGPT conversations, while v2 adopts cleaned ~80k ShareGPT conversations with a conditioning strategy and weighted loss. Despite our methods being simple, OpenChat has demonstrated remarkable performance. Our final vision is to develop a high-performance, open-source and commercially available large language model, and we are still moving on.


**💥 50.9% win-rate v.s. ChatGPT on [MT-bench](https://chat.lmsys.org/?leaderboard)**

**🚀 79.4% win-rate v.s. ChatGPT on [Vicuna GPT-4 eval](https://lmsys.org/blog/2023-03-30-vicuna/)**

**🔥 87.1% win-rate v.s. Davinci003 on [AlpacaEval](https://tatsu-lab.github.io/alpaca_eval/), rank #1 of open-source models**

**🤗 Using [~6K GPT-4 data](https://huggingface.co/datasets/openchat/openchat_sharegpt4_dataset) in v1 and [~80K cleaned ShareGPT data](https://huggingface.co/datasets/openchat/openchat_sharegpt4_dataset) in v2**

**🕒 Padding-free training, ~1 hour for v1 and ~6 hours for v2 (8xA100 80G)**

[![DOI](https://zenodo.org/badge/645397533.svg)](https://zenodo.org/badge/latestdoi/645397533)

## News

- [2023/07/07] We released the [OpenChat v2 model series](#v2).

- [2023/07/01] We released the [OpenChat v1 model series](#v1).


## <a id="section1"></a> Models and Dataset

*⚠️ Note:* The evaluation metrics represent a quantified measure of a subset of the model's capabilities. A win-rate greater than 50% does not necessarily indicate that the model is better than ChatGPT in all scenarios or for all use cases. It is essential to consider the specific tasks or applications for which the model was evaluated and compare the results accordingly.

#### <a id="v2"></a> OpenChat v2 
The OpenChat v2 family is inspired by offline reinforcement learning, including conditional behavior cloning (OpenChat-v2) and weighted behavior cloning (OpenChat-v2-w).

 - **[OpenChat-v2-w](https://huggingface.co/openchat/openchat_v2_w)**: ~80k cleaned ShareGPT data with conditioning and weighted loss, based on LLaMA-13B with a context length of 2048.
   - Achieves **50.9%** win-rate over ChatGPT on MT-bench.
   - Achieves **79.4%** win-rate over ChatGPT on Vicuna-bench.
   - Achieves **87.1%** win-rate over text-davinci-003 on AlpacaEval.
 - **[OpenChat-v2](https://huggingface.co/openchat/openchat_v2)**: ~80k cleaned ShareGPT data with only conditioning, based on LLaMA-13B with a context length of 2048.
   - Achieves **48.1%** win-rate over ChatGPT on MT-bench.
   - Achieves **80.6%** win-rate over ChatGPT on Vicuna-bench.
   - Achieves **85.0%** win-rate over text-davinci-003 on AlpacaEval.

#### <a id="v1"></a> OpenChat v1
The OpenChat v1 family is to validate the importance of data quality.
<details>
 <summary>Click to expand</summary>
 
 - **[OpenChat-v1](https://huggingface.co/openchat/openchat)**: only ~6k GPT-4 conversations, based on LLaMA-13B with a context length of 2048.
   - Achieves **73.1%** win-rate over ChatGPT on Vicuna-bench.
   - Achieves **80.9%** win-rate over text-davinci-003 on AlpacaEval.
 - **[OpenChat-v1-8192](https://huggingface.co/openchat/openchat_8192)**: only ~6k GPT-4 conversations, based on LLaMA-13B, with an extended context length of 8192.
   - Achieves **76.3%** win-rate over ChatGPT on Vicuna-bench.
   - Achieves **79.5%** win-rate over text-davinci-003 on AlpacaEval.
 - **[OpenCoderPlus-v1-8192](https://huggingface.co/openchat/opencoderplus)**: based on StarCoderPlus with a native context length of 8192.
   - Achieves **78.7%** win-rate over text-davinci-003 on AlpacaEval.
  
 </details>

#### Dataset:

  - **[openchat_sharegpt4_dataset](https://huggingface.co/datasets/openchat/openchat_sharegpt4_dataset)**: ~80k mixed (GPT-3.5 & GPT-4) and ~6k GPT-4 only data from ShareGPT.

## <a id="section2"></a> Model Evaluation

We have evaluated our models using the three most popular evaluation benchmarks, including AlpacaEval, MT-bench, and Vicuna benchmarks. 
Here we list the minimal version of benchmarks with our released models. The full version can be found on [MT-bench](https://chat.lmsys.org/?leaderboard) and [AlpacaEval](https://tatsu-lab.github.io/alpaca_eval/).

### Leaderboard

|                       |**AlpacaEval (win rate %)**| **MT-bench (win_rate_adjusted %)** | **Vicuna-bench (win_rate_adjusted %)**| **MT-bench (score)** | 
|-----------------------|--------------|---------------|--------------|--------------|
|                       |**v.s. Davinci003**     | **v.s. ChatGPT** | **v.s. ChatGPT** |   | 
| gpt4                  | 95.3         | 80.6          | 90.0         | 8.99         | 
| claude                | 88.4         | 62.8          | 76.3         | 7.90         |
| **openchat-v2-w-13b** | **87.1**     | **50.9**      | **79.4**     | **6.32**     |
|chatgpt (gpt-3.5-turbo)| 86.1         | 50.0          | 50.0         | 7.94         |
| **openchat-v2-13b**   | **85.0**     | **48.1**      | **80.6**     | **6.67**     |
| **openchat-13b**      | **80.9**     |               | **73.1**     |              |
| **openchat8192-13b**  | **79.5**     |               | **76.3**     |              |
| wizardlm-13b          | 75.3         | -             | -            | 6.35         |
| guanaco-65b           | 71.8         | -             | -            | 6.41         |
| vicuna-13b            | 70.4         | 34.1          | 50.0         | 6.39         |
| guanaco-33b           | 66.0         | -             | -            | 6.53         |
| text-davinci-003      | 50.0         | -             | -            | -            |
| falcon-40b-instruct   | 45.7         | -             | -            | 5.17         |

We are also trying to use extensive standard benchmarks to evaluate the performance of OpenChat, such as MMLU, we will release the evaluation results as soon as possible!

## <a id="section3"></a> Installation

To use OpenChat, you need to have CUDA and PyTorch installed. You can clone this repository and install the dependencies via pip:

```bash
git clone git@github.com:imoneoi/openchat.git
```

```bash
pip install --no-build-isolation flash-attn
pip install -r requirements.txt
```

*Note:* FlashAttention may have compatibility issues. If you encounter these problems, you can try to create a new "conda" environment and follow the instructions below.

```
conda install python
conda install cudatoolkit-dev -c conda-forge
pip3 install torch torchvision torchaudio

pip install packaging
pip install --no-build-isolation flash-attn

pip install -r requirements.txt
```

## <a id="section4"></a> Weights & Serving

We provide full weights of all models as Hugging Face repos. You can use the following commands to start a local API server at `http://localhost:18888`. Please note that models should be used under their foundation models' license.

The server is based on [vLLM](https://github.com/vllm-project/vllm/), to run on multiple GPUs with small VRAM, you can enable tensor parallel, e.g. `--tensor-parallel-size 2`

| Model         | Size | Context | Weights                                                                 | Serve                                                                                                      |
|---------------|------|---------|-------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| openchat-v2-w | 13B  | 2048    | [openchat/openchat-v2-w](https://huggingface.co/openchat/openchat_v2_w) | `python -m ochat.serving.openai_api_server --model_type openchat_v2 --model openchat/openchat_v2_w --engine-use-ray --worker-use-ray`      |
| openchat-v2   | 13B  | 2048    | [openchat/openchat-v2](https://huggingface.co/openchat/openchat_v2)     | `python -m ochat.serving.openai_api_server --model_type openchat_v2 --model openchat/openchat_v2 --engine-use-ray --worker-use-ray`        |

<details>
  <summary>OpenChat v1: Click to expand</summary>

| Model         | Size | Context | Weights                                                                 | Serve                                                                                                      |
|---------------|------|---------|-------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| openchat      | 13B  | 2048    | [openchat/openchat](https://huggingface.co/openchat/openchat)           | `python -m ochat.serving.openai_api_server --model_type openchat --model openchat/openchat --engine-use-ray --worker-use-ray`           |
| openchat8192  | 13B  | 8192    | [openchat/openchat_8192](https://huggingface.co/openchat/openchat_8192) | `python -m ochat.serving.openai_api_server --model_type openchat_8192 --model openchat/openchat_8192 --engine-use-ray --worker-use-ray` |
| opencoderplus | 15B  | 8192    | [openchat/opencoderplus](https://huggingface.co/openchat/opencoderplus) | `python -m ochat.serving.openai_api_server --model_type opencoder --model openchat/opencoderplus --engine-use-ray --worker-use-ray`     |

</details>

The server is compatible with the `ChatCompletions` protocol (please note that some functionalities are not fully supported) and the `openai` package. You can specify the server of `openai` package by setting:

```python
openai.api_base = "http://localhost:18888/v1"
```

We also provide a **Web UI** for a better user experience, please refer to the following section for details.


## <a id="section5"></a> Web UI

You can interact with the model using [OpenChat-UI](https://github.com/imoneoi/openchat-ui), which is a fork of Chatbot UI with support for OpenChat models.

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

## <a id="section6"></a> Model modifications

We added an EOT (end-of-turn) token to every base model. For LLaMA models, the embedding of EOT is initialized as the average of all existing token embeddings. For StarCoder models, the embedding of EOT is randomly initialized with 0.02 standard deviation.

For LLaMA-based models with 8192 context, the `max_position_embeddings` was set to 8192, and RoPE codes were extrapolated. An attempt to interpolate the RoPE code was made, but it resulted in a significant drop in performance without mixing pretraining data.

## <a id="section7"></a> Dataset

**🤗 Converted dataset available at [openchat_sharegpt4_dataset](https://huggingface.co/datasets/openchat/openchat_sharegpt4_dataset)**

The dataset used in the project is a cleaned and filtered version of ShareGPT, retaining only GPT-4 conversations. The original ShareGPT contained approximately 90K conversations, and only 6K cleaned GPT-4 conversations were retained for fine-tuning.

The cleaned GPT-4 conversations were combined with conversation templates and end-of-turn tokens, then cut to the context limit of the model (further content was simply discarded).

To run the data pipeline, execute the following command:

```bash
./ochat/data/run_data_pipeline.sh INPUT_FOLDER OUTPUT_FOLDER
```

The input folder should contain a `ShareGPT` folder with `.html` files for each ShareGPT conversation page inside.

The data pipeline consists of three steps:

- Cleaning: HTML cleaning and conversion to Markdown, removing conversations with the wrong format, removing conversations with blocked words, and hash-based exact deduplication
- Filtering: Preserving only conversations marked as `Model: GPT-4`
- Converting: Converting and tokenizing all conversations for finetuning

**The final converted dataset follows the format:**

*MODEL_TYPE.train.json / .eval.json*

```
[
    [token_id_list, supervise_mask_list],
    [token_id_list, supervise_mask_list],
    ...
]
```

*MODEL_TYPE.train.text.json / .eval.text.json*

Plain text decoded from `token_id_list`

### Dataset visualization

We provide a tool for visualizing the embeddings of conversations. To use this tool, open `ochat/visualization/ui/visualizer.html` using a browser and drag `MODEL_TYPE.visualizer.json` into the webpage. Click on 3D plot points to show the corresponding conversation.

The embeddings are created using `openai_embeddings.py` to calculate embeddings of conversations, then UMAP dimension reduction and K-Means coloring with `dim_reduction.ipynb`.

![embedding](assets/embeddings.svg)

## <a id="section7"></a> Training

OpenChat V2 leverages padding-free training and [Multipack Sampler](https://github.com/imoneoi/multipack_sampler), achieving a 3x speedup compared to the last release. Now the V2 series can be trained in 6 hours and the V1 series in 1 hour.

The hyperparameters used in training the models are the same across all models:

| Global Batch Size | Learning rate | Epochs | Length Grouping | Warmup Ratio | Weight decay |
| --- | --- | --- | --- | --- | --- |
| 128 | 2e-5 | 5 | True | 0.03 | 0 |

To train using 8xA100 80GB:

```bash
NUM_GPUS=8

deepspeed --num_gpus=$NUM_GPUS --module ochat.training_deepspeed.train \
    --model_type MODEL_TYPE \
    --model_path BASE_MODEL_PATH \
    --save_path TARGET_FOLDER \
    --epochs 5 \
    --data_path DATASET_PATH \
    --deepspeed \
    --deepspeed_config ochat/training_deepspeed/deepspeed_config.json

```

## <a id="section8"></a> Evaluation

The same routine as ChatGPT / GPT-4 was used to run other benchmarks or evaluations such as AlpacaEval. Simply run the API server and set the `openai.api_base` of the benchmark program.

## Limitations

**Foundation Model Limitations**
Despite its advanced capabilities, OpenChat is still bound by the limitations inherent in its foundation models. These limitations may impact the model's performance in areas such as:

 - Complex reasoning
 - Mathematical and arithmetic tasks
 - Programming and coding challenges

**Hallucination of Non-existent Information**
OpenChat may sometimes generate information that does not exist or is not accurate, also known as "hallucination". Users should be aware of this possibility and verify any critical information obtained from the model.

## TODO

**High-priority**

- [ ] Improving reasoning and math skills
- [ ] Updating performance on more standard benchmarks
- [ ] Training larger LLaMA models

**Low-priority**

- [ ] Mixing SFT data with pretraining data (e.g. RedPajama)
- [ ] Extending context by interpolating RoPE (requires mixing with pretraining data)
- [ ] Improving conversation splitting

## License

Our weight license is subject to their corresponding base model. For example, OpenChat and OpenChat-8192 are the same as the model [License](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) of LLaMA for non-commercial use only, while OpenCoderPlus is under the [License](https://huggingface.co/blog/starcoder) of StarCoder. Furthermore, we should follow [Privacy Practices](https://chrome.google.com/webstore/detail/sharegpt-share-your-chatg/daiacboceoaocpibfodeljbdfacokfjb) of ShareGPT. The code is released under Apache License 2.0.

## Contact

💓 We are a student team from Tsinghua University. Considering that we hope to further move on our OpenChat, we need support for more computing power or LLMs API keys. If you are interested in our OpenChat, welcome to contact Wang Guan (Project Leader; imonenext@gmail.com) or Cheng Sijie (LeslieCheng0701@outlook.com).

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

## Acknowledgements

We thank the great work by [LLaMA](https://arxiv.org/abs/2302.13971), [self-instruct](https://arxiv.org/abs/2212.10560), [FastChat (Vicuna)](https://github.com/lm-sys/FastChat), [Alpaca](https://github.com/tatsu-lab/stanford_alpaca.git) and [StarCoder](https://github.com/bigcode-project/starcoder).

We also thank the great support by GPT Desk Pte. Ltd. and Tsinghua Laboratory of Brain and Intelligence (THBI).

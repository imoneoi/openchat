"""
Generate training data based on conversations

Usage: python -m ochat.data.generate_data --in-file sharegpt_gpt4.json --tokenizer-name HF_REPO_NAME --out-dir .
"""

import argparse
import os
from copy import deepcopy

import orjson
import ray
import pyarrow
from pyarrow import parquet


IGNORE_TOKEN_ID = -100


def _split(a, n):
    # Split list a to n chunks
    # https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m): (i+1)*k+min(i+1, m)] for i in range(n)]


def add_token_group(output, max_context, max_group_len, tokens_list, masks_list, weights_list, numseq_list):
    assert max_group_len >= max_context

    EMPTY_RESULTS = {
        "total_length": 0,
        "num_seqs": 0,

        "seqlens": [],
        "nz_input_ids": [],
        "nz_position_ids": [],
        "nz_shifted_label_ids": [],
        "nz_shifted_loss_weights": []
    }

    results = deepcopy(EMPTY_RESULTS)
    for tokens, masks, weights, numseq in zip(tokens_list, masks_list, weights_list, numseq_list):
        # cut
        if max_context is not None:
            tokens  = tokens[:max_context]
            masks   = masks[:max_context]
            weights = weights[:max_context]

        # length
        length = len(tokens)
        assert len(masks) == length
        assert len(weights) == length

        if sum(masks) == 0:
            continue

        # slice groups
        if results["total_length"] + length > max_group_len:
            for k, v in results.items():
                output[k].append(v)

            results = deepcopy(EMPTY_RESULTS)

        # labels
        labels = [(t if m else IGNORE_TOKEN_ID) for t, m in zip(tokens, masks)]

        # populate results
        results["total_length"] += length
        results["num_seqs"] += numseq
        results["seqlens"].append(length)

        results["nz_input_ids"].extend(tokens)
        results["nz_position_ids"].extend(list(range(length)))

        results["nz_shifted_label_ids"].extend(labels[1:]     + [IGNORE_TOKEN_ID])
        results["nz_shifted_loss_weights"].extend(weights[1:] + [0.0])

        for m, w in zip(masks, weights):
            assert m == (w != 0)

    # Prepend to output
    if results["total_length"] > 0:
        for k, v in results.items():
            output[k].append(v)


@ray.remote
def convert_prm800k_batch(model_type: str, model_path: str, batch: list, field_names: list, max_group_len: int, wrong_weight: float, no_neutral: bool):
    from ochat.config.model_config import MODEL_CONFIG_MAP

    # Tokenization
    model_config = MODEL_CONFIG_MAP[model_type]
    tokenizer = model_config.model_tokenizer_create(model_path)

    def _tokenize(text):
        """Tokenize text AND special tokens"""
        return tokenizer.convert_tokens_to_ids(tokenizer._tokenize(text))

    def _tokenize_special(special_name):
        return [tokenizer.convert_tokens_to_ids(special_name)]

    # Constants
    PREFIX_MAP = {
        True:  _tokenize_special(model_config.bos_token) + _tokenize("Below are the steps that lead to a correct solution to the problem.")   + _tokenize_special(model_config.eot_token),
        False: _tokenize_special(model_config.bos_token) + _tokenize("Below are the steps that lead to an incorrect solution to the problem.") + _tokenize_special(model_config.eot_token),
    }
    STEP_FN = lambda text: _tokenize(text + "\n")

    # Generate data
    outputs = {k: [] for k in field_names}
    for c in batch:
        c = orjson.loads(c)

        # Skip if given up / invalid problem
        if c["label"]["finish_reason"] not in ["solution", "found_error"]:
            continue

        # tokenize problem
        problem_tokens = _tokenize(c["question"]["problem"]) + _tokenize_special(model_config.eot_token)

        # find number of steps
        if "pre_generated_steps" in c["question"]:
            num_steps = len(c["question"]["pre_generated_steps"])
        else:
            num_steps = len(c["label"]["steps"])

        # Convert process
        tokens_list = []
        masks_list = []
        weights_list = []
        numseq_list = []

        hist_steps   = []
        for step_idx, step_data in enumerate(c["label"]["steps"]):
            # find chosen step
            chosen_step = None
            if step_data["chosen_completion"] is not None:
                chosen_step = step_data["completions"][step_data["chosen_completion"]]
            else:
                if step_data["human_completion"] is not None:
                    chosen_step = step_data["human_completion"]
                else:
                    ref_text = c["question"]["pre_generated_steps"][step_idx]
                    for completion in step_data["completions"]:
                        if completion["text"] == ref_text:
                            chosen_step = completion
                            break

            if chosen_step["rating"] is None and (chosen_step["source"] == "human"):
                chosen_step["rating"] = 0

            # All leaves
            for alt_step in step_data["completions"]:
                if alt_step["rating"] is None:
                    continue  # Skip no rating
                if no_neutral and alt_step["rating"] == 0:
                    continue  # Skip neutral

                is_correct = alt_step["rating"] >= 0
                t_weight = 1.0 if is_correct else wrong_weight

                if t_weight == 0:
                    continue  # Skip weight 0

                t_hist = PREFIX_MAP[is_correct if wrong_weight >= 0 else True] + problem_tokens + hist_steps
                t_now  = STEP_FN(alt_step["text"])

                if (step_idx == num_steps - 1) and ("# Answer" in alt_step["text"]):
                    t_now += _tokenize_special(model_config.eot_token)

                tokens_list.append(t_hist + t_now)
                masks_list.append([False] * len(t_hist) + [True]                  * len(t_now))
                weights_list.append([0.]  * len(t_hist) + [t_weight / len(t_now)] * len(t_now))
                numseq_list.append(1)

            # add hist tokens
            hist_steps.extend(STEP_FN(chosen_step["text"]))

        # Add to results (shuffled)
        for tokens, mask, weights, numseqs in zip(tokens_list, masks_list, weights_list, numseq_list):
            add_token_group(outputs, model_config.model_max_context, max_group_len,
                            [tokens], [mask], [weights], [numseqs])

    return outputs


def generate_split(model_type: str, model_path: str, conversations: list, split_name: str, out_prefix: str, max_group_len: int, wrong_weight: float, no_neutral: bool, num_cpus: int = os.cpu_count()):
    # schema
    metadata = {
        "model_type": model_type
    }
    schema = [
        pyarrow.field("total_length", pyarrow.int32()),
        pyarrow.field("num_seqs", pyarrow.int32()),

        pyarrow.field(f"seqlens", pyarrow.list_(pyarrow.int32())),
        pyarrow.field(f"nz_input_ids", pyarrow.list_(pyarrow.int32())),
        pyarrow.field(f"nz_position_ids", pyarrow.list_(pyarrow.int32())),
        pyarrow.field(f"nz_shifted_label_ids", pyarrow.list_(pyarrow.int32())),
        pyarrow.field(f"nz_shifted_loss_weights", pyarrow.list_(pyarrow.float32()))
    ]

    schema = pyarrow.schema(schema, metadata={"metadata_json": orjson.dumps(metadata)})

    # launch remote workers
    ray.init(num_cpus=num_cpus)

    handles = [convert_prm800k_batch.remote(
        model_type=model_type,
        model_path=model_path,
        batch=batch,
        field_names=schema.names,
        max_group_len=max_group_len,
        wrong_weight=wrong_weight,
        no_neutral=no_neutral
    ) for batch in _split(conversations, num_cpus)]

    # aggegrate results
    results = {k: [] for k in schema.names}
    for handle in handles:
        batch_result = ray.get(handle)

        for k, v in batch_result.items():
            results[k].extend(v)

    print (f"Total sequences: {sum(results['num_seqs'])}")

    parquet.write_table(pyarrow.Table.from_pydict(results, schema=schema), f"{out_prefix}.{split_name}.parquet")

    ray.shutdown()


def generate_dataset(model_type, model_path, in_path, out_prefix, max_group_len, wrong_weight, no_neutral):
    # Load PRM800K
    SPLITS = {
        "train": ["phase1_train.jsonl", "phase2_train.jsonl"],
        "eval":  ["phase1_test.jsonl",  "phase2_test.jsonl"],
    }

    for split, files in SPLITS.items():
        data = []
        for filename in files:
            with open(os.path.join(in_path, filename), "r") as f:
                data.extend(f.readlines())

        generate_split(model_type, model_path, data, split, out_prefix, max_group_len, wrong_weight, no_neutral)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, default="discernia")
    parser.add_argument("--model-path", type=str, default="imone/deprecated_LLaMA2_13B_with_EOT_token")
    parser.add_argument("--max-group-len", type=int, default=32768)
    parser.add_argument("--wrong-weight", type=float, default=1.0)
    parser.add_argument("--no-neutral", action="store_true")

    parser.add_argument("--in-path", type=str, required=True)
    parser.add_argument("--out-prefix", type=str,required=True)
    args = parser.parse_args()

    generate_dataset(**vars(args))

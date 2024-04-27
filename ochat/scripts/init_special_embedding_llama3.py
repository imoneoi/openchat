import argparse

import transformers
import torch


def init_eot_embedding_llama3(model_path, output_dir, special_tokens=["<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>"], mean_cutoff=128000, dtype=torch.bfloat16):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, torch_dtype=dtype)

    assert model.model.embed_tokens.weight.shape[0] >= mean_cutoff
    assert model.lm_head.weight.shape[0]            >= mean_cutoff

    with torch.no_grad():
        for token in special_tokens:
            token_id = tokenizer.convert_tokens_to_ids(token)

            print (f"Token {token} ID {token_id}")

            model.model.embed_tokens.weight[token_id] = torch.mean(model.model.embed_tokens.weight[:mean_cutoff].to(torch.float32), dim=0).to(dtype)
            model.lm_head.weight[token_id]            = torch.mean(model.lm_head.weight[:mean_cutoff].to(torch.float32), dim=0).to(dtype)

    # Save
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        help="Location of model, or HuggingFace repo ID",
    )
    parser.add_argument(
        "--output-dir",
        help="Location to write resulting model and tokenizer",
    )

    init_eot_embedding_llama3(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()

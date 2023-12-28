import argparse

import transformers
import torch


def modify_eos_embeddings(model_path, output_dir, eos_token):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)

    eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)

    print (f"EOS Token {eos_token} ID {eos_token_id}")
    with torch.no_grad():
        model.model.embed_tokens.weight[eos_token_id] = torch.mean(model.model.embed_tokens.weight, dim=0)
        model.lm_head.weight[eos_token_id]            = torch.mean(model.lm_head.weight, dim=0)

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
    parser.add_argument(
        "--eos-token",
        default="</s>",
        help="EOS Token",
    )

    modify_eos_embeddings(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()

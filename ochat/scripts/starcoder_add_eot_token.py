import argparse

import transformers


def add_tokens(input_dir, output_dir, add_tokens):
    # Add tokens for tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(input_dir)
    num_new_tokens = tokenizer.add_special_tokens({"additional_special_tokens": add_tokens})
    tokenizer.save_pretrained(output_dir)

    # Add token embeddings for model
    model = transformers.AutoModelForCausalLM.from_pretrained(input_dir, low_cpu_mem_usage=True)

    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

    model.save_pretrained(output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="Location or name for StarCoder model")
    parser.add_argument("--output_dir", type=str, help="Output location")
    parser.add_argument("--add_tokens", type=str, nargs="+", help="Add tokens to model and tokenizer",)

    args = parser.parse_args()
    add_tokens(**vars(args))


if __name__ == "__main__":
    main()

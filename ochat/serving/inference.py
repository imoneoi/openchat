from typing import List, Optional, Dict

import torch
import transformers

from ochat.config.model_config import ModelConfig


@torch.jit.script
def sample_top_p(logits: torch.Tensor, temperature: float, top_p: float):
    # top-p sampling
    # Reference:
    # https://github.com/huggingface/transformers/blob/master/src/transformers/generation_logits_process.py

    # logits: [BS, NToken]
    device = logits.device
    # WARNING: Sort first then Softmax, otherwise logits will be too small
    # sort & cumsum
    sorted_logits, sorted_indices = torch.sort(logits / temperature, dim=-1, descending=True)
    sorted_probs = sorted_logits.softmax(dim=-1)
    # remove any > p
    sorted_probs_to_remove = sorted_probs.cumsum(dim=-1) > top_p
    sorted_probs_to_remove[:, 0] = 0  # keep first
    sorted_probs[sorted_probs_to_remove] = 0

    return sorted_indices[torch.arange(sorted_indices.shape[0], device=device),
                            torch.multinomial(sorted_probs, num_samples=1).squeeze(1)]


def sample(logits: torch.Tensor, temperature: float, top_p: float):
    # TODO: Fast sampling (no top-p) kernel

    # Deterministic kernel
    if temperature < 1e-5 or top_p < 1e-8:
        return torch.argmax(logits, dim=-1)

    # Fast Top-P kernel
    return sample_top_p(logits, temperature, top_p)


@torch.inference_mode()
def generate_stream(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    model_config: ModelConfig,

    conversation: List[Dict[str, str]],
    # Generation
    max_generated_tokens: int = int(1e6),
    stream_period: Optional[int] = None,
    # Settings
    temperature: Optional[float] = 1.0,
    top_p: Optional[float] = 1.0
):
    # Tokenization
    def _tokenize(text):
        return tokenizer.convert_tokens_to_ids(tokenizer._tokenize(text))

    def _tokenize_special(special_name):
        return tokenizer.convert_tokens_to_ids(special_name)

    def _detokenize_with_space(space, token_ids):
        if not len(token_ids):
            return ""

        text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=True))
        first_token = tokenizer.convert_ids_to_tokens(token_ids[0])

        if first_token.startswith(space) and not text.startswith(" "):
            return " " + text
        return text

    # Get prompts
    conversation.append({"from": model_config.ai_role})
    prompt_tokens, _ = model_config.generate_conversation_template(_tokenize, _tokenize_special, conversation)
    eot_token        = _tokenize_special(model_config.eot_token)
    space_token      = tokenizer.tokenize(" ")[0][0]

    # Streaming generation
    max_generated_tokens = min(max_generated_tokens, model_config.model_max_context - len(prompt_tokens) - 1)
    generated_tokens     = torch.zeros((1, max_generated_tokens), dtype=torch.long, device=model.device)
    generated_token_idx  = 0

    # Pregeneration
    logits, past_key_values = model(input_ids=torch.as_tensor(prompt_tokens, device=model.device).unsqueeze(0),
                                    use_cache=True, return_dict=False)

    token = sample(logits[:, -1], temperature, top_p)
    generated_tokens[:, generated_token_idx] = token
    generated_token_idx += 1

    # Generation
    finish_reason = "length"

    streamed_token_idx = 0
    while generated_token_idx < max_generated_tokens:
        if token.item() == eot_token:
            finish_reason = "stop"
            break

        logits, past_key_values = model(input_ids=token.unsqueeze(1), past_key_values=past_key_values,
                                        use_cache=True, return_dict=False)

        token = sample(logits[:, -1], temperature, top_p)
        generated_tokens[:, generated_token_idx] = token
        generated_token_idx += 1

        if (stream_period is not None) and (0 == generated_token_idx % stream_period):
            streamed = _detokenize_with_space(space_token, generated_tokens[0, streamed_token_idx: generated_token_idx].tolist())
            if "\ufffd" not in streamed:  # Ensure whole UTF-8 (no unknown tokens 0xFFFD)
                yield True, streamed
                streamed_token_idx = generated_token_idx

    yield True, _detokenize_with_space(space_token, generated_tokens[0, streamed_token_idx: generated_token_idx].tolist())
    yield False, finish_reason

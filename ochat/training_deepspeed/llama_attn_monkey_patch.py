from typing import Optional, Tuple
import warnings

import torch
from torch import nn

import transformers
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    attn_train_mode: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # qkv
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    # RoPE
    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    # caching
    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # flash attn
    if attn_train_mode:
        # Train mode, only causal mask
        assert attention_mask is None
        attn_output = nn.functional.scaled_dot_product_attention(query_states, key_states, value_states, is_causal=True)

    else:
        # Other modes, use attention mask
        attn_output = nn.functional.scaled_dot_product_attention(query_states, key_states, value_states, attention_mask)

    # projection
    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    assert not output_attentions, "output_attentions is not supported."
    return attn_output, None, past_key_value


def replace_llama_attn(attn_train_mode: bool = False):
    warnings.warn("LLaMA Attention patch enabled!")

    if attn_train_mode:
        warnings.warn("Training mode attention enabled. This have no additional mask instead of causal, and must be used on right-padded autoregressive training only.")

        # Attention mask is not used.
        transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = lambda *args, **kwargs: None

        # Enable FlashAttention only
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)

    # Forward wrapper
    def _attn_forward(*args, **kwargs):
        return forward(*args, **kwargs, attn_train_mode=attn_train_mode)

    transformers.models.llama.modeling_llama.LlamaAttention.forward = _attn_forward

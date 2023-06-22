from typing import Optional, Tuple
import logging

import torch
from einops import rearrange

import transformers

from flash_attn.flash_attn_interface import flash_attn_unpadded_func
from flash_attn.bert_padding import pad_input, unpad_input, index_first_axis


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # q, k:     [BS, T, num_heads, head_dim]
    # position_ids: [BS, T]
    # cos, sin: [1, 1, seq_len, head_dim]

    cos = cos.squeeze((0, 1))  # [seq_len, head_dim]
    sin = sin.squeeze((0, 1))  # [seq_len, head_dim]
    cos = cos[position_ids].unsqueeze(2)  # [BS, T, 1, head_dim]
    sin = sin[position_ids].unsqueeze(2)  # [BS, T, 1, head_dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    BS, T, _ = hidden_states.shape

    # qkv proj, shape: [BS, T, num_heads, head_dim]
    query_states = self.q_proj(hidden_states).view(BS, T, self.num_heads, self.head_dim)
    key_states = self.k_proj(hidden_states).view(BS, T,   self.num_heads, self.head_dim)
    value_states = self.v_proj(hidden_states).view(BS, T, self.num_heads, self.head_dim)

    # RoPE
    kv_seq_len = T
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[1]

    cos, sin = self.rotary_emb(query_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    # Cache
    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states   = torch.cat([past_key_value[0], key_states],   dim=1)
        value_states = torch.cat([past_key_value[1], value_states], dim=1)

    past_key_value = (key_states, value_states) if use_cache else None

    # unpad
    if attention_mask is not None:
        # q
        unpadded_q, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(query_states, attention_mask[..., -T:])

        # kv
        unpadded_k, indices_k, cu_seqlens_k, max_seqlen_k = unpad_input(key_states, attention_mask)
        unpadded_v                                        = index_first_axis(rearrange(value_states, 'b s ... -> (b s) ...'), indices_k)
    else:
        # q
        unpadded_q   = query_states.view(-1, self.hidden_size)

        max_seqlen_q = T
        cu_seqlens_q = torch.arange(0, (BS + 1) * T, step=T, dtype=torch.int32, device=hidden_states.device)

        # kv
        unpadded_k   = key_states.view(-1, self.hidden_size)
        unpadded_v   = value_states.view(-1, self.hidden_size)

        max_seqlen_k = kv_seq_len
        cu_seqlens_k = torch.arange(0, (BS + 1) * kv_seq_len, step=kv_seq_len, dtype=torch.int32, device=hidden_states.device)

    # flash attn
    assert not output_attentions, "output_attentions is not supported."

    attn_output = flash_attn_unpadded_func(
        q=unpadded_q, k=unpadded_k, v=unpadded_v,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
        dropout_p=0.0, causal=True)

    # attn_output: [num_heads, total_nnz, head_dim]
    attn_output = attn_output.view(-1, self.hidden_size)

    if attention_mask is not None:
        attn_output = pad_input(attn_output, indices_q, BS, T)
    else:
        attn_output = attn_output.view(BS, T, self.hidden_size)

    return self.o_proj(attn_output), None, past_key_value


# Disable the transformation of the attention mask in LlamaModel as the flash attention
# requires the attention mask to be the same as the key_padding_mask
def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
    # [bsz, seq_len]
    return attention_mask


# NOTE: Support cache but slower, for inference usage.
def replace_llama_attn():
    cuda_major, cuda_minor = torch.cuda.get_device_capability()
    if cuda_major < 8:
        logging.warning(
            "Flash attention is only supported on A100 or H100 GPU during training due to head dim > 64 backward."
            "ref: https://github.com/HazyResearch/flash-attention/issues/190#issuecomment-1523359593"
        )

    transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask
    transformers.models.llama.modeling_llama.LlamaAttention.forward = forward

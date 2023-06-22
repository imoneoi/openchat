from typing import Optional, Tuple
import logging

import torch

import transformers

from flash_attn.flash_attn_interface import flash_attn_unpadded_func
from flash_attn.bert_padding import pad_input, unpad_input


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def unpadded_apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # q, k:     [total_nnz, num_heads, head_dim]
    # position_ids: [total_nnz]
    # cos, sin: [1, 1, seq_len, head_dim]

    cos = cos.squeeze((0, 1))  # [seq_len, head_dim]
    sin = sin.squeeze((0, 1))  # [seq_len, head_dim]
    cos = cos[position_ids].unsqueeze(1)  # [total_nnz, 1, head_dim]
    sin = sin[position_ids].unsqueeze(1)  # [total_nnz, 1, head_dim]
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

    # unpad
    if attention_mask is not None:
        unpadded_hidden_state, indices, cu_seqlens, max_seqlen_in_batch = unpad_input(hidden_states, attention_mask)
        unpadded_position_ids                                           = position_ids.broadcast_to((BS, T)).flatten()[indices]
    else:
        unpadded_hidden_state = hidden_states.view(-1, self.hidden_size)
        unpadded_position_ids = position_ids.broadcast_to((BS, T)).flatten()

        max_seqlen_in_batch  = T
        cu_seqlens           = torch.arange(0, (BS + 1) * T, step=T, dtype=torch.int32, device=hidden_states.device)

    # qkv proj, shape: [num_heads, total_nnz, head_dim]
    query_states = self.q_proj(unpadded_hidden_state).view(-1, self.num_heads, self.head_dim)
    key_states = self.k_proj(unpadded_hidden_state).view(-1,   self.num_heads, self.head_dim)
    value_states = self.v_proj(unpadded_hidden_state).view(-1, self.num_heads, self.head_dim)

    # RoPE
    cos, sin = self.rotary_emb(value_states, seq_len=max_seqlen_in_batch)
    query_states, key_states = unpadded_apply_rotary_pos_emb(query_states, key_states, cos, sin, unpadded_position_ids)

    # flash attn
    assert not use_cache, "use_cache is not supported."
    assert not output_attentions, "output_attentions is not supported."

    attn_output = flash_attn_unpadded_func(
        q=query_states, k=key_states, v=value_states,
        cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen_in_batch, max_seqlen_k=max_seqlen_in_batch,
        dropout_p=0.0, causal=True)

    # projection
    # attn_output: [num_heads, total_nnz, head_dim]
    attn_output = attn_output.view(-1, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    # pad
    if attention_mask is not None:
        attn_output = pad_input(attn_output, indices, BS, T)
    else:
        attn_output = attn_output.view(BS, T, self.hidden_size)

    return attn_output, None, None


# Disable the transformation of the attention mask in LlamaModel as the flash attention
# requires the attention mask to be the same as the key_padding_mask
def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
    # [bsz, seq_len]
    return attention_mask


def replace_llama_attn():
    cuda_major, cuda_minor = torch.cuda.get_device_capability()
    if cuda_major < 8:
        logging.warning(
            "Flash attention is only supported on A100 or H100 GPU during training due to head dim > 64 backward."
            "ref: https://github.com/HazyResearch/flash-attention/issues/190#issuecomment-1523359593"
        )

    transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask
    transformers.models.llama.modeling_llama.LlamaAttention.forward = forward

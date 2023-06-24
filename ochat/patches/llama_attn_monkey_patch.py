from typing import Optional, Tuple, List, Union

import torch
from einops import rearrange

import transformers
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging

from flash_attn.flash_attn_interface import flash_attn_unpadded_func
from flash_attn.bert_padding import pad_input, unpad_input, index_first_axis


logger = logging.get_logger(__name__)


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


def attn_forward(
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
        unpadded_q   = query_states.view(-1, self.num_heads, self.head_dim)

        max_seqlen_q = T
        cu_seqlens_q = torch.arange(0, (BS + 1) * T, step=T, dtype=torch.int32, device=hidden_states.device)

        # kv
        unpadded_k   = key_states.view(-1, self.num_heads, self.head_dim)
        unpadded_v   = value_states.view(-1, self.num_heads, self.head_dim)

        max_seqlen_k = kv_seq_len
        cu_seqlens_k = torch.arange(0, (BS + 1) * kv_seq_len, step=kv_seq_len, dtype=torch.int32, device=hidden_states.device)

    # flash attn
    assert not output_attentions, "output_attentions is not supported."

    is_causal = T > 1  # T == 1 (decoding) do not need causal mask.
    attn_output = flash_attn_unpadded_func(
        q=unpadded_q, k=unpadded_k, v=unpadded_v,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
        dropout_p=0.0, causal=is_causal)

    # attn_output: [num_heads, total_nnz, head_dim]
    attn_output = attn_output.view(-1, self.hidden_size)

    if attention_mask is not None:
        attn_output = pad_input(attn_output, indices_q, BS, T)
    else:
        attn_output = attn_output.view(BS, T, self.hidden_size)

    return self.o_proj(attn_output), None, past_key_value


def model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

    seq_length_with_past = seq_length
    past_key_values_length = 0

    if past_key_values is not None:
        # past_key_values_length = past_key_values[0][0].shape[2]
        # NOTE: PATCH 1: Use dim 1 for length as in attention.
        past_key_values_length = past_key_values[0][0].shape[1]
        seq_length_with_past = seq_length_with_past + past_key_values_length

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    # embed positions
    # NOTE: PATCH 2: Do not process attention mask.
    # if attention_mask is None:
    #     attention_mask = torch.ones(
    #         (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
    #     )
    # attention_mask = self._prepare_decoder_attention_mask(
    #     attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
    # )

    hidden_states = inputs_embeds

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        if self.gradient_checkpointing and self.training:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs, output_attentions, None)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(decoder_layer),
                hidden_states,
                attention_mask,
                position_ids,
                None,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


# NOTE: Support cache but slower, for inference usage.
def replace_llama_attn():
    cuda_major, cuda_minor = torch.cuda.get_device_capability()
    if cuda_major < 8:
        logger.warning_once(
            "Flash attention is only supported on A100 or H100 GPU during training due to head dim > 64 backward."
            "ref: https://github.com/HazyResearch/flash-attention/issues/190#issuecomment-1523359593"
        )

    transformers.models.llama.modeling_llama.LlamaModel.forward = model_forward
    transformers.models.llama.modeling_llama.LlamaAttention.forward = attn_forward

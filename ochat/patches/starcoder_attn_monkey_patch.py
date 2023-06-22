from typing import Optional, Tuple, Union, List
import logging

import torch

import transformers
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from flash_attn.flash_attn_interface import flash_attn_unpadded_kvpacked_func
from flash_attn.bert_padding import pad_input, unpad_input


def attn_forward(
    self,
    hidden_states: torch.Tensor,
    layer_past: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
) -> Union[
    Tuple[torch.Tensor, Optional[torch.Tensor]],
    Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[torch.Tensor, ...]],
]:
    BS, T, _ = hidden_states.shape

    # qkv proj: [BS x T x D]
    assert self.multi_query
    query, key_value = self.c_attn(hidden_states).split((self.embed_dim, 2 * self.head_dim), dim=-1)

    # Cache
    if layer_past is not None:
        # reuse k, v, self_attention
        key_value   = torch.cat([layer_past, key_value], dim=1)

    past_key_value = key_value if use_cache else None

    kv_seq_len = key_value.shape[1]

    # unpad
    if attention_mask is not None:
        # q
        unpadded_q, indices_q, cu_seqlens_q,  max_seqlen_q  = unpad_input(query, attention_mask[..., -T:])
        # kv
        unpadded_kv, _,        cu_seqlens_kv, max_seqlen_kv = unpad_input(key_value, attention_mask)
    else:
        # q
        unpadded_q   = query.view(-1, self.hidden_size)

        max_seqlen_q = T
        cu_seqlens_q = torch.arange(0, (BS + 1) * T, step=T, dtype=torch.int32, device=hidden_states.device)
        # kv
        unpadded_kv   = key_value.view(-1, 2 * self.head_dim)

        max_seqlen_kv = kv_seq_len
        cu_seqlens_kv = torch.arange(0, (BS + 1) * kv_seq_len, step=kv_seq_len, dtype=torch.int32, device=hidden_states.device)

    # MQA
    unpadded_q  = unpadded_q.view(-1, self.num_heads, self.head_dim)
    unpadded_kv = unpadded_kv.view(-1, 2, 1, self.head_dim).broadcast_to(-1, 2, self.num_heads, self.head_dim)

    # flash attn
    assert not output_attentions, "output_attentions is not supported."

    attn_output = flash_attn_unpadded_kvpacked_func(
        q=unpadded_q, kv=unpadded_kv,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_kv,
        max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_kv,
        dropout_p=self.attn_dropout.p if self.training else 0.0,
        causal=True
    )

    # attn_output: [num_heads, total_nnz, head_dim]
    attn_output = attn_output.view(-1, self.embed_dim)

    if attention_mask is not None:
        attn_output = pad_input(attn_output, indices_q, BS, T)
    else:
        attn_output = attn_output.view(BS, T, self.hidden_size)

    # final projection
    attn_output = self.c_proj(attn_output)
    attn_output = self.resid_dropout(attn_output)

    return attn_output, past_key_value


# Patched forward for attention mask
def model_forward(
    self,
    input_ids: Optional[torch.Tensor] = None,
    past_key_values: Optional[List[torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    token_type_ids: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
        batch_size = inputs_embeds.shape[0]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    if batch_size <= 0:
        raise ValueError("batch_size has to be defined and > 0")

    device = input_ids.device if input_ids is not None else inputs_embeds.device

    if token_type_ids is not None:
        token_type_ids = token_type_ids.view(-1, input_shape[-1])
    if position_ids is not None:
        position_ids = position_ids.view(-1, input_shape[-1])

    if past_key_values is None:
        past_length = 0
        past_key_values = tuple([None] * len(self.h))
    else:
        past_length = past_key_values[0].size(-2)

    if attention_mask is not None and len(attention_mask.shape) == 2 and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_length > 0:
            position_ids = position_ids[:, past_length : input_shape[-1] + past_length :]
    elif position_ids is None:
        position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

    # # Self-attention mask.
    # query_length = input_shape[-1]
    # key_length = past_length + query_length
    # self_attention_mask = self.bias[None, key_length - query_length : key_length, :key_length]

    # if attention_mask is not None:
    #     self_attention_mask = self_attention_mask * attention_mask.view(batch_size, 1, -1).to(
    #         dtype=torch.bool, device=self_attention_mask.device
    #     )

    # # MQA models: (batch_size, query_length, n_heads, key_length)
    # # MHA models: (batch_size, n_heads, query_length, key_length)
    # attention_mask = self_attention_mask.unsqueeze(2 if self.multi_query else 1)

    # If a 2D or 3D attention mask is provided for the cross-attention
    # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
    if (
        self.config.add_cross_attention
        and encoder_hidden_states is not None
        and encoder_attention_mask is not None
    ):
        if encoder_attention_mask.dim() == 2:
            encoder_attention_mask.unsqueeze(1)
        assert encoder_attention_mask.dim() == 3
        encoder_attention_mask = encoder_attention_mask.bool().unsqueeze(2 if self.multi_query else 1)
    else:
        encoder_attention_mask = None

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape bsz x n_heads x N x N
    # head_mask has shape n_layer x batch x n_heads x N x N
    head_mask = self.get_head_mask(head_mask, self.config.n_layer)

    if inputs_embeds is None:
        inputs_embeds = self.wte(input_ids)
    position_embeds = self.wpe(position_ids)
    hidden_states = inputs_embeds + position_embeds

    if token_type_ids is not None:
        token_type_embeds = self.wte(token_type_ids)
        hidden_states = hidden_states + token_type_embeds

    hidden_states = self.drop(hidden_states)

    output_shape = input_shape + (hidden_states.size(-1),)

    presents = [] if use_cache else None
    all_self_attentions = () if output_attentions else None
    all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
    all_hidden_states = () if output_hidden_states else None
    for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if self.gradient_checkpointing and self.training:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs, use_cache, output_attentions)

                return custom_forward

            outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states,
                None,
                attention_mask,
                head_mask[i],
                encoder_hidden_states,
                encoder_attention_mask,
            )
        else:
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

        hidden_states = outputs[0]
        if use_cache:
            presents.append(outputs[1])

        if output_attentions:
            all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
            if self.config.add_cross_attention:
                all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

    hidden_states = self.ln_f(hidden_states)

    hidden_states = hidden_states.view(output_shape)
    # Add last hidden state
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
            if v is not None
        )

    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=presents,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
        cross_attentions=all_cross_attentions,
    )


def replace_starcoder_attn(attn_train_mode: bool = False):
    cuda_major, cuda_minor = torch.cuda.get_device_capability()
    if cuda_major < 8:
        logging.warning(
            "Flash attention is only supported on A100 or H100 GPU during training due to head dim > 64 backward."
            "ref: https://github.com/HazyResearch/flash-attention/issues/190#issuecomment-1523359593"
        )

    transformers.models.gpt_bigcode.modeling_gpt_bigcode.GPTBigCodeAttention.forward = attn_forward
    transformers.models.gpt_bigcode.modeling_gpt_bigcode.GPTBigCodeModel.forward = model_forward

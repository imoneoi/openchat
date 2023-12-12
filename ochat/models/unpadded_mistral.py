# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch Unpadded & Fused Mistral model. Compatible with HF. """

from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.models.mistral.configuration_mistral import MistralConfig

try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input
except ImportError:
    print ("FlashAttention not found. Install it if you need to train models.")


logger = logging.get_logger(__name__)


@torch.jit.script  # type: ignore
def weighted_token_accuracy(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor):
    return (weights * (torch.argmax(logits, dim=-1) == labels)).sum()


@torch.jit.script  # type: ignore
def weighted_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor):
    return (weights * torch.nn.functional.cross_entropy(logits, labels, reduction="none")).sum()


def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor):
    # q, k:     [nnz, num_heads, head_dim]
    # position_ids: [nnz]
    # cos, sin: [max_seq_len, head_dim]
    cos = cos[position_ids].unsqueeze(-2)  # [nnz, 1, head_dim]
    sin = sin[position_ids].unsqueeze(-2)  # [nnz, 1, head_dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Mistral
RMS_NORM_TRACED = None


def rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor, variance_epsilon: torch.Tensor):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    variance = (hidden_states * hidden_states).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return weight * hidden_states.to(input_dtype)


class UnpaddedMistralRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        """
        UnpaddedMistralRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()

        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = torch.tensor(eps, dtype=torch.get_default_dtype())

        global RMS_NORM_TRACED
        if RMS_NORM_TRACED is None:
            RMS_NORM_TRACED = torch.jit.trace(rms_norm, (torch.ones(hidden_size), torch.ones(hidden_size), self.variance_epsilon))

    def forward(self, hidden_states):
        global RMS_NORM_TRACED
        return RMS_NORM_TRACED(hidden_states, self.weight, self.variance_epsilon)


# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->Mistral
class UnpaddedMistralRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()

        # RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        dtype = torch.get_default_dtype()
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached


class UnpaddedMistralMLP(nn.Module):
    def __init__(self, config: MistralConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class UnpaddedMistralAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: MistralConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.sliding_window = config.sliding_window

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        cos_sin: Tuple[torch.Tensor, torch.Tensor],
        # Unpadded inputs
        nz_hidden_states: torch.Tensor,
        nz_position_ids: torch.LongTensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int
    ) -> torch.Tensor:
        # nz_hidden_states: [nnz, num_heads, head_dim]
        # nz_position_ids:  [nnz]
        # cu_seqlens:       [bs + 1]

        query_states = self.q_proj(nz_hidden_states).view(-1, self.num_heads, self.head_dim)
        key_states = self.k_proj(nz_hidden_states).view(-1,   self.num_key_value_heads, self.head_dim)
        value_states = self.v_proj(nz_hidden_states).view(-1, self.num_key_value_heads, self.head_dim)

        # RoPE
        cos, sin = cos_sin
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, nz_position_ids)

        # flash attn
        attn_output = flash_attn_varlen_func(
            q=query_states, k=key_states, v=value_states,
            cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,

            dropout_p=0.0, causal=True,
            window_size=(self.sliding_window, self.sliding_window))

        # attn_output: [total_nnz, num_heads, head_dim]
        attn_output = attn_output.view(-1, self.hidden_size)  # type: ignore
        return self.o_proj(attn_output)


class UnpaddedMistralDecoderLayer(nn.Module):
    def __init__(self, config: MistralConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.self_attn = UnpaddedMistralAttention(config=config)
        self.mlp = UnpaddedMistralMLP(config=config)
        self.input_layernorm = UnpaddedMistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = UnpaddedMistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        cos_sin: Tuple[torch.Tensor, torch.Tensor],
        # Unpadded inputs
        nz_hidden_states: torch.Tensor,
        nz_position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int
    ) -> torch.Tensor:
        # Self Attention
        residual = nz_hidden_states

        nz_hidden_states = self.input_layernorm(nz_hidden_states)
        nz_hidden_states = self.self_attn(
            cos_sin=cos_sin,

            nz_hidden_states=nz_hidden_states,
            nz_position_ids=nz_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen
        )
        nz_hidden_states = residual + nz_hidden_states

        # Fully Connected
        residual = nz_hidden_states

        nz_hidden_states = self.post_attention_layernorm(nz_hidden_states)
        nz_hidden_states = self.mlp(nz_hidden_states)
        nz_hidden_states = residual + nz_hidden_states

        return nz_hidden_states


class UnpaddedMistralPreTrainedModel(PreTrainedModel):
    config_class = MistralConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["UnpaddedMistralDecoderLayer"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class UnpaddedMistralModel(UnpaddedMistralPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`UnpaddedMistralDecoderLayer`]

    Args:
        config: MistralConfig
    """

    def __init__(self, config: MistralConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.rotary_emb   = UnpaddedMistralRotaryEmbedding(config.hidden_size // config.num_attention_heads,
                                                         max_position_embeddings=config.max_position_embeddings,
                                                         base=config.rope_theta)

        self.layers = nn.ModuleList([UnpaddedMistralDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = UnpaddedMistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        # Unpadded inputs
        nz_input_ids: torch.Tensor,
        nz_position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        nz_hidden_states = self.embed_tokens(nz_input_ids)
        cos_sin          = self.rotary_emb()

        # decoder layers
        for decoder_layer in self.layers:
            if self.gradient_checkpointing and self.training:
                nz_hidden_states = self._gradient_checkpointing_func(
                    decoder_layer.__call__,

                    cos_sin,
                    nz_hidden_states,
                    nz_position_ids,
                    cu_seqlens,
                    max_seqlen
                )
            else:
                nz_hidden_states = decoder_layer(
                    cos_sin,
                    
                    nz_hidden_states,
                    nz_position_ids,
                    cu_seqlens,
                    max_seqlen
                )

        nz_hidden_states = self.norm(nz_hidden_states)

        return nz_hidden_states


class MistralForCausalLM(UnpaddedMistralPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = UnpaddedMistralModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model
    
    def forward(
        self,
        # Unpadded inputs
        nz_input_ids: torch.Tensor,
        nz_position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        # Unpadded labels
        nz_shifted_label_ids: Optional[torch.Tensor] = None,
        nz_shifted_loss_weights:      Optional[torch.Tensor] = None
    ) -> CausalLMOutputWithPast:
        # Model logits
        hidden_states = self.model(
            nz_input_ids=nz_input_ids,
            nz_position_ids=nz_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen
        )
        logits = self.lm_head(hidden_states)

        loss = None
        if nz_shifted_label_ids is not None:
            assert nz_shifted_loss_weights is not None

            loss = weighted_cross_entropy(logits, nz_shifted_label_ids, nz_shifted_loss_weights), \
                   weighted_token_accuracy(logits.detach(), nz_shifted_label_ids, nz_shifted_loss_weights)

        return CausalLMOutputWithPast(
            loss=loss,  # type: ignore
            logits=logits
        )


class PaddedMistralForCausalLM(MistralForCausalLM):
    """Compat layer for padded inputs"""

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        # unused
        return_dict: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False
    ):
        batch_size, seq_len = input_ids.shape
        if position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)

        # get indices
        seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
        indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        max_seqlen_in_batch = int(seqlens_in_batch.max().item())
        cu_seqlens = torch.nn.functional.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))

        # Unpad inputs
        nz_input_ids    = torch.take_along_dim(input_ids,    indices)
        nz_position_ids = torch.take_along_dim(position_ids, indices)

        # Unpadded forward
        logits = super().forward(
            nz_input_ids=nz_input_ids,
            nz_position_ids=nz_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen_in_batch
        ).logits

        # Pad logits
        logits = pad_input(logits, indices, batch_size, seq_len)

        return CausalLMOutputWithPast(logits=logits)  # type: ignore

    def prepare_inputs_for_generation(self,
                                      input_ids: torch.Tensor,
                                      **kwargs):
        return {
            "input_ids": input_ids,
            "attention_mask": kwargs.get("attention_mask"),
            "position_ids": kwargs.get("position_ids")
        }

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
""" PyTorch Unpadded & Fused Gemma model. Compatible with HF. """

from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import nn

from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.models.gemma.configuration_gemma import GemmaConfig

try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input
except ImportError:
    print ("FlashAttention not found. Install it if you need to train models.")


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


@torch.jit.script
def lm_head_with_loss(embed_weights: torch.Tensor, hidden_states: torch.Tensor, nz_shifted_label_ids: torch.Tensor, nz_shifted_loss_weights: torch.Tensor):
    logits = F.linear(hidden_states, embed_weights)

    loss = (nz_shifted_loss_weights * torch.nn.functional.cross_entropy(logits, nz_shifted_label_ids, reduction="none")).sum()
    token_accuracy = (nz_shifted_loss_weights * (torch.argmax(logits.detach(), dim=-1) == nz_shifted_label_ids)).sum()
    return loss, token_accuracy


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Gemma
RMS_NORM_TRACED = None


def rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor, variance_epsilon: torch.Tensor):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return (1 + weight) * hidden_states.to(input_dtype)


class UnpaddedGemmaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        """
        UnpaddedGemmaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = torch.tensor(eps, dtype=torch.get_default_dtype())

        global RMS_NORM_TRACED
        if RMS_NORM_TRACED is None:
            RMS_NORM_TRACED = torch.jit.trace(rms_norm, (torch.ones(hidden_size), torch.ones(hidden_size), self.variance_epsilon))

    def forward(self, hidden_states):
        global RMS_NORM_TRACED
        return RMS_NORM_TRACED(hidden_states, self.weight, self.variance_epsilon)


# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->Gemma
class UnpaddedGemmaRotaryEmbedding(torch.nn.Module):
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


class UnpaddedGemmaMLP(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class UnpaddedGemmaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: GemmaConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

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

            dropout_p=0.0, causal=True)

        # attn_output: [total_nnz, num_heads, head_dim]
        attn_output = attn_output.view(-1, self.num_heads * self.head_dim)  # type: ignore
        return self.o_proj(attn_output)


class UnpaddedGemmaDecoderLayer(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.self_attn = UnpaddedGemmaAttention(config=config)
        self.mlp = UnpaddedGemmaMLP(config=config)
        self.input_layernorm = UnpaddedGemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = UnpaddedGemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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


class UnpaddedGemmaPreTrainedModel(PreTrainedModel):
    config_class = GemmaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["UnpaddedGemmaDecoderLayer"]

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


class UnpaddedGemmaModel(UnpaddedGemmaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`UnpaddedGemmaDecoderLayer`]

    Args:
        config: GemmaConfig
    """

    def __init__(self, config: GemmaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.normalization_factor = config.hidden_size ** 0.5

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.rotary_emb   = UnpaddedGemmaRotaryEmbedding(config.head_dim,
                                                         max_position_embeddings=config.max_position_embeddings,
                                                         base=config.rope_theta)

        self.layers = nn.ModuleList([UnpaddedGemmaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = UnpaddedGemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
        nz_hidden_states = self.embed_tokens(nz_input_ids) * self.normalization_factor  # Normalized
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


class GemmaForCausalLM(UnpaddedGemmaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = UnpaddedGemmaModel(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.model.embed_tokens

    def set_output_embeddings(self, new_embeddings):
        self.model.embed_tokens = new_embeddings

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

        # Loss
        loss = lm_head_with_loss(
            self.model.embed_tokens.weight,  # Tied embeddings
            hidden_states,
            nz_shifted_label_ids,
            nz_shifted_loss_weights
        )

        return CausalLMOutputWithPast(
            loss=loss  # type: ignore
        )

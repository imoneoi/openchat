from functools import partial

import torch
import transformers

import ochat.models
from ochat.config.conversation_template import Conversation, ConversationTemplate, Message
from ochat.config.model_config import ModelConfig

_V3_2_PREFIXES = {
    # OpenAI mapping

    "user": "User:",
    "assistant": "Assistant:"
}


_GEMMA_IT_PREFIXES = {
    "user": "user",
    "assistant": "model"
}


def _v3_2_role_prefix(from_role, condition):
    return f"{condition} {_V3_2_PREFIXES[from_role]}".strip()


MODEL_CONFIG_MAP = {
    # OpenChat V3.6 (MoE)
    "openchat_3.6": ModelConfig(
        # Model
        model_max_context=8192,
        model_tokenizer_create=partial(transformers.AutoTokenizer.from_pretrained, use_fast=False),
        model_create_for_training=lambda: None,  # NOTE(one): MoE trainer decoupled from the codebase

        # Conversation Template
        conversation_template=partial(ConversationTemplate,
                                      role_prefix=_v3_2_role_prefix,
                                      eot="</s>",
                                      inference_condition="GPT4 Correct")
    ),

    # OpenChat V3.2
    "openchat_v3.2": ModelConfig(
        # Model
        model_max_context=4096,
        model_tokenizer_create=partial(transformers.AutoTokenizer.from_pretrained, use_fast=False),
        model_create_for_training=partial(ochat.models.LlamaForCausalLM.from_pretrained,
                                          low_cpu_mem_usage=True,
                                          torch_dtype=torch.bfloat16),

        # Conversation Template
        conversation_template=partial(ConversationTemplate,
                                      role_prefix=_v3_2_role_prefix,
                                      eot="<|end_of_turn|>",
                                      inference_condition="GPT4")
    ),

    "openchat_v3.2_mistral": ModelConfig(
        serving_aliases=("openchat_3.5", ),

        # Model
        model_max_context=8192,
        model_tokenizer_create=partial(transformers.AutoTokenizer.from_pretrained, use_fast=False),
        model_create_for_training=partial(ochat.models.MistralForCausalLM.from_pretrained,
                                          low_cpu_mem_usage=True,
                                          torch_dtype=torch.bfloat16),

        # Conversation Template
        conversation_template=partial(ConversationTemplate,
                                      role_prefix=_v3_2_role_prefix,
                                      eot="<|end_of_turn|>",
                                      inference_condition="GPT4 Correct")
    ),

    "openchat_v3.2_gemma_new": ModelConfig(
        serving_aliases=("openchat_3.5_gemma_new", ),

        # Model
        model_max_context=8192,
        model_tokenizer_create=partial(transformers.AutoTokenizer.from_pretrained, use_fast=False),
        model_create_for_training=partial(ochat.models.GemmaForCausalLM.from_pretrained,
                                          low_cpu_mem_usage=True,
                                          torch_dtype=torch.bfloat16),

        # Conversation Template
        conversation_template=partial(ConversationTemplate,
                                      role_prefix=_v3_2_role_prefix,
                                      eot="<end_of_turn>",
                                      inference_condition="GPT4 Correct")
    ),

    ### Other models
    "chatml_mistral": ModelConfig(
        # Model
        model_max_context=8192,
        model_tokenizer_create=partial(transformers.AutoTokenizer.from_pretrained, use_fast=False),
        model_create_for_training=partial(ochat.models.MistralForCausalLM.from_pretrained,
                                          low_cpu_mem_usage=True,
                                          torch_dtype=torch.bfloat16),

        # Conversation Template
        conversation_template=partial(ConversationTemplate,
                                      role_prefix=lambda from_role, condition: f"<|im_start|>{from_role}\n",
                                      eot="<|im_end|>",
                                      inference_condition="")
    ),
    "zephyr_mistral": ModelConfig(
        # Model
        model_max_context=8192,
        model_tokenizer_create=partial(transformers.AutoTokenizer.from_pretrained, use_fast=False),
        model_create_for_training=partial(ochat.models.MistralForCausalLM.from_pretrained,
                                          low_cpu_mem_usage=True,
                                          torch_dtype=torch.bfloat16),

        # Conversation Template
        conversation_template=partial(ConversationTemplate,
                                      role_prefix=lambda from_role, condition: f"<|{from_role}|>\n",
                                      eot="</s>",
                                      inference_condition="")
    ),
    "gemma_it": ModelConfig(
        # Model
        model_max_context=8192,
        model_tokenizer_create=partial(transformers.AutoTokenizer.from_pretrained, use_fast=False),
        model_create_for_training=partial(ochat.models.GemmaForCausalLM.from_pretrained,
                                          low_cpu_mem_usage=True,
                                          torch_dtype=torch.bfloat16),

        # Conversation Template
        conversation_template=partial(ConversationTemplate,
                                      role_prefix=lambda from_role, condition: f"<start_of_turn>{_GEMMA_IT_PREFIXES[from_role]}\n",
                                      eot="<end_of_turn>",
                                      inference_condition="")
    ),
}

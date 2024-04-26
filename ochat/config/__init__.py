from functools import partial

import torch
import transformers

from ochat.config.model_config import ModelConfig
from ochat.config.conversation_template import Message, Conversation, ConversationTemplate
import ochat.models


_V3_2_PREFIXES = {
    # OpenAI mapping

    "user": "User:",
    "assistant": "Assistant:"
}

_V3_6_PREFIXES = {
    "user": "User",
    "assistant": "Assistant",
    "system": "System"
}


_GEMMA_IT_PREFIXES = {
    "user": "user",
    "assistant": "model"
}


def _v3_2_role_prefix(from_role, condition):
    return f"{condition} {_V3_2_PREFIXES[from_role]}".strip()

def _v3_6_role_prefix(from_role, condition, role_start_token="", role_end_token=""):
    return role_start_token + f"{condition} {_V3_6_PREFIXES[from_role]}".strip() + role_end_token

MODEL_CONFIG_MAP = {
    # OpenChat V3.6 (llama 3)
    "openchat_3.6": ModelConfig(
        # Model
        model_max_context=8192,
        model_tokenizer_create=partial(transformers.AutoTokenizer.from_pretrained, use_fast=True),  # Llama 3 only has fast tokenizer
        model_create_for_training=partial(ochat.models.LlamaForCausalLM.from_pretrained,
                                          low_cpu_mem_usage=True,
                                          torch_dtype=torch.bfloat16),
        # Conversation Template
        conversation_template=partial(ConversationTemplate,
                                      role_prefix=partial(_v3_6_role_prefix,
                                                          role_start_token="<|start_header_id|>",
                                                          role_end_token="<|end_header_id|>"),
                                      bos="<|begin_of_text|>",  # Llama 3 tokenizer needs manually specifing tokenizer
                                      eot="<|eot_id|>",
                                      system_as_role=True,
                                      strip_message=True,
                                      inference_condition="GPT4 Correct",
                                      message_prefix="\n\n"),
        hf_chat_template="{% set loop_messages = messages %}{% for message in loop_messages %}{% if message['role'] in ['user', 'assistant'] %}{% set content = '<|start_header_id|>GPT4 Correct ' + message['role'].title() + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}{% elif message['role'] == 'system' %}{% set content = '<|start_header_id|>System<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}{% endif %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>GPT4 Correct Assistant<|end_header_id|>\n\n' }}{% endif %}",
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
                                      inference_condition="GPT4 Correct"),
        hf_chat_template="{{ bos_token }}{% for message in messages %}{{ 'GPT4 Correct ' + message['role'].title() + ': ' + message['content'] + '<|end_of_turn|>'}}{% endfor %}{% if add_generation_prompt %}{{ 'GPT4 Correct Assistant:' }}{% endif %}"
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
                                      inference_condition="GPT4 Correct"),
        hf_chat_template="{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"
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

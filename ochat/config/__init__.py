from functools import partial

import torch
import transformers

from ochat.config.model_config import ModelConfig
from ochat.config.conversation_template import Message, Conversation, ConversationTemplate
import ochat.models


_GEMMA_IT_PREFIXES = {
    "user": "user",
    "assistant": "model"
}


def _v3_2_role_prefix(from_role, condition):
    return f"{condition} {from_role.title()}:".strip()


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
                                      role_prefix=_v3_2_role_prefix,
                                      add_space_before_msg=True,  # Llama 3 tokenizer needs manually adding space
                                      eot="<|eot_id|>",
                                      inference_condition="GPT4 Correct"),
        hf_chat_template="{{ bos_token }}{% for message in messages %}{% if message['role'] in ['user', 'assistant'] %}{% set content = 'GPT4 Correct ' + message['role'].title() + ': ' + message['content'] + '<|eot_id|>' %}{% elif message['role'] == 'system' %}{% set content = message['content'] + '<|eot_id|>' %}{% else %}{{ raise_exception('Only user, assistant and system roles are supported!') }}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ 'GPT4 Correct Assistant:' }}{% endif %}",
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
        model_tokenizer_create=partial(transformers.AutoTokenizer.from_pretrained, use_fast=True),
        model_create_for_training=partial(ochat.models.MistralForCausalLM.from_pretrained,
                                          low_cpu_mem_usage=True,
                                          torch_dtype=torch.bfloat16),

        # Conversation Template
        conversation_template=partial(ConversationTemplate,
                                      role_prefix=_v3_2_role_prefix,
                                      eot="<|end_of_turn|>",
                                      inference_condition="GPT4 Correct"),
        hf_chat_template="{{ bos_token }}{% for message in messages %}{% if message['role'] in ['user', 'assistant'] %}{% set content = 'GPT4 Correct ' + message['role'].title() + ': ' + message['content'] + '<|end_of_turn|>' %}{% elif message['role'] == 'system' %}{% set content = message['content'] + '<|end_of_turn|>' %}{% else %}{{ raise_exception('Only user, assistant and system roles are supported!') }}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ 'GPT4 Correct Assistant:' }}{% endif %}",    ),

    "openchat_v3.2_gemma_new": ModelConfig(
        serving_aliases=("openchat_3.5_gemma_new", ),

        # Model
        model_max_context=8192,
        model_tokenizer_create=partial(transformers.AutoTokenizer.from_pretrained, use_fast=True),
        model_create_for_training=partial(ochat.models.GemmaForCausalLM.from_pretrained,
                                          low_cpu_mem_usage=True,
                                          torch_dtype=torch.bfloat16),

        # Conversation Template
        conversation_template=partial(ConversationTemplate,
                                      role_prefix=_v3_2_role_prefix,
                                      eot="<end_of_turn>",
                                      inference_condition="GPT4 Correct"),
        hf_chat_template="{{ bos_token }}{% for message in messages %}{% if message['role'] in ['user', 'assistant'] %}{% set content = 'GPT4 Correct ' + message['role'].title() + ': ' + message['content'] + '<end_of_turn>' %}{% elif message['role'] == 'system' %}{% set content = message['content'] + '<end_of_turn>' %}{% else %}{{ raise_exception('Only user, assistant and system roles are supported!') }}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ 'GPT4 Correct Assistant:' }}{% endif %}",
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
    "llama3_instruct": ModelConfig(
        # Model
        model_max_context=8192,
        model_tokenizer_create=partial(transformers.AutoTokenizer.from_pretrained, use_fast=True),  # Llama 3 only has fast tokenizer
        model_create_for_training=partial(ochat.models.LlamaForCausalLM.from_pretrained,
                                          low_cpu_mem_usage=True,
                                          torch_dtype=torch.bfloat16),

        # Conversation Template
        conversation_template=partial(ConversationTemplate,
                                      role_prefix=lambda from_role, condition: f"<|start_header_id|>{from_role}<|end_header_id|>\n\n",
                                      eot="<|eot_id|>",
                                      inference_condition="")
    ),
}

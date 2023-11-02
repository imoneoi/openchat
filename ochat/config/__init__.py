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


def _v3_2_role_prefix(from_role, condition):
    return f"{condition} {_V3_2_PREFIXES[from_role]}".strip()


MODEL_CONFIG_MAP = {
    # OpenChat V3.2
    "openchat_v3.2": ModelConfig(
        # Model
        model_max_context=4096,
        model_tokenizer_create=partial(transformers.AutoTokenizer.from_pretrained,
                                       use_fast=False,
                                       legacy=False),
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
        model_tokenizer_create=partial(transformers.AutoTokenizer.from_pretrained,
                                       use_fast=False,
                                       legacy=True),  # Mistral use legacy=True https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/tokenizer_config.json
        model_create_for_training=partial(ochat.models.MistralForCausalLM.from_pretrained,
                                          low_cpu_mem_usage=True,
                                          torch_dtype=torch.bfloat16),

        # Conversation Template
        conversation_template=partial(ConversationTemplate,
                                      role_prefix=_v3_2_role_prefix,
                                      eot="<|end_of_turn|>",
                                      inference_condition="GPT4 Correct")
    ),
}

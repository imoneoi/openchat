from typing import Optional, Callable, Union
from dataclasses import dataclass
from functools import partial

import torch
import transformers
import ochat.models


@dataclass
class ModelConfig:
    name: str

    # Prompt
    role_prefix: Union[dict, Callable]
    ai_role: str
    eot_token: str
    bos_token: Optional[str] = None

    condition_fn: Optional[Callable] = None

    # Label
    group_fn: Optional[Callable] = None
    num_groups: int = 1

    # Model
    model_max_context: Optional[int] = None
    model_create: Optional[Callable] = None
    model_tokenizer_create: Optional[Callable] = None

    # Get template
    def generate_conversation_template(self, tokenize_fn, tokenize_special_fn, system_prompt, message_list, message_props=None):
        tokens = []
        masks = []

        # begin of sentence (bos)
        if self.bos_token:
            t = tokenize_special_fn(self.bos_token)
            tokens.append(t)
            masks.append(False)

        # Condition
        if self.condition_fn is not None:
            t = tokenize_fn(self.condition_fn(message_props)) + [tokenize_special_fn(self.eot_token)]
            tokens.extend(t)
            masks.extend([False] * len(t))

        # System
        if system_prompt:
            t = tokenize_fn(system_prompt) + [tokenize_special_fn(self.eot_token)]
            tokens.extend(t)
            masks.extend([False] * len(t))

        # Messages
        for idx, message in enumerate(message_list):
            # Prefix
            if callable(self.role_prefix):
                role_prefix = self.role_prefix(message["from"], message_props)
            else:
                role_prefix = self.role_prefix[message["from"]]

            t = tokenize_fn(role_prefix)
            tokens.extend(t)
            masks.extend([False] * len(t))

            # Message
            if "value" in message:
                t = tokenize_fn(message["value"]) + [tokenize_special_fn(self.eot_token)]
                tokens.extend(t)
                masks.extend([message["from"] == self.ai_role] * len(t))
            else:
                assert idx == len(message_list) - 1, "Empty message for completion must be on the last."

        group = 0
        if self.group_fn:
            group = self.group_fn(message_props)

        return tokens, masks, group


def _v2_conditional_prefix(from_role, props):
    human_prefix = "User:"
    gpt4_prefix  = "Assistant GPT4:"
    other_prefix = "Assistant GPT3:"

    if from_role == "human":
        return human_prefix
    
    if from_role == "gpt":
        if props is None:
            return gpt4_prefix  # inference using gpt-4 prefix
        
        return gpt4_prefix if props["is_gpt4"] else other_prefix
    
    raise NotImplementedError(f"Unknown role {from_role}")


def _v3_2_conditional_prefix(from_role, props):
    gpt3_prefixes = {
        "human": "GPT3 User:",
        "gpt": "GPT3 Assistant:"
    }
    gpt4_prefixes = {
        "human": "GPT4 User:",
        "gpt": "GPT4 Assistant:"
    }
    prefixes = gpt4_prefixes if props is None or props["is_gpt4"] else gpt3_prefixes

    return prefixes[from_role]


def _v2_v3_group(props):
    if props is None:
        return 1

    return 1 if props["is_gpt4"] else 0


def _v3_condition(props):
    gpt4_condition = "Assistant is GPT4"
    gpt3_condition = "Assistant is GPT3"

    if props is None:
        return gpt4_condition

    return gpt4_condition if props["is_gpt4"] else gpt3_condition


MODEL_CONFIG_MAP = {
    # OpenChat V3.2
    "openchat_v3.2": ModelConfig(
        name="OpenChat V3.2",

        # Prompt
        role_prefix=_v3_2_conditional_prefix,
        ai_role="gpt",
        eot_token="<|end_of_turn|>",
        bos_token="<s>",

        # Label
        group_fn=_v2_v3_group,
        num_groups=2,

        # Tokenize
        model_max_context=4096,
        model_create=partial(ochat.models.LlamaForCausalLM.from_pretrained,
                             low_cpu_mem_usage=True,
                             torch_dtype=torch.bfloat16),
        model_tokenizer_create=partial(transformers.AutoTokenizer.from_pretrained,
                                       use_fast=False,
                                       use_auth_token=True),
    ),

    "openchat_v3.1_llama2": ModelConfig(
        name="OpenChat V3.1 Llama 2",

        # Prompt
        role_prefix={
            "human": "User:",
            "gpt": "Assistant:"
        },
        ai_role="gpt",
        eot_token="<|end_of_turn|>",
        bos_token="<s>",

        condition_fn=_v3_condition,

        # Label
        group_fn=_v2_v3_group,
        num_groups=2,

        # Tokenize
        model_max_context=4096,
        model_create=partial(ochat.models.LlamaForCausalLM.from_pretrained,
                             low_cpu_mem_usage=True,
                             torch_dtype=torch.bfloat16),
        model_tokenizer_create=partial(transformers.AutoTokenizer.from_pretrained,
                                       use_fast=False,
                                       use_auth_token=True),
    ),

    # OpenChat V2
    "openchat_v2": ModelConfig(
        name="OpenChat_v2",

        # Prompt
        role_prefix=_v2_conditional_prefix,
        ai_role="gpt",
        eot_token="<|end_of_turn|>",
        bos_token="<s>",

        # Label
        group_fn=_v2_v3_group,
        num_groups=2,

        # Tokenize
        model_max_context=2048,
        model_create=partial(ochat.models.LlamaForCausalLM.from_pretrained,
                             low_cpu_mem_usage=True,
                             torch_dtype=torch.bfloat16),
        model_tokenizer_create=partial(transformers.AutoTokenizer.from_pretrained,
                                       use_fast=False,
                                       use_auth_token=True),
    ),

    # OpenChat
    "openchat_llama2": ModelConfig(
        name="OpenChat Llama 2",

        # Prompt
        role_prefix={
            "human": "User:",
            "gpt": "Assistant:"
        },
        ai_role="gpt",
        eot_token="<|end_of_turn|>",
        bos_token="<s>",

        # Tokenize
        model_max_context=4096,
        model_create=partial(ochat.models.LlamaForCausalLM.from_pretrained,
                             low_cpu_mem_usage=True,
                             torch_dtype=torch.bfloat16),
        model_tokenizer_create=partial(transformers.AutoTokenizer.from_pretrained,
                                       use_fast=False,
                                       use_auth_token=True),
    ),

    "openchat": ModelConfig(
        name="OpenChat",

        # Prompt
        role_prefix={
            "human": "Human: ",
            "gpt": "Assistant: "
        },
        ai_role="gpt",
        eot_token="<|end_of_turn|>",
        bos_token="<s>",

        # Tokenize
        model_max_context=2048,
        model_create=partial(ochat.models.LlamaForCausalLM.from_pretrained,
                             low_cpu_mem_usage=True,
                             torch_dtype=torch.bfloat16),
        model_tokenizer_create=partial(transformers.AutoTokenizer.from_pretrained,
                                       use_fast=False,
                                       use_auth_token=True),
    ),

    # OpenChat 8192
    "openchat_8192": ModelConfig(
        name="OpenChat_8192",

        # Prompt
        role_prefix={
            "human": "Human: ",
            "gpt": "Assistant: "
        },
        ai_role="gpt",
        eot_token="<|end_of_turn|>",
        bos_token="<s>",

        # Model
        model_max_context=8192,
        model_create=partial(ochat.models.LlamaForCausalLM.from_pretrained,
                             extend_context_to=8192,
                             low_cpu_mem_usage=True,
                             torch_dtype=torch.bfloat16),
        model_tokenizer_create=partial(transformers.AutoTokenizer.from_pretrained,
                                       use_fast=False,
                                       use_auth_token=True),
    ),

    # OpenCoder / OpenCoderPlus
    "opencoder": ModelConfig(
        name="OpenCoder",

        # Prompt
        role_prefix={
            "human": "User:",
            "gpt": "Assistant:"
        },
        ai_role="gpt",
        eot_token="<|end_of_turn|>",
        bos_token=None,

        # Tokenize
        model_max_context=8192,
        model_create=None,  # TODO: StarCoder Unpadded
        model_tokenizer_create=partial(transformers.AutoTokenizer.from_pretrained,
                                       use_fast=False,
                                       use_auth_token=True)
    )
}

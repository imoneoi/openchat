from typing import Optional, Callable
from dataclasses import dataclass
from functools import partial

import torch
import transformers
import ochat.models


@dataclass
class ModelConfig:
    name: str

    # Prompt
    system: Optional[str]

    role_prefix: dict
    ai_role: str
    eot_token: str
    bos_token: Optional[str] = None

    # Model
    model_max_context: Optional[int] = None
    model_create: Optional[Callable] = None
    model_tokenizer_create: Optional[Callable] = None

    # Get template
    def generate_conversation_template(self, tokenize_fn, tokenize_special_fn, message_list):
        tokens = []
        masks = []

        # begin of sentence (bos)
        if self.bos_token:
            t = tokenize_special_fn(self.bos_token)
            tokens.append(t)
            masks.append(False)

        # System
        if self.system:
            t = tokenize_fn(self.system) + [tokenize_special_fn(self.eot_token)]
            tokens.extend(t)
            masks.extend([False] * len(t))

        # Messages
        for idx, message in enumerate(message_list):
            # Prefix
            t = tokenize_fn(self.role_prefix[message["from"]])
            tokens.extend(t)
            masks.extend([False] * len(t))

            # Message
            if "value" in message:
                t = tokenize_fn(message["value"]) + [tokenize_special_fn(self.eot_token)]
                tokens.extend(t)
                masks.extend([message["from"] == self.ai_role] * len(t))
            else:
                assert idx == len(message_list) - 1, "Empty message for completion must be on the last."

        return tokens, masks


MODEL_CONFIG_MAP = {
    # OpenChat 8192
    "openchat_8192": ModelConfig(
        name="OpenChat_8192",

        # Prompt
        system=None,

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
                                       use_fast=False),
    ),

    # OpenChat
    "openchat": ModelConfig(
        name="OpenChat",

        # Prompt
        system=None,

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
                                       use_fast=False),
    ),

    # OpenCoder / OpenCoderPlus
    "opencoder": ModelConfig(
        name="OpenCoder",

        # Prompt
        system=None,

        role_prefix={
            "human": "User:",
            "gpt": "Assistant:"
        },
        ai_role="gpt",
        eot_token="<|end_of_turn|>",
        bos_token=None,

        # Tokenize
        model_max_context=8192,
        model_create=partial(ochat.models.GPTBigCodeForCausalLM.from_pretrained,
                             low_cpu_mem_usage=True,
                             torch_dtype=torch.bfloat16),
        model_tokenizer_create=partial(transformers.AutoTokenizer.from_pretrained,
                                       use_fast=False)
    )
}

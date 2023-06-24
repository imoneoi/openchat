from typing import Optional
from dataclasses import dataclass


@dataclass
class ModelConfig:
    name: str

    # Prompt
    system: Optional[str]

    role_prefix: dict
    ai_role: str
    eot_token: str
    bos_token: Optional[str] = None

    # Tokenize
    max_tokens: Optional[int] = None

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
    # OpenChat
    "openchat": ModelConfig(
        name="OpenChat",

        # Prompt
        system="A chat between a curious user and an AI assistant. The AI assistant works step by step to ensure correct and helpful answers are provided to the user, and responds using Markdown.",

        role_prefix={
            "human": "User:",
            "gpt": "Assistant:"
        },
        ai_role="gpt",
        eot_token="<|end_of_turn|>",
        bos_token="<s>",

        # Tokenize
        max_tokens=2048
    ),

    # OpenChat (extended 8192 context)
    "openchat_8192": ModelConfig(
        name="OpenChat-8192",

        # Prompt
        system="A chat between a curious user and an AI assistant. The AI assistant works step by step to ensure correct and helpful answers are provided to the user, and responds using Markdown.",

        role_prefix={
            "human": "User:",
            "gpt": "Assistant:"
        },
        ai_role="gpt",
        eot_token="<|end_of_turn|>",
        bos_token="<s>",

        # Tokenize
        max_tokens=8192
    ),

    # OpenCoder / OpenCoderPlus
    "opencoder": ModelConfig(
        name="OpenCoder",

        # Prompt
        system="A chat between a curious user and an AI assistant. The AI assistant works step by step to ensure correct and helpful answers are provided to the user, and responds using Markdown.",

        role_prefix={
            "human": "User:",
            "gpt": "Assistant:"
        },
        ai_role="gpt",
        eot_token="<|end_of_turn|>",
        bos_token=None,

        # Tokenize
        max_tokens=8192
    )
}

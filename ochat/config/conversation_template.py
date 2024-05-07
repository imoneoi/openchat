from typing import Optional, Callable, Iterable, List

from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str

    weight: Optional[float] = None


class Conversation(BaseModel):
    items: List[Message]

    condition: str = ""
    system: str = ""


class ConversationTemplate(BaseModel):
    tokenizer: Callable

    # Prompt
    role_prefix: Callable
    eot: str

    inference_condition: Optional[str] = None
    add_space_before_msg: bool = False

    # Private
    bos_tokens_: List[int]
    eot_tokens_: List[int]

    def __init__(self, **data):
        tokenizer = data["tokenizer"]
        eot = data["eot"]
        bos_tokens_ = tokenizer("").input_ids
        eot_tokens_ = tokenizer(eot, add_special_tokens=False).input_ids
        super().__init__(**data, bos_tokens_=bos_tokens_, eot_tokens_=eot_tokens_)

    def _tokenize(self, strings: Iterable[str], ignore_special: bool = True) -> List[List[int]]:
        if self.tokenizer.is_fast:
            # Support for fast tokenizer
            # https://github.com/huggingface/tokenizers/pull/1419
            self.tokenizer._tokenizer.encode_special_tokens = ignore_special
            result = self.tokenizer(strings, return_attention_mask=False, add_special_tokens=False).input_ids
            self.tokenizer._tokenizer.encode_special_tokens = False
        else:
            result = self.tokenizer(strings, split_special_tokens=ignore_special, return_attention_mask=False, add_special_tokens=False).input_ids

        return result

    def tokenize_conversations(self, conversations: Iterable[Conversation], inference: bool = False, seq_level_weight: bool = False):
        # Pre-tokenize all conversations
        default_condition = self.inference_condition if inference else ""
        message_prefix = " " if self.add_space_before_msg else ""

        sys_mappings = set()
        role_mappings = set()
        all_text = []
        for conv in conversations:
            sys_mappings.add(conv.system)

            last_idx = len(conv.items) - 1
            for idx, msg in enumerate(conv.items):
                role_mappings.add((msg.role, conv.condition or default_condition))

                if inference and idx == last_idx and not msg.content:
                    all_text.append("")  # Do not prepend any prefix to the beginning of generation
                else:
                    all_text.append(message_prefix + msg.content)

        sys_mappings = list(sys_mappings)
        role_mappings = list(role_mappings)

        sys_mappings = dict(zip(sys_mappings, self._tokenize(sys_mappings)))
        role_mappings = dict(zip(role_mappings, self._tokenize([self.role_prefix(*args) for args in role_mappings], ignore_special=False)))
        all_text = self._tokenize(all_text)

        # Convert
        result_tokens = []
        result_weights = []
        all_text_idx = 0
        for conv in conversations:
            tokens = []
            weights = []

            # bos tokens
            tokens.extend(self.bos_tokens_)
            weights.extend([0.] * len(self.bos_tokens_))

            # System
            if conv.system:
                system = sys_mappings[conv.system]
                tokens.extend(system)
                weights.extend([0.] * len(system))

                tokens.extend(self.eot_tokens_)
                weights.extend([0.] * len(self.eot_tokens_))

            # Messages
            last_idx = len(conv.items) - 1
            for idx, msg in enumerate(conv.items):
                # Role Prefix
                role = role_mappings[(msg.role, conv.condition or default_condition)]
                tokens.extend(role)
                weights.extend([0.] * len(role))

                # Message
                text = all_text[all_text_idx]
                all_text_idx += 1

                # weight
                w = None
                if not inference:
                    assert msg.weight is not None

                    w = msg.weight
                    if seq_level_weight:
                        w /= len(text) + len(self.eot_tokens_)

                # Message tokens
                tokens.extend(text)
                weights.extend([w] * len(text))

                if not (inference and idx == last_idx):  # Do not add EOT on last turn during inference
                    tokens.extend(self.eot_tokens_)
                    weights.extend([w] * len(self.eot_tokens_))

            # Append result
            result_tokens.append(tokens)
            result_weights.append(weights)

        # Sanity check
        assert all_text_idx == len(all_text)

        return result_tokens, result_weights

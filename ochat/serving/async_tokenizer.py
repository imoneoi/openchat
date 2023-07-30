import ray


@ray.remote
class AsyncTokenizer:
    def __init__(self, model_type: str, model_path: str) -> None:
        from ochat.config.model_config import MODEL_CONFIG_MAP

        self.config = MODEL_CONFIG_MAP[model_type]
        self.role_map = {
            "user": "human",
            "assistant": self.config.ai_role
        }

        self.tokenizer = self.config.model_tokenizer_create(model_path)

    def _tokenize(self, text):
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer._tokenize(text))

    def _tokenize_special(self, special_name):
        return self.tokenizer.convert_tokens_to_ids(special_name)

    def tokenize(self, messages):
        # get conversation
        conversation = []
        for message in messages:
            msg_role, msg_text = message["role"], message["content"]
            if msg_role == "system":
                # FIXME: Ignoring system prompt.
                continue

            conversation.append({"from": self.role_map[msg_role], "value": msg_text})

        # append ai role
        if not (len(conversation) and conversation[-1]["from"] == self.config.ai_role):
            conversation.append({"from": self.config.ai_role})

        input_ids, _, _ = self.config.generate_conversation_template(self._tokenize, self._tokenize_special, system_prompt="", message_list=conversation)
        return input_ids

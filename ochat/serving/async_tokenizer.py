import ray

from ochat.config import Message, Conversation


@ray.remote
class AsyncTokenizer:
    def __init__(self, model_type: str, model_path: str) -> None:
        from ochat.config import MODEL_CONFIG_MAP

        config = MODEL_CONFIG_MAP[model_type]
        tokenizer = config.model_tokenizer_create(model_path)

        self.conv_template = config.conversation_template(tokenizer=tokenizer)

    def tokenize(self, messages, condition, enable_sys_prompt=False):
        # get system messages
        system_message = ""
        items = []

        for msg_raw in messages:
            msg = Message(**msg_raw)
            if msg.role == "system":
                # Use system prompt only when enabled
                if enable_sys_prompt:
                    system_message = msg.content.strip()

                continue

            items.append(msg)

        assert len(items)

        # append ai role
        if items[-1].role != "assistant":
            items.append(Message(role="assistant", content=""))

        tokens, _ = self.conv_template.tokenize_conversations([Conversation(items=items, system=system_message, condition=condition)],
                                                              inference=True)
        return tokens[0]
    
    def get_eot_tokens(self):
        assert len(self.conv_template.eot_tokens_) == 1

        return self.conv_template.eot_tokens_

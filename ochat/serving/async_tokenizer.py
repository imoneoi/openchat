import ray

from ochat.config import Message, Conversation


@ray.remote
class AsyncTokenizer:
    def __init__(self, model_type: str, model_path: str) -> None:
        from ochat.config import MODEL_CONFIG_MAP

        config = MODEL_CONFIG_MAP[model_type]
        tokenizer = config.model_tokenizer_create(model_path)

        self.conv_template = config.conversation_template(tokenizer=tokenizer)

    def tokenize(self, messages):
        # get system messages
        system_message = ""
        items = []

        for msg_raw in messages:
            msg = Message(**msg_raw)
            if msg.role == "system":
                system_message = msg.value
            else:
                items.append(msg)

        # append ai role
        if not (len(items) and items[-1].role != "assistant"):
            items.append(Message(role="assistant", value=""))

        tokens, _ = self.conv_template.tokenize_conversations(Conversation(items=items, system=system_message), inference=True)
        return tokens

from typing import Optional, Callable, Iterable

from pydantic import BaseModel, ConfigDict


class ModelConfig(BaseModel):
    # Alias
    serving_aliases: Iterable[str] = ()

    # Model
    model_max_context: int
    model_tokenizer_create: Callable
    model_create_for_training: Callable

    # conversation template
    conversation_template: Callable
    hf_chat_template: Optional[str] = None

    model_config = ConfigDict(protected_namespaces=())  # Disables warnings for the model_ namespace used above

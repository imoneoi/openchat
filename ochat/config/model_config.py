from typing import Callable

from pydantic import BaseModel


class ModelConfig(BaseModel):
    # Model
    model_max_context: int
    model_tokenizer_create: Callable
    model_create_for_training: Callable

    # conversation template
    conversation_template: Callable

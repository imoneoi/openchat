from typing import List, Optional, Dict
from dataclasses import dataclass
import argparse
import json
import logging
import os

import shortuuid
import fastapi
import uvicorn
from pydantic import BaseSettings
from fastapi import Depends, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer

import torch
import transformers

from ochat.config.model_config import MODEL_CONFIG_MAP, ModelConfig
from ochat.serving.inference import generate_stream
from ochat.serving import openai_api_protocol


@dataclass
class Model:
    model: transformers.PreTrainedModel = None
    tokenizer: transformers.PreTrainedTokenizer = None

    model_config: ModelConfig = None

    default_temperature: float = None
    default_top_p:       float = None

    max_generate_tokens:   int = None
    stream_tokens:         int = None


class AppSettings(BaseSettings):
    api_keys: List[str] = None


app_settings = AppSettings()
app = fastapi.FastAPI()

model = Model()


async def check_api_key(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
) -> str:
    if app_settings.api_keys:
        if auth is None or (token := auth.credentials) not in app_settings.api_keys:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "message": "",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "invalid_api_key",
                    }
                },
            )
        return token
    else:
        # api_keys not set; allow all
        return None


@app.get("/v1/models", dependencies=[Depends(check_api_key)])
async def show_available_models():
    return openai_api_protocol.ModelList(data=[
        openai_api_protocol.ModelCard(id=model.model_config.name, root=model.model_config.name, permission=[openai_api_protocol.ModelPermission()])
    ])


async def chat_completion_streaming_generator(stream_response_generator, logging_info):
    """
    Event stream format:
    https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#event_stream_format
    """
    id = f"chatcmpl-{shortuuid.random()}"

    # First chunk with role
    choice_data = openai_api_protocol.ChatCompletionResponseStreamChoice(
        index=0,
        delta=openai_api_protocol.DeltaMessage(role="assistant"),
        finish_reason=None,
    )
    chunk = openai_api_protocol.ChatCompletionStreamResponse(
        id=id, choices=[choice_data], model=model.model_config.name
    )
    yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"

    # Subsequent chunks with delta content
    logging_text = ""
    logging_finish_reason = ""

    for is_generating, content in stream_response_generator:
        if is_generating:
            choice_data = openai_api_protocol.ChatCompletionResponseStreamChoice(
                index=0,
                delta=openai_api_protocol.DeltaMessage(content=content),
                finish_reason=None,
            )
            logging_text += content
        else:
            # Finish chunk
            choice_data = openai_api_protocol.ChatCompletionResponseStreamChoice(
                index=0,
                delta=openai_api_protocol.DeltaMessage(content=None),
                finish_reason=content,
            )
            logging_finish_reason = content

        chunk = openai_api_protocol.ChatCompletionStreamResponse(
            id=id, choices=[choice_data], model=model.model_config.name
        )

        yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"

    # logging
    logging.info(json.dumps({"completion": logging_text, "finish_reason": logging_finish_reason, **logging_info}))

    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions", dependencies=[Depends(check_api_key)])
async def create_chat_completion(request: openai_api_protocol.ChatCompletionRequest):
    # get conversation
    role_map = {"user": "human", "assistant": model.model_config.ai_role}

    conversation = []
    for message in request.messages:
        msg_role, msg_text = message["role"], message["content"]
        if msg_role == "system":
            # FIXME: Ignoring system prompt.
            continue

        conversation.append({"from": role_map[msg_role], "value": msg_text})

    # get params
    generation_params = dict(
        temperature=request.temperature if request.temperature is not None else model.default_temperature,
        top_p=request.top_p if request.top_p is not None else model.default_top_p
    )

    stream_tokens = model.stream_tokens if request.stream else None

    # logging
    logging_info = {"conversation": conversation, "params": generation_params}

    # response
    stream_response = generate_stream(
        model=model.model, tokenizer=model.tokenizer, model_config=model.model_config,
        conversation=conversation,
        max_generated_tokens=model.max_generate_tokens,
        stream_period=stream_tokens,
        **generation_params
    )

    # stream
    if request.stream:
        return StreamingResponse(chat_completion_streaming_generator(stream_response, logging_info),
                                 media_type="text/event-stream")
    
    # normal
    is_generating, text = next(stream_response)
    assert is_generating

    is_generating, finish_reason = next(stream_response)
    assert not is_generating

    # logging
    logging.info(json.dumps({"completion": text, "finish_reason": finish_reason, **logging_info}))

    return openai_api_protocol.ChatCompletionResponse(
        model=model.model_config.name,
        choices=[openai_api_protocol.ChatCompletionResponseChoice(
            index=0,
            message=openai_api_protocol.ChatMessage(role="assistant", content=text),
            finish_reason=finish_reason
        )],
        usage=openai_api_protocol.UsageInfo()
    )


def main():
    parser = argparse.ArgumentParser(description="OpenChat ChatGPT-Compatible RESTful API server.")

    # Model
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--default_temperature", type=float, default=0.7)
    parser.add_argument("--default_top_p",       type=float, default=0.9)
    parser.add_argument("--max_generate_tokens", type=int,   default=768)

    parser.add_argument("--stream_tokens",       type=int,   default=6)

    # Server
    parser.add_argument("--host", type=str, default="localhost", help="host name")
    parser.add_argument("--port", type=int, default=18888, help="port number")
    parser.add_argument("--allow-credentials", action="store_true", help="allow credentials")
    parser.add_argument("--allowed-origins", type=json.loads, default=["*"], help="allowed origins")
    parser.add_argument("--allowed-methods", type=json.loads, default=["*"], help="allowed methods")
    parser.add_argument("--allowed-headers", type=json.loads, default=["*"], help="allowed headers")

    # Logging
    parser.add_argument("--logfile", type=str, default=None, help="Log file")

    # Server API keys
    parser.add_argument("--api-keys", type=lambda s: s.split(","), help="Optional list of comma separated API keys",)
    args = parser.parse_args()

    # Load model
    model.model_config        = MODEL_CONFIG_MAP[args.model_type]

    model.model     = model.model_config.model_create(args.model_path).cuda()
    model.tokenizer = model.model_config.model_tokenizer_create(args.model_path)
    model.model.eval()

    # Config
    model.default_temperature = args.default_temperature
    model.default_top_p       = args.default_top_p
    model.max_generate_tokens = args.max_generate_tokens

    model.stream_tokens       = args.stream_tokens

    # Test
    test_question = "How can I improve my time management skills?"
    print (f"Test generate: {test_question}")
    for is_generating, t in generate_stream(model.model, model.tokenizer, model.model_config,
                            conversation=[{"from": "human", "value": test_question}],
                            max_generated_tokens=model.max_generate_tokens, stream_period=6,
                            temperature=model.default_temperature, top_p=model.default_top_p):
        print (t, end="")

    # Logging setup
    if args.logfile is not None:
        logging.basicConfig(level=logging.INFO, filename=args.logfile, filemode="a")
    else:
        # To console
        logging.basicConfig(level=logging.INFO)

    # Load app
    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )
    app_settings.api_keys = args.api_keys

    logging.info(f"args: {args}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

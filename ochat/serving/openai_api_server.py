# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/api_server.py

import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass
from http import HTTPStatus
from logging.handlers import RotatingFileHandler
from typing import AsyncGenerator, Optional

import fastapi
import ray
import uvicorn
from fastapi import BackgroundTasks, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer
from transformers.utils.hub import cached_file
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

from ochat.config import MODEL_CONFIG_MAP
from ochat.serving import async_tokenizer, openai_api_protocol

TIMEOUT_KEEP_ALIVE = 5  # seconds


@dataclass
class ModelConfig:
    names: set = None

    max_length: int = None
    stream_period: int = None
    eot_tokens: list = None

    enable_sys_prompt: bool = None
    api_keys: list = None


logger = None
app = fastapi.FastAPI()

model = ModelConfig()
tokenizer = None


def _strip_first_space(s: str):
    if s[0] == " ":
        return s[1:]
    return s


def log_request(created_time: int, request: openai_api_protocol.ChatCompletionRequest, output: RequestOutput):
    if logger is not None:
        logger.info(openai_api_protocol.LoggingRecord(
            time=created_time,
            request=request,
            outputs=[o.text for o in output.outputs]
        ).json(exclude_unset=True, ensure_ascii=False))


def create_error_response(status_code: HTTPStatus,
                          message: str) -> JSONResponse:
    return JSONResponse(openai_api_protocol.ErrorResponse(message=message,
                                                          type="invalid_request_error").dict(),
                        status_code=status_code.value)


def check_model(request) -> Optional[JSONResponse]:
    if request.model in model.names:
        return

    return create_error_response(
        HTTPStatus.NOT_FOUND,
        f"The model `{request.model}` does not exist.",
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):  # pylint: disable=unused-argument
    return create_error_response(HTTPStatus.BAD_REQUEST, str(exc))


async def check_api_key(
    auth: Optional[HTTPAuthorizationCredentials] = fastapi.Depends(HTTPBearer(auto_error=False)),
):
    if not model.api_keys:
        return

    if auth is None or auth.credentials not in model.api_keys:
        raise fastapi.HTTPException(
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


@app.get("/v1/models", dependencies=[fastapi.Depends(check_api_key)])
async def show_available_models():
    """Show available models. Right now we only have one model."""
    return openai_api_protocol.ModelList(data=[
        openai_api_protocol.ModelCard(id=name,
                                      root=name,
                                      permission=[openai_api_protocol.ModelPermission()])
    for name in model.names])


@app.post("/v1/chat/completions", dependencies=[fastapi.Depends(check_api_key)])
async def create_chat_completion(raw_request: Request, background_tasks: BackgroundTasks):
    """Completion API similar to OpenAI's API.

    See  https://platform.openai.com/docs/api-reference/chat/create
    for the API specification. This API mimics the OpenAI ChatCompletion API.

    NOTE: Currently we do not support the following features:
        - function_call (Users should implement this by themselves)
        - logit_bias (to be supported by vLLM engine)
    """

    request = openai_api_protocol.ChatCompletionRequest(**await raw_request.json())

    error_check_ret = check_model(request)
    if error_check_ret is not None:
        return error_check_ret

    if request.logit_bias is not None and len(request.logit_bias) > 0:
        # TODO: support logit_bias in vLLM engine.
        return create_error_response(HTTPStatus.BAD_REQUEST,
                                     "logit_bias is not currently supported")

    # input ids
    input_ids = await tokenizer.tokenize.remote(request.messages, condition=request.condition,
                                                enable_sys_prompt=model.enable_sys_prompt)
    input_num_tokens = len(input_ids)

    # check length
    if request.max_tokens is None:
        request.max_tokens = model.max_length - input_num_tokens

    if input_num_tokens + request.max_tokens > model.max_length:
        return input_ids, create_error_response(
            HTTPStatus.BAD_REQUEST,
            f"This model's maximum context length is {model.max_length} tokens. "
            f"However, you requested {input_num_tokens + request.max_tokens} tokens "
            f"({input_num_tokens} in the messages, "
            f"{request.max_tokens} in the completion). "
            f"Please reduce the length of the messages or completion.",
        )

    # completion
    model_name = request.model
    request_id = f"cmpl-{random_uuid()}"
    created_time = int(time.time())

    try:
        sampling_params = SamplingParams(
            n=request.n,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            # Override stop tokens
            stop_token_ids=model.eot_tokens,
            ignore_eos=True
        )
    except ValueError as e:
        return create_error_response(HTTPStatus.BAD_REQUEST, str(e))

    result_generator = engine.generate(prompt=None,
                                       prompt_token_ids=input_ids,
                                       sampling_params=sampling_params,
                                       request_id=request_id)

    def create_stream_response_json(
        index: int,
        text: str,
        finish_reason: Optional[str] = None,
    ) -> str:
        choice_data = openai_api_protocol.ChatCompletionResponseStreamChoice(
            index=index,
            delta=openai_api_protocol.DeltaMessage(content=text),
            finish_reason=finish_reason,
        )
        response = openai_api_protocol.ChatCompletionStreamResponse(
            id=request_id,
            choices=[choice_data],
            model=model_name,
        )

        return response.json(exclude_unset=True, ensure_ascii=False)

    async def completion_stream_generator() -> AsyncGenerator[str, None]:
        # First chunk with role
        for i in range(request.n):
            choice_data = openai_api_protocol.ChatCompletionResponseStreamChoice(
                index=i,
                delta=openai_api_protocol.DeltaMessage(role="assistant"),
                finish_reason=None,
            )
            chunk = openai_api_protocol.ChatCompletionStreamResponse(id=request_id,
                                                                     choices=[choice_data],
                                                                     model=model_name)

            yield f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"

        previous_texts = [""] * request.n
        previous_num_tokens = [0] * request.n

        stream_index = 0
        final_res = None
        is_first = True
        async for res in result_generator:
            stream_index += 1
            final_res = res

            for output in res.outputs:
                # stream on end or every stream_period
                if (stream_index % model.stream_period == 0) or (output.finish_reason is not None):
                    i = output.index
                    delta_text = output.text[len(previous_texts[i]):]
                    if "\ufffd" not in delta_text:
                        previous_texts[i] = output.text
                        previous_num_tokens[i] = len(output.token_ids)

                        if is_first:
                            # Strip first space
                            is_first = False
                            delta_text = _strip_first_space(delta_text)

                        yield f"data: {create_stream_response_json(index=i, text=delta_text)}\n\n"
                        if output.finish_reason is not None:
                            yield f"data: {create_stream_response_json(index=i, text='', finish_reason=output.finish_reason)}\n\n"

        yield "data: [DONE]\n\n"

        # Log request
        background_tasks.add_task(log_request, created_time, request, final_res)

    # Streaming response
    if request.stream:
        return StreamingResponse(completion_stream_generator(),
                                 media_type="text/event-stream")

    # Non-streaming response
    final_res: RequestOutput = None
    async for res in result_generator:
        if await raw_request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)
            return create_error_response(HTTPStatus.BAD_REQUEST,
                                         "Client disconnected")
        final_res = res
    assert final_res is not None
    choices = []
    for output in final_res.outputs:
        choice_data = openai_api_protocol.ChatCompletionResponseChoice(
            index=output.index,
            message=openai_api_protocol.ChatMessage(role="assistant", content=_strip_first_space(output.text)),
            finish_reason=output.finish_reason,
        )
        choices.append(choice_data)

    num_prompt_tokens = len(final_res.prompt_token_ids)
    num_generated_tokens = sum(
        len(output.token_ids) for output in final_res.outputs)
    usage = openai_api_protocol.UsageInfo(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=num_generated_tokens,
        total_tokens=num_prompt_tokens + num_generated_tokens,
    )
    response = openai_api_protocol.ChatCompletionResponse(
        id=request_id,
        created=created_time,
        model=model_name,
        choices=choices,
        usage=usage,
    )

    # Log request
    background_tasks.add_task(log_request, created_time, request, final_res)

    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenChat OpenAI-Compatible RESTful API server.")

    # Model
    parser.add_argument("--model-type", type=str, default=None, help="Model type. Leave empty to auto-detect.")

    parser.add_argument("--stream-period", type=int, default=6, help="Number of tokens per stream event")
    parser.add_argument("--api-keys", type=str, nargs="*", default=[], help="Allowed API Keys. Leave blank to not verify")
    parser.add_argument("--enable-sys-prompt", default=False, action="store_true")

    # Server
    parser.add_argument("--host", type=str, default="localhost", help="Host name")
    parser.add_argument("--port", type=int, default=18888, help="Port number")
    parser.add_argument("--allow-credentials", action="store_true", help="Allow credentials")
    parser.add_argument("--allowed-origins", type=json.loads, default=["*"], help="Allowed origins")
    parser.add_argument("--allowed-methods", type=json.loads, default=["*"], help="Allowed methods")
    parser.add_argument("--allowed-headers", type=json.loads, default=["*"], help="Allowed headers")

    # Logging
    parser.add_argument("--log-file", type=str, default=None, help="Log file. Leave blank to disable logging")
    parser.add_argument("--log-max-mb", type=int, default=128, help="Max log size in MB")
    parser.add_argument("--log-max-count", type=int, default=10, help="Max log file versions to keep")

    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    # App and logging
    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    if args.log_file:
        logger = logging.getLogger(__name__)

        logger.setLevel(logging.INFO)
        logger.addHandler(RotatingFileHandler(
            args.log_file,
            maxBytes=args.log_max_mb * 1048576,
            backupCount=args.log_max_count)
        )
        logger.propagate = False

    # Load model type
    if args.model_type is None:
        with open(cached_file(path_or_repo_id=args.model, filename="openchat.json"), "r") as f:
            args.model_type = json.load(f)["model_type"]

    # Load tokenizer
    tokenizer = async_tokenizer.AsyncTokenizer.remote(args.model_type, args.model)

    # Model config
    model.names = set(list(MODEL_CONFIG_MAP[args.model_type].serving_aliases) + [args.model_type])
    model.max_length = MODEL_CONFIG_MAP[args.model_type].model_max_context
    model.eot_tokens = ray.get(tokenizer.get_eot_tokens.remote())

    model.enable_sys_prompt = args.enable_sys_prompt
    model.stream_period = args.stream_period
    model.api_keys = args.api_keys

    # Set max num batched tokens
    args.max_num_batched_tokens = max(args.max_num_batched_tokens or model.max_length, model.max_length)
    args.max_model_len = model.max_length

    # Load model engine
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    engine_model_config = asyncio.run(engine.get_model_config())

    # Run
    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="info",
                access_log=False,
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)

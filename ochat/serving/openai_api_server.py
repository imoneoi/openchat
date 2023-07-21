# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/api_server.py

import argparse
import asyncio
from http import HTTPStatus
import json
import time
from typing import AsyncGenerator, Optional, Union
from dataclasses import dataclass
from functools import partial

import fastapi
from fastapi import BackgroundTasks, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

import uvicorn

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils import tokenizer
from vllm.utils import random_uuid

from ochat.config.model_config import MODEL_CONFIG_MAP, ModelConfig
from ochat.serving import openai_api_protocol


TIMEOUT_KEEP_ALIVE = 5  # seconds


@dataclass
class Model:
    name: str = None

    config: ModelConfig = None
    tokenizer: object = None

    stream_period: int = None


logger = init_logger(__name__)
app = fastapi.FastAPI()

model = Model()


def create_error_response(status_code: HTTPStatus,
                          message: str) -> JSONResponse:
    return JSONResponse(openai_api_protocol.ErrorResponse(message=message,
                                                          type="invalid_request_error").dict(),
                        status_code=status_code.value)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):  # pylint: disable=unused-argument
    return create_error_response(HTTPStatus.BAD_REQUEST, str(exc))


async def check_model(request) -> Optional[JSONResponse]:
    if request.model == model.name:
        return

    ret = create_error_response(
        HTTPStatus.NOT_FOUND,
        f"The model `{request.model}` does not exist.",
    )
    return ret


@app.get("/v1/models")
async def show_available_models():
    """Show available models. Right now we only have one model."""
    return openai_api_protocol.ModelList(data=[
        openai_api_protocol.ModelCard(id=model.name,
                                      root=model.name,
                                      permission=[openai_api_protocol.ModelPermission()])
    ])


def _tokenize(text):
    return model.tokenizer.convert_tokens_to_ids(model.tokenizer._tokenize(text))


def _tokenize_special(special_name):
    return model.tokenizer.convert_tokens_to_ids(special_name)


@app.post("/v1/chat/completions")
async def create_chat_completion(raw_request: Request):
    """Completion API similar to OpenAI's API.

    See  https://platform.openai.com/docs/api-reference/chat/create
    for the API specification. This API mimics the OpenAI ChatCompletion API.

    NOTE: Currently we do not support the following features:
        - function_call (Users should implement this by themselves)
        - logit_bias (to be supported by vLLM engine)
    """

    request = openai_api_protocol.ChatCompletionRequest(**await raw_request.json())

    error_check_ret = await check_model(request)
    if error_check_ret is not None:
        return error_check_ret

    if request.logit_bias is not None:
        # TODO: support logit_bias in vLLM engine.
        return create_error_response(HTTPStatus.BAD_REQUEST,
                                     "logit_bias is not currently supported")

    # get conversation
    role_map = {"user": "human", "assistant": model.config.ai_role}

    conversation = []
    for message in request.messages:
        msg_role, msg_text = message["role"], message["content"]
        if msg_role == "system":
            # FIXME: Ignoring system prompt.
            continue

        conversation.append({"from": role_map[msg_role], "value": msg_text})

    # append ai role
    if not (len(conversation) and conversation[-1]["from"] == model.config.ai_role):
        conversation.append({"from": model.config.ai_role})

    input_ids, _, _ = model.config.generate_conversation_template(_tokenize, _tokenize_special, conversation)

    # check length
    request.max_tokens = min(request.max_tokens, model.config.model_max_context - len(input_ids))
    if request.max_tokens <= 0:
        return create_error_response(
            HTTPStatus.BAD_REQUEST,
            f"This model's maximum context length is {model.config.model_max_context} tokens. "
            f"However, you requested {len(input_ids)} tokens. "
            f"Please reduce the length of the messages.",
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
            stop=[model.config.eot_token],
            max_tokens=request.max_tokens
        )
    except ValueError as e:
        return create_error_response(HTTPStatus.BAD_REQUEST, str(e))

    result_generator = engine.generate(prompt=None,
                                       prompt_token_ids=input_ids,
                                       sampling_params=sampling_params,
                                       request_id=request_id)

    async def abort_request() -> None:
        await engine.abort(request_id)

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
        async for res in result_generator:
            stream_index += 1

            for output in res.outputs:
                # stream on end or every stream_period
                if (stream_index % model.stream_period == 0) or (output.finish_reason is not None):
                    i = output.index
                    delta_text = output.text[len(previous_texts[i]):]
                    previous_texts[i] = output.text
                    previous_num_tokens[i] = len(output.token_ids)

                    yield f"data: {create_stream_response_json(index=i, text=delta_text)}\n\n"
                    if output.finish_reason is not None:
                        yield f"data: {create_stream_response_json(index=i, text='', finish_reason=output.finish_reason)}\n\n"

        yield "data: [DONE]\n\n"

    # Streaming response
    if request.stream:
        background_tasks = BackgroundTasks()
        # Abort the request if the client disconnects.
        background_tasks.add_task(abort_request)
        return StreamingResponse(completion_stream_generator(),
                                 media_type="text/event-stream",
                                 background=background_tasks)

    # Non-streaming response
    final_res: RequestOutput = None
    async for res in result_generator:
        if await raw_request.is_disconnected():
            # Abort the request if the client disconnects.
            await abort_request()
            return create_error_response(HTTPStatus.BAD_REQUEST,
                                         "Client disconnected")
        final_res = res
    assert final_res is not None
    choices = []
    for output in final_res.outputs:
        choice_data = openai_api_protocol.ChatCompletionResponseChoice(
            index=output.index,
            message=openai_api_protocol.ChatMessage(role="assistant", content=output.text),
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

    if request.stream:
        # When user requests streaming but we don't stream, we still need to
        # return a streaming response with a single event.
        response_json = response.json(ensure_ascii=False)

        async def fake_stream_generator() -> AsyncGenerator[str, None]:
            yield f"data: {response_json}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(fake_stream_generator(),
                                 media_type="text/event-stream")

    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenChat OpenAI-Compatible RESTful API server.")

    # Model
    parser.add_argument("--model_type", type=str, required=True)

    parser.add_argument("--stream_period", type=int, default=6)

    # Server
    parser.add_argument("--host", type=str, default="localhost", help="host name")
    parser.add_argument("--port", type=int, default=18888, help="port number")
    parser.add_argument("--allow-credentials", action="store_true", help="allow credentials")
    parser.add_argument("--allowed-origins", type=json.loads, default=["*"], help="allowed origins")
    parser.add_argument("--allowed-methods", type=json.loads, default=["*"], help="allowed methods")
    parser.add_argument("--allowed-headers", type=json.loads, default=["*"], help="allowed headers")

    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    # Model config
    model.name = args.model_type
    model.config = MODEL_CONFIG_MAP[args.model_type]
    model.tokenizer = model.config.model_tokenizer_create(args.model)

    model.stream_period = args.stream_period

    # Load model
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    engine_model_config = asyncio.run(engine.get_model_config())

    # Run
    logger.info(f"args: {args}")

    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="info",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)

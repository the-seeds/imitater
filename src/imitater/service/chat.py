import argparse
import uuid
from typing import Any, AsyncGenerator, Dict, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException, status
from sse_starlette import EventSourceResponse

from ..model import ChatConfig, ChatModel
from ..utils.generic import dictify, jsonify
from .protocol import (
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionStreamResponse,
    ChatCompletionStreamResponseChoice,
    Finish,
    Function,
    FunctionCall,
    Role,
    UsageInfo,
)


async def create_chat_completion(
    chat_model: "ChatModel", request: "ChatCompletionRequest"
) -> Union["ChatCompletionResponse", "EventSourceResponse"]:
    msg_id = uuid.uuid4().hex
    input_kwargs = {
        "messages": [dictify(message) for message in request.messages],
        "request_id": "chatcmpl-{}".format(msg_id),
        "temperature": request.temperature,
        "top_p": request.top_p,
        "max_tokens": request.max_tokens,
    }

    if request.stream:
        if request.tools is not None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot stream function calls.")

        generator = create_stream_chat_completion(chat_model, request, input_kwargs)
        return EventSourceResponse(generator, media_type="text/event-stream")

    if request.tools is not None:
        input_kwargs["tools"] = [dictify(tool) for tool in request.tools]
        result, prompt_tokens, completion_tokens = await chat_model.function_call(**input_kwargs)
    else:
        result, prompt_tokens, completion_tokens = await chat_model.chat(**input_kwargs)

    if isinstance(result, tuple):
        name, arguments = result[0], result[1]
        tool_call = FunctionCall(id="call_{}".format(msg_id), function=Function(name=name, arguments=arguments))
        choice = ChatCompletionResponseChoice(
            index=0,
            message=ChatCompletionMessage(role=Role.ASSISTANT, tool_calls=[tool_call]),
            finish_reason=Finish.TOOL,
        )
    else:
        choice = ChatCompletionResponseChoice(
            index=0, message=ChatCompletionMessage(role=Role.ASSISTANT, content=result), finish_reason=Finish.STOP
        )

    usage = UsageInfo(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return ChatCompletionResponse(
        id=input_kwargs["request_id"],
        model=request.model,
        choices=[choice],
        usage=usage,
    )


def create_stream_chunk(
    request_id: str, model: str, delta: "ChatCompletionMessage", finish_reason: Optional[Finish] = None
) -> str:
    choice = ChatCompletionStreamResponseChoice(index=0, delta=delta, finish_reason=finish_reason)
    chunk = ChatCompletionStreamResponse(id=request_id, model=model, choices=[choice])
    return jsonify(chunk)


async def create_stream_chat_completion(
    chat_model: "ChatModel", request: "ChatCompletionRequest", input_kwargs: Dict[str, Any]
) -> AsyncGenerator[str, None]:
    yield create_stream_chunk(
        input_kwargs["request_id"], request.model, ChatCompletionMessage(role=Role.ASSISTANT, content="")
    )
    async for new_token in chat_model.stream_chat(**input_kwargs):
        yield create_stream_chunk(input_kwargs["request_id"], request.model, ChatCompletionMessage(content=new_token))

    yield create_stream_chunk(
        input_kwargs["request_id"], request.model, ChatCompletionMessage(), finish_reason=Finish.STOP
    )
    yield "[DONE]"


def launch_server(config: "ChatConfig") -> None:
    model = ChatModel(config)
    app = FastAPI()

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse, status_code=status.HTTP_200_OK)
    async def create_chat_completion_v1(request: "ChatCompletionRequest"):
        return await create_chat_completion(model, request)

    uvicorn.run(app, host="127.0.0.1", port=config.port, workers=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ChatConfig.add_cli_args(parser)
    args = parser.parse_args()
    config = ChatConfig.from_cli_args(args)
    launch_server(config)

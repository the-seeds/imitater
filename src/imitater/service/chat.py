import argparse
import uuid
from typing import Any, AsyncGenerator, Dict

import uvicorn
from fastapi import FastAPI, HTTPException, status
from sse_starlette import EventSourceResponse

from ..model import ChatConfig, ChatModel
from ..utils.generic import dictify
from .common import create_stream_chunk
from .protocol import (
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    Finish,
    Function,
    FunctionCall,
    Role,
    UsageInfo,
)


async def _create_stream_chat_completion(
    request: "ChatCompletionRequest", model: "ChatModel", input_kwargs: Dict[str, Any]
) -> AsyncGenerator[str, None]:
    yield create_stream_chunk(
        input_kwargs["request_id"], request.model, ChatCompletionMessage(role=Role.ASSISTANT, content="")
    )
    async for new_token in model.stream_chat(**input_kwargs):
        yield create_stream_chunk(input_kwargs["request_id"], request.model, ChatCompletionMessage(content=new_token))

    yield create_stream_chunk(
        input_kwargs["request_id"], request.model, ChatCompletionMessage(), finish_reason=Finish.STOP
    )
    yield "[DONE]"


async def _create_local_chat_completion(
    request: "ChatCompletionRequest", model: "ChatModel"
) -> "ChatCompletionResponse":
    msg_id = uuid.uuid4().hex
    input_kwargs = {
        "messages": [dictify(message) for message in request.messages],
        "request_id": "chatcmpl-{}".format(msg_id),
        "temperature": request.temperature,
        "top_p": request.top_p,
        "max_tokens": request.max_tokens,
        "stop": request.stop,
    }

    if request.n != 1:
        raise NotImplementedError("Multiple responses are not supported yet.")

    if request.stream:
        if request.tools is not None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot stream function calls.")

        generator = _create_stream_chat_completion(request, model, input_kwargs)
        return EventSourceResponse(generator, media_type="text/event-stream")

    if request.tools is not None:
        input_kwargs["tools"] = [dictify(tool) for tool in request.tools]
        result, prompt_tokens, completion_tokens = await model.function_call(**input_kwargs)
    else:
        result, prompt_tokens, completion_tokens = await model.chat(**input_kwargs)

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


def launch_server(config: "ChatConfig") -> None:
    model = ChatModel(config)
    app = FastAPI()

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse, status_code=status.HTTP_200_OK)
    async def create_chat_completion(request: "ChatCompletionRequest"):
        return await _create_local_chat_completion(request, model)

    uvicorn.run(app, port=config.port)


def main():
    parser = argparse.ArgumentParser()
    ChatConfig.add_cli_args(parser)
    args = parser.parse_args()
    config = ChatConfig.from_cli_args(args)
    launch_server(config)


if __name__ == "__main__":
    main()

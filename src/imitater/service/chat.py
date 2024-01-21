import uuid
from typing import TYPE_CHECKING, Any, Dict, Generator, Union

from sse_starlette import EventSourceResponse

from ..utils.generic import dictify, jsonify
from .protocol import (
    ChatCompletionMessage,
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


if TYPE_CHECKING:
    from ..model import ChatModel
    from .protocol import ChatCompletionRequest


async def create_chat_completion(
    chat_model: "ChatModel", request: "ChatCompletionRequest"
) -> Union[ChatCompletionResponse, EventSourceResponse]:
    msg_id = uuid.uuid4().hex
    input_kwargs = {
        "messages": [dictify(message) for message in request.messages],
        "request_id": "chatcmpl-{}".format(msg_id),
        "temperature": request.temperature,
        "top_p": request.top_p,
        "max_tokens": request.max_tokens,
    }

    if request.stream:
        generator = create_stream_chat_completion(request, input_kwargs)
        return EventSourceResponse(generator, media_type="text/event-stream")

    if request.tools is not None:
        input_kwargs["tools"] = [dictify(tool) for tool in request.tools]
        result = await chat_model.function_call(**input_kwargs)
    else:
        result = await chat_model.chat(**input_kwargs)

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

    return ChatCompletionResponse(
        id=input_kwargs["request_id"],
        model=request.model,
        choices=[choice],
        usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0),
    )


async def create_stream_chat_completion(
    chat_model: "ChatModel", request: "ChatCompletionRequest", input_kwargs: Dict[str, Any]
) -> Generator[str, None, None]:
    choice = ChatCompletionStreamResponseChoice(
        index=0, delta=ChatCompletionMessage(role=Role.ASSISTANT, content=""), finish_reason=None
    )
    chunk = ChatCompletionStreamResponse(id=input_kwargs["request_id"], model=request.model, choices=[choice])
    yield jsonify(chunk)

    async for new_token in chat_model.stream_chat(**input_kwargs):
        choice = ChatCompletionStreamResponseChoice(
            index=0, delta=ChatCompletionMessage(content=new_token), finish_reason=None
        )
        chunk = ChatCompletionStreamResponse(id=input_kwargs["request_id"], model=request.model, choices=[choice])
        yield jsonify(chunk)

    choice = ChatCompletionStreamResponseChoice(index=0, delta=ChatCompletionMessage(), finish_reason=Finish.STOP)
    chunk = ChatCompletionStreamResponse(id=input_kwargs["request_id"], model=request.model, choices=[choice])
    yield jsonify(chunk)
    yield "[DONE]"

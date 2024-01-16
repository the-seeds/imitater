import base64
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict

import numpy as np
import uvicorn
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette import EventSourceResponse

from ..model import ChatModel, EmbedModel
from ..utils.generic import dictify, jsonify, torch_gc
from .protocol import (
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionStreamResponse,
    ChatCompletionStreamResponseChoice,
    Embeddings,
    EmbeddingsRequest,
    EmbeddingsResponse,
    Finish,
    Function,
    FunctionCall,
    ModelCard,
    ModelList,
    Role,
    UsageInfo,
)


@asynccontextmanager
async def lifespan(app: "FastAPI") -> None:
    yield
    torch_gc()


def launch_app() -> None:
    app = FastAPI(lifespan=lifespan)
    chat_model = ChatModel()
    embed_model = EmbedModel()

    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
    )

    @app.get("/v1/models", response_model=ModelList)
    async def list_models():
        model_card = ModelCard(id="gpt-3.5-turbo")
        return ModelList(data=[model_card])

    @app.post("/v1/embeddings", response_model=EmbeddingsResponse, status_code=status.HTTP_200_OK)
    async def create_embeddings(request: EmbeddingsRequest):
        texts = request.input
        if isinstance(texts, str):
            texts = [texts]

        embed_output = await embed_model(texts)
        embeddings = []
        for i in range(len(embed_output)):
            embed_data = embed_output[i]
            if request.encoding_format == "base64":
                embed_data = base64.b64encode(np.array(embed_data, dtype=np.float32))
            embeddings.append(Embeddings(embedding=embed_data, index=i))

        return EmbeddingsResponse(
            data=embeddings,
            model=request.model,
            usage=UsageInfo(prompt_tokens=0, completion_tokens=None, total_tokens=0),
        )

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse, status_code=status.HTTP_200_OK)
    async def create_chat_completion(request: ChatCompletionRequest):
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

    async def create_stream_chat_completion(request: ChatCompletionRequest, input_kwargs: Dict[str, Any]):
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

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("SERVICE_PORT", 8000)), workers=1)


if __name__ == "__main__":
    launch_app()

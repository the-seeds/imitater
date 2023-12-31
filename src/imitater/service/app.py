import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette import EventSourceResponse

from ..model.chat_model import ChatModel
from ..model.embed_model import EmbedModel
from ..utils.generic import dictify, jsonify, torch_gc
from .protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionStreamResponse,
    ChatCompletionStreamResponseChoice,
    ChatMessage,
    DeltaMessage,
    Embeddings,
    EmbeddingsRequest,
    EmbeddingsResponse,
    Finish,
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
            embeddings.append(Embeddings(embedding=embed_output[i], index=i))

        return EmbeddingsResponse(
            data=embeddings,
            model=request.model,
            usage=UsageInfo(prompt_tokens=0, completion_tokens=None, total_tokens=0),
        )

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse, status_code=status.HTTP_200_OK)
    async def create_chat_completion(request: ChatCompletionRequest):
        input_kwargs = {
            "messages": [dictify(message) for message in request.messages],
            "request_id": "chatcmpl-{}".format(uuid.uuid4().hex),
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
        }

        if request.stream:
            generator = create_stream_chat_completion(request, input_kwargs)
            return EventSourceResponse(generator, media_type="text/event-stream")

        response = await chat_model.chat(**input_kwargs)
        choice = ChatCompletionResponseChoice(
            index=0, message=ChatMessage(role=Role.ASSISTANT, content=response), finish_reason=Finish.STOP
        )
        return ChatCompletionResponse(
            id=input_kwargs["request_id"],
            model=request.model,
            choices=[choice],
            usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )

    async def create_stream_chat_completion(request: ChatCompletionRequest, input_kwargs: Dict[str, Any]):
        choice = ChatCompletionStreamResponseChoice(
            index=0, delta=DeltaMessage(role=Role.ASSISTANT, content=""), finish_reason=None
        )
        chunk = ChatCompletionStreamResponse(id=input_kwargs["request_id"], model=request.model, choices=[choice])
        yield jsonify(chunk)

        async for new_token in chat_model.stream_chat(**input_kwargs):
            choice = ChatCompletionStreamResponseChoice(
                index=0, delta=DeltaMessage(content=new_token), finish_reason=None
            )
            chunk = ChatCompletionStreamResponse(id=input_kwargs["request_id"], model=request.model, choices=[choice])
            yield jsonify(chunk)

        choice = ChatCompletionStreamResponseChoice(index=0, delta=DeltaMessage(), finish_reason=Finish.STOP)
        chunk = ChatCompletionStreamResponse(id=input_kwargs["request_id"], model=request.model, choices=[choice])
        yield jsonify(chunk)
        yield "[DONE]"

    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)


if __name__ == "__main__":
    launch_app()

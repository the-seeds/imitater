import os
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware

from ..config import Config
from ..model import ChatModel, EmbedModel
from ..utils.generic import torch_gc
from .chat import create_chat_completion
from .embed import create_embeddings
from .protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    EmbeddingsRequest,
    EmbeddingsResponse,
    ModelCard,
    ModelList,
)


@asynccontextmanager
async def lifespan(app: "FastAPI") -> None:
    yield
    torch_gc()


class Imitater:
    def __init__(self) -> None:
        self.app = FastAPI(lifespan=lifespan)
        self.models = []
        load_dotenv()

    def _load_config(self) -> None:
        self._config = Config(
            agent_type=os.environ.get("AGENT_TYPE"),
            chat_model_path=os.environ.get("CHAT_MODEL_PATH"),
            chat_model_device=list(map(int, os.environ.get("CHAT_MODEL_DEVICE").split(","))),
            chat_template_path=os.environ.get("CHAT_TEMPLATE_PATH"),
            generation_config_path=os.environ.get("GENERATION_CONFIG_PATH"),
            embed_model_path=os.environ.get("EMBED_MODEL_PATH"),
            embed_model_device=list(map(int, os.environ.get("EMBED_MODEL_DEVICE").split(","))),
            embed_batch_size=int(os.environ.get("EMBED_BATCH_SIZE", 16)),
        )

    def _load_models(self) -> None:
        self._chat_model = ChatModel(self._config)
        self.models.append("gpt-3.5-turbo")
        self._embed_model = EmbedModel(self._config)
        self.models.append("text-embedding-ada-002")

    def launch(self) -> None:
        self.app.add_middleware(
            CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
        )

        self._load_config()
        self._load_models()

        @self.app.get("/v1/models", response_model=ModelList)
        async def list_models():
            return ModelList(data=[ModelCard(id=model) for model in self.models])

        @self.app.post("/v1/embeddings", response_model=EmbeddingsResponse, status_code=status.HTTP_200_OK)
        async def create_embeddings_v1(request: "EmbeddingsRequest"):
            return await create_embeddings(self._embed_model, request)

        @self.app.post("/v1/chat/completions", response_model=ChatCompletionResponse, status_code=status.HTTP_200_OK)
        async def create_chat_completion_v1(request: "ChatCompletionRequest"):
            return await create_chat_completion(self._chat_model, request)

        uvicorn.run(self.app, host="0.0.0.0", port=int(os.environ.get("SERVICE_PORT", 8000)), workers=1)

import argparse
import base64
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, status

from ..model import EmbedConfig, EmbedModel
from .protocol import (
    Embeddings,
    EmbeddingsRequest,
    EmbeddingsResponse,
    UsageInfo,
)


async def create_embeddings(embed_model: "EmbedModel", request: "EmbeddingsRequest") -> "EmbeddingsResponse":
    texts = request.input
    if isinstance(texts, str):
        texts = [texts]

    embed_output, embed_tokens = await embed_model.embed(texts)
    embeddings = []
    for i, embed_data in enumerate(embed_output):
        if request.encoding_format == "base64":
            embedding = base64.b64encode(embed_data)
        else:
            embedding = embed_data.tolist()

        embeddings.append(Embeddings(embedding=embedding, index=i))

    return EmbeddingsResponse(
        data=embeddings,
        model=request.model,
        usage=UsageInfo(prompt_tokens=embed_tokens, completion_tokens=None, total_tokens=embed_tokens),
    )


def launch_server(config: "EmbedConfig") -> None:
    model = EmbedModel(config)

    @asynccontextmanager
    async def lifespan(app: "FastAPI") -> None:
        await model.startup()
        yield
        await model.shutdown()

    app = FastAPI(lifespan=lifespan)

    @app.post("/v1/embeddings", response_model=EmbeddingsResponse, status_code=status.HTTP_200_OK)
    async def create_embeddings_v1(request: "EmbeddingsRequest"):
        return await create_embeddings(model, request)

    uvicorn.run(app, host="127.0.0.1", port=config.port, workers=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    EmbedConfig.add_cli_args(parser)
    args = parser.parse_args()
    config = EmbedConfig.from_cli_args(args)
    launch_server(config)

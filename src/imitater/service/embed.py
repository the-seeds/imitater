import base64
from typing import TYPE_CHECKING

import numpy as np

from .protocol import (
    Embeddings,
    EmbeddingsRequest,
    EmbeddingsResponse,
    UsageInfo,
)


if TYPE_CHECKING:
    from ..model import EmbedModel


async def create_embeddings(embed_model: "EmbedModel", request: "EmbeddingsRequest") -> "EmbeddingsResponse":
    texts = request.input
    if isinstance(texts, str):
        texts = [texts]

    embed_output = await embed_model.embed(texts)
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

import asyncio
import os
from typing import TYPE_CHECKING, List, Optional

import torch
from transformers import AutoModel, AutoTokenizer


if TYPE_CHECKING:
    from transformers import BatchEncoding, PreTrainedModel, PreTrainedTokenizerBase


@torch.inference_mode()
def _get_embeddings(model: "PreTrainedModel", batch_encoding: "BatchEncoding") -> List[List[float]]:
    output = model(**batch_encoding.to(model.device))
    embeddings = output[0][:, 0]
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1).tolist()
    return embeddings


class EmbedModel:
    def __init__(self, max_tasks: Optional[int] = 5) -> None:
        self._semaphore = asyncio.Semaphore(max_tasks)
        self._batch_size = int(os.environ.get("EMBED_BATCH_SIZE"))
        self._model: "PreTrainedModel" = AutoModel.from_pretrained(
            pretrained_model_name_or_path=os.environ.get("EMBED_MODEL")
        ).cuda()
        self._model.eval()
        self._tokenizer: "PreTrainedTokenizerBase" = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=os.environ.get("EMBED_MODEL")
        )
        self._tokenizer.padding_side = "right"

    async def _run_task(self, batch_encoding: "BatchEncoding") -> List[List[float]]:
        async with self._semaphore:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, _get_embeddings, self._model, batch_encoding)

    async def __call__(self, texts: List[str]) -> List[List[float]]:
        results = []
        for i in range(0, len(texts), self._batch_size):
            batch_encoding = self._tokenizer(
                texts[i : i + self._batch_size], padding=True, truncation=True, return_tensors="pt"
            )
            embeddings = await self._run_task(batch_encoding)
            results.extend(embeddings)

        return results

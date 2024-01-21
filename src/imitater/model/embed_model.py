import asyncio
from typing import TYPE_CHECKING, List, Optional

import torch
from transformers import AutoModel, AutoTokenizer


if TYPE_CHECKING:
    from transformers import BatchEncoding, PreTrainedModel

    from ..config import Config


@torch.inference_mode()
def _get_embeddings(model: "PreTrainedModel", batch_encoding: "BatchEncoding") -> List[List[float]]:
    output = model(**batch_encoding.to(model.device))
    embeddings = output[0][:, 0]
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1).tolist()
    return embeddings


class EmbedModel:
    def __init__(self, config: "Config", max_tasks: Optional[int] = 5) -> None:
        self._semaphore = asyncio.Semaphore(max_tasks)
        self._batch_size = config.embed_batch_size
        self._model: "PreTrainedModel" = AutoModel.from_pretrained(
            config.embed_model_path,
            device_map={"": config.embed_model_device[0]},
            torch_dtype=torch.float16,
        )
        self._model.eval()
        self._tokenizer = AutoTokenizer.from_pretrained(config.embed_model_path)
        self._tokenizer.padding_side = "right"

    async def _run_task(self, batch_encoding: "BatchEncoding") -> List[List[float]]:
        async with self._semaphore:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, _get_embeddings, self._model, batch_encoding)

    async def embed(self, texts: List[str]) -> List[List[float]]:
        results = []
        for i in range(0, len(texts), self._batch_size):
            batch_encoding = self._tokenizer(
                texts[i : i + self._batch_size], padding=True, truncation=True, return_tensors="pt"
            )
            embeddings = await self._run_task(batch_encoding)
            results.extend(embeddings)

        return results

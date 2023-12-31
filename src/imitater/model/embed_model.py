import os
import torch
from typing import TYPE_CHECKING, List
from transformers import AutoModel, AutoTokenizer

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


class EmbedModel:

    def __init__(self) -> None:
        self._batch_size = int(os.environ.get("EMBED_BATCH_SIZE"))
        self._model: "PreTrainedModel" = AutoModel.from_pretrained(
            pretrained_model_name_or_path=os.environ.get("EMBED_MODEL")
        ).cuda()
        self._model.eval()
        self._tokenizer: "PreTrainedTokenizerBase" = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=os.environ.get("EMBED_MODEL")
        )
        self._tokenizer.padding_side = "right"

    @torch.inference_mode()
    async def __call__(self, texts: List[str]) -> List[List[float]]:
        results = []
        for i in range(0, len(texts), self._batch_size):
            batch_encoding = self._tokenizer(
                texts[i : i + self._batch_size],
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self._model.device)

            model_output = self._model(**batch_encoding)
            embeddings = model_output[0][:, 0]
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1).tolist()
            results.extend(embeddings)

        return results

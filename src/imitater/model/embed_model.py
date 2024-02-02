from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, List

from infinity_emb import AsyncEmbeddingEngine
from typing_extensions import Self


if TYPE_CHECKING:
    from argparse import ArgumentParser, Namespace

    from numpy import float32
    from numpy.typing import NDArray


@dataclass
class EmbedConfig:
    name: str
    path: str
    device: List[int]
    batch_size: int
    port: int

    @staticmethod
    def add_cli_args(parser: "ArgumentParser") -> None:
        parser.add_argument("--name", type=str)
        parser.add_argument("--path", type=str)
        parser.add_argument("--device", type=int, nargs="+")
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--port", type=int)

    @classmethod
    def from_cli_args(cls, args: "Namespace") -> Self:
        attrs = [attr.name for attr in fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})


class EmbedModel:
    def __init__(self, config: "EmbedConfig") -> None:
        self.config = config
        self.name = config.name
        if len(config.device) != 1:
            raise ValueError("Embedding model only accepts one device.")

        self._engine = AsyncEmbeddingEngine(
            model_name_or_path=config.path,
            batch_size=config.batch_size,
            engine="torch",
            device="cuda",
        )

    async def startup(self) -> None:
        await self._engine.astart()

    async def shutdown(self) -> None:
        await self._engine.astop()

    async def embed(self, texts: List[str]) -> List["NDArray[float32]"]:
        embeddings, _ = await self._engine.embed(texts)
        return embeddings

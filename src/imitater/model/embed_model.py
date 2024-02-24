from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, List, Tuple

from infinity_emb import AsyncEmbeddingEngine
from typing_extensions import Self

from ..utils.modelscope import try_download_model_from_ms


if TYPE_CHECKING:
    from argparse import ArgumentParser, Namespace

    from numpy import float32
    from numpy.typing import NDArray


@dataclass
class EmbedConfig:
    r"""
    Creates configuration for an embedding model.

    Methods:
        add_cli_args: adds arguments to a argument parser.
        from_cli_args: builds configuration based on the command line arguments.
    """

    name: str
    path: str
    device: List[int]
    port: int
    batch_size: int

    @staticmethod
    def add_cli_args(parser: "ArgumentParser") -> None:
        parser.add_argument("--name", type=str, required=True)
        parser.add_argument("--path", type=str, required=True)
        parser.add_argument("--device", type=int, nargs="+", required=True)
        parser.add_argument("--port", type=int, required=True)
        parser.add_argument("--batch_size", type=int, default=64)

    @classmethod
    def from_cli_args(cls, args: "Namespace") -> Self:
        attrs = [attr.name for attr in fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})


class EmbedModel:
    r"""
    Creates an embedding model for text embeddings.

    Methods:
        startup: starts the embedding engine.
        shutdown: stops the embedding engine.
        embed: calculates text embeddings.
    """

    def __init__(self, config: "EmbedConfig") -> None:
        config.path = try_download_model_from_ms(config.path)
        self.config = config
        self.name = config.name
        self._init_infinity_engine()

    def _init_infinity_engine(self) -> None:
        if len(self.config.device) != 1:
            raise ValueError("Embedding model only accepts one device.")

        self._engine = AsyncEmbeddingEngine(
            model_name_or_path=self.config.path,
            batch_size=self.config.batch_size,
            device="cuda",
        )

    async def startup(self) -> None:
        r"""
        Starts the embedding engine.
        """
        await self._engine.astart()

    async def shutdown(self) -> None:
        r"""
        Stops the embedding engine.
        """
        await self._engine.astop()

    async def embed(self, texts: List[str]) -> Tuple[List["NDArray[float32]"], int]:
        r"""
        Calculates the text embeddings.

        Args:
            texts: the batched text input.

        Returns:
            embeddings: the batched embeddings.
            usage: the number of input tokens.
        """
        return await self._engine.embed(texts)

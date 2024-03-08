from typing import Any


# 重写__getattr__包加载器属性检查函数，按需加载
def _import_chatmodel() -> Any:
    from imitater.model.chat_model import ChatModel
    return ChatModel


def _import_chatconfig() -> Any:
    from imitater.model.chat_model import ChatConfig
    return ChatConfig


def _import_embedmodel() -> Any:
    from imitater.model.embed_model import EmbedModel
    return EmbedModel


def _import_embedconfig() -> Any:
    from imitater.model.embed_model import EmbedConfig
    return EmbedConfig


def _import_vinochatmodel() -> Any:
    from imitater.model.vino_model import VinoChatModel
    return VinoChatModel


def _import_vinochatconfig() -> Any:
    from imitater.model.vino_model import VinoChatConfig
    return VinoChatConfig


def __getattr__(name: str) -> Any:
    if name == "ChatModel":
        return _import_chatmodel()
    elif name == "ChatConfig":
        return _import_chatconfig()
    elif name == "EmbedModel":
        return _import_embedmodel()
    elif name == "EmbedConfig":
        return _import_embedconfig()
    elif name == "VinoChatModel":
        return _import_vinochatmodel()
    elif name == "VinoChatConfig":
        return _import_vinochatconfig()
    else:
        raise AttributeError(f"Could not find: {name}")


__all__ = [
    "ChatConfig",
    "ChatModel",
    "EmbedConfig",
    "EmbedModel",
    "VinoChatModel",
    "VinoChatConfig",
]

import os


def use_modelscope() -> bool:
    return os.environ.get("USE_MODELSCOPE_HUB", "0").lower() in ["true", "1"]


def try_download_model_from_ms(model_path: str) -> str:
    if not use_modelscope() or os.path.exists(model_path):
        return model_path

    try:
        from modelscope import snapshot_download

        return snapshot_download(model_path)
    except ImportError:
        raise ImportError("Please install modelscope via `pip install modelscope -U`")

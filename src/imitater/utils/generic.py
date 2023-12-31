import json
from typing import TYPE_CHECKING, Any, Dict

import torch


if TYPE_CHECKING:
    from pydantic import BaseModel


def dictify(data: "BaseModel") -> Dict[str, Any]:
    try:  # pydantic v2
        return data.model_dump(exclude_unset=True)
    except Exception:  # pydantic v1
        return data.dict(exclude_unset=True)


def jsonify(data: "BaseModel") -> str:
    try:  # pydantic v2
        return json.dumps(data.model_dump(exclude_unset=True), ensure_ascii=False)
    except Exception:  # pydantic v1
        return data.json(exclude_unset=True, ensure_ascii=False)


def torch_gc() -> None:
    r"""
    Collects GPU memory.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

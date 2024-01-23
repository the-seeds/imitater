from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Config:
    agent_type: str

    chat_model_path: str
    chat_model_device: List[int]
    chat_template_path: Optional[str]
    generation_config_path: Optional[str]

    embed_model_path: str
    embed_model_device: List[int]
    embed_batch_size: int

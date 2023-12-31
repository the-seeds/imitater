import time
from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel, Field


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Finish(str, Enum):
    STOP = "stop"
    LENGTH = "length"


class ModelCard(BaseModel):
    id: str
    object: Optional[str] = "model"
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    owned_by: Optional[str] = "owner"


class ModelList(BaseModel):
    object: Optional[str] = "list"
    data: Optional[List[ModelCard]] = []


class ChatMessage(BaseModel):
    role: Role
    content: str


class DeltaMessage(BaseModel):
    role: Optional[Role] = None
    content: Optional[str] = None


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: Optional[int] = None
    total_tokens: int


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = 1
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Finish


class ChatCompletionStreamResponseChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Finish] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: Optional[str] = "chat.completion"
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: Optional[str] = "chat.completion.chunk"
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionStreamResponseChoice]


class EmbeddingsRequest(BaseModel):
    input: Union[str, List[str]]
    model: str
    encoding_format: Optional[str] = "float"


class Embeddings(BaseModel):
    object: Optional[str] = "embedding"
    embedding: List[float]
    index: int


class EmbeddingsResponse(BaseModel):
    object: Optional[str] = "list"
    data: List[Embeddings]
    model: str
    usage: UsageInfo

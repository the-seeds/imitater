from typing import Any, Dict, Optional

import torch.nn as nn
from vllm.model_executor.layers.attention import PagedAttention
from vllm.model_executor.layers.linear import LinearMethodBase, QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.llama import LlamaAttention
from vllm.model_executor.parallel_utils.parallel_state import get_tensor_model_parallel_world_size


def __init__(
    self: "LlamaAttention",
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    rope_theta: Optional[float] = 10000,
    rope_scaling: Optional[Dict[str, Any]] = None,
    max_position_embeddings: int = 8192,
    linear_method: Optional[LinearMethodBase] = None,
) -> None:
    nn.Module.__init__(self)
    self.hidden_size = hidden_size
    tp_size = get_tensor_model_parallel_world_size()
    self.total_num_heads = num_heads
    assert self.total_num_heads % tp_size == 0
    self.num_heads = self.total_num_heads // tp_size
    self.total_num_kv_heads = num_kv_heads

    if self.total_num_kv_heads >= tp_size:
        # Number of KV heads is greater than TP size, so we partition
        # the KV heads across multiple tensor parallel GPUs.
        assert self.total_num_kv_heads % tp_size == 0
    else:
        # Number of KV heads is less than TP size, so we replicate
        # the KV heads across multiple tensor parallel GPUs.
        assert tp_size % self.total_num_kv_heads == 0

    self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
    self.head_dim = hidden_size // self.total_num_heads
    self.q_size = self.num_heads * self.head_dim
    self.kv_size = self.num_kv_heads * self.head_dim
    self.scaling = self.head_dim**-0.5
    self.rope_theta = rope_theta
    self.max_position_embeddings = max_position_embeddings

    self.qkv_proj = QKVParallelLinear(
        hidden_size,
        self.head_dim,
        self.total_num_heads,
        self.total_num_kv_heads,
        bias=True,
        linear_method=linear_method,
    )
    self.o_proj = RowParallelLinear(
        self.total_num_heads * self.head_dim, hidden_size, bias=True, linear_method=linear_method
    )
    self.rotary_emb = get_rope(
        self.head_dim,
        rotary_dim=self.head_dim,
        max_position=max_position_embeddings,
        base=rope_theta,
        rope_scaling=rope_scaling,
    )
    self.attn = PagedAttention(self.num_heads, self.head_dim, self.scaling, num_kv_heads=self.num_kv_heads)


def llama_attn_bias_monkey_patch():
    LlamaAttention.__init__ = __init__

import inspect
from dataclasses import dataclass
from typing import Tuple

import mlx.core as mx
from mlx import nn


def create_additive_causal_mask(N: int, offset: int = 0):
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    mask = linds[:, None] < rinds[None]
    return mask * -1e9


class KVCache:

    def __init__(self, head_dim, n_kv_heads):
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.keys = None
        self.values = None
        self.offset = 0
        self.step = 256

    def update_and_fetch(self, keys, values):
        prev = self.offset
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            shape = (1, self.n_kv_heads, n_steps * self.step, self.head_dim)
            new_k = mx.zeros(shape, keys.dtype)
            new_v = mx.zeros(shape, values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self.offset += keys.shape[2]
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]


class MemoryCache:
    def __init__(
            self,
            batch: int,
            seq_len: int,
            head_dim: int,
        ) -> None:
        # memory: (batch, seq_len, head_dim, head_dim)
        # norm_term: (batch, seq_len, head_dim, 1)

        self.memory = mx.zeros((batch, seq_len, head_dim, head_dim))
        self.norm_term = mx.ones((batch, seq_len, head_dim, 1))
    
    def update_and_fetch(
            self,
            keys: mx.array,
            values: mx.array,
        ):
        # keys: (Batch, Len = 1, NKVHead, KeyDim)
        # values: (Batch, Len = 1, NKVHead, ValueDim)
        # memory: (Batch, Len = 1, KeyDim, ValueDim)
        # norm_term: (Batch, Len = 1, KeyDim, 1)

        keys = nn.elu(keys) + 1

        matmul_memory = mx.matmul(keys, self.memory)
        matmul_norm_term = mx.matmul(keys, self.norm_term)
        matmul_memory = matmul_memory / matmul_norm_term
        values = values - matmul_memory

        self.memory = self.memory + mx.matmul(keys.transpose(0, 1, 3, 2), values)
        self.norm_term = self.norm_term + keys.sum(axis=2, keepdims=True).transpose(0, 1, 3, 2)


@dataclass
class BaseModelArgs:
    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )

from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path
import json
import logging
import glob

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, KVCache, MemoryCache


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    head_dim: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    rope_theta: float = 10000
    rope_traditional: bool = False


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.head_dim = head_dim = args.head_dim

        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.rope = nn.RoPE(
            head_dim,
            traditional=args.rope_traditional,
            base=args.rope_theta,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class InfiniAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.head_dim = head_dim = args.head_dim
        self.repeats = n_heads // n_kv_heads

        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.rope = nn.RoPE(
            head_dim,
            traditional=args.rope_traditional,
            base=args.rope_theta,
        )

        self.gate = mx.zeros((1, 1, self.n_heads, 1))

    def __call__(
            self,
            x: mx.array,
            cache: Optional[Tuple[mx.array, mx.array]] = None,
            mask: Optional[mx.array] = None,
            is_training: bool = True,
        ) -> Tuple[mx.array, mx.array, mx.array]:
        B, L, _ = x.shape
        # print(x.shape)

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1)
        keys = keys.reshape(B, L, self.n_kv_heads, -1)
        values = values.reshape(B, L, self.n_kv_heads, -1)

        if cache is not None:
            memory, norm_term = cache
            # retrieve memory
            def retrieve_memory(queries, memory, norm_term):
                # queries: (Batch, Len = 1, NHead, KeyDim) KeyDim == head_dim
                # memory: (Batch, Len = 1, KeyDim, ValueDim) ValueDim == head_dim
                # norm_term: (Batch, Len = 1, KeyDim, 1)
                queries = nn.elu(queries) + 1

                A_mem = mx.matmul(queries, memory)
                matmul_norm_term = mx.matmul(queries, norm_term)

                # A_mem: (Batch, Len = 1, NHead, ValueDim) == queries.shape
                return A_mem / matmul_norm_term

            #update memory
            def update_memory_at_train(keys, values, memory, norm_term):
                L = keys.shape[1]
                keys = nn.elu(keys) + 1

                matmul_memory = mx.matmul(keys, memory)
                matmul_norm_term = mx.matmul(keys, norm_term)
                matmul_memory = matmul_memory / matmul_norm_term
                values = values - matmul_memory

                for i in range(1, L):
                    memory[:, i] = memory[:, i-1] + mx.matmul(keys[:, i].transpose(0, 2, 1), values[:, i])
                    norm_term[:, i] = norm_term[:, i-1] + keys[:, i].sum(axis=1, keepdims=True).transpose(0, 2, 1)
                return memory, norm_term
            
            def update_memory(keys, values, memory, norm_term):
                keys = nn.elu(keys) + 1

                matmul_memory = mx.matmul(keys, memory)
                matmul_norm_term = mx.matmul(keys, norm_term)
                matmul_memory = matmul_memory / matmul_norm_term
                values = values - matmul_memory
                
                memory = memory + mx.matmul(keys.transpose(0, 1, 3, 2), values)
                norm_term = norm_term + keys.sum(axis=2, keepdims=True).transpose(0, 1, 3, 2)
                return memory, norm_term
            
            if is_training:
                memory, norm_term = update_memory_at_train(keys, values, memory, norm_term)
                A_mem = retrieve_memory(queries, memory, norm_term)
                cache = (memory, norm_term)
            else:
                A_mem = retrieve_memory(queries, memory, norm_term)
                cache = update_memory(keys, values, memory, norm_term)

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        queries = self.rope(queries)
        keys = self.rope(keys)

        #(B, L, N, D)
        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, mask=mask, scale=self.scale
        ).transpose(0, 2, 1, 3)

        # print(output.shape, A_mem.shape, self.gate.shape)
        if cache is not None:
            sig_gate = self.gate
            sig_gate = nn.sigmoid(sig_gate)
            output = sig_gate * A_mem + (1 - sig_gate) * output

        output = output.reshape(B, L, -1)
        return self.o_proj(output), cache


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.gelu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.self_attn = Attention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out
    

class InfiniTransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.self_attn = InfiniAttention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
        is_training: bool = True,
    ) -> mx.array:
        r, cache = self.self_attn(self.input_layernorm(x), cache, mask, is_training=is_training)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out, cache


class GemmaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        h = self.embed_tokens(inputs)
        h = h * (self.args.hidden_size**0.5)

        mask = None
        if h.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
            mask = mask.astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        return self.norm(h)
    

class InfiniGemmaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            InfiniTransformerBlock(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        is_training: bool = True,
    ):
        h = self.embed_tokens(inputs)
        h = h * (self.args.hidden_size**0.5)

        mask = None
        if h.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
            mask = mask.astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        for i in range(self.num_hidden_layers):
            h, cache[i] = self.layers[i](h, mask, cache[i], is_training=is_training)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model_type = args.model_type
        self.model = GemmaModel(args)
        self.args = args

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        out = self.model(inputs, cache)
        out = self.model.embed_tokens.as_linear(out)
        return out

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.head_dim

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads
    
    @staticmethod
    def from_pretrain(path: str):
        model_path = Path(path)
        if not model_path.exists():
            raise Exception(
                f"Model not found for path or HF repo: {path}.\n"
                "Please make sure you specified the local path correctly."
            ) from None
        
        def load_config(model_path: Path) -> dict:
            try:
                with open(model_path / "config.json", "r") as f:
                    config = json.load(f)
            except FileNotFoundError:
                logging.error(f"Config file not found in {model_path}")
                raise
            return config
        config = load_config(model_path=model_path)
        weight_files = glob.glob(str(model_path / "model*.safetensors"))

        if not weight_files:
            # Try weight for back-compat
            weight_files = glob.glob(str(model_path / "weight*.safetensors"))

        if not weight_files:
            logging.error(f"No safetensors found in {model_path}")
            raise FileNotFoundError(f"No safetensors found in {model_path}")

        weights = {}
        for wf in weight_files:
            weights.update(mx.load(wf))

        model_args = ModelArgs.from_dict(config)
        model = Model(model_args)

        if hasattr(model, "sanitize"):
            weights = model.sanitize(weights)

        if (quantization := config.get("quantization", None)) is not None:
            # Handle legacy models which may not have everything quantized
            def class_predicate(p, m):
                if not hasattr(m, "to_quantized"):
                    return False
                return f"{p}.scales" in weights

            nn.quantize(
                model,
                **quantization,
                class_predicate=class_predicate,
            )

        model.load_weights(list(weights.items()))

        mx.eval(model.parameters())

        model.eval()
        return model


class InfiniModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model_type = args.model_type
        self.model = InfiniGemmaModel(args)
        self.args = args

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        is_training: bool = True,
    ):
        out = self.model(inputs, cache, is_training=is_training)
        out = self.model.embed_tokens.as_linear(out)
        return out

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.head_dim

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads
    
    @staticmethod
    def from_pretrain(path: str):
        model_path = Path(path)
        if not model_path.exists():
            raise Exception(
                f"Model not found for path or HF repo: {path}.\n"
                "Please make sure you specified the local path correctly."
            ) from None
        
        def load_config(model_path: Path) -> dict:
            try:
                with open(model_path / "config.json", "r") as f:
                    config = json.load(f)
            except FileNotFoundError:
                logging.error(f"Config file not found in {model_path}")
                raise
            return config
        config = load_config(model_path=model_path)
        weight_files = glob.glob(str(model_path / "model*.safetensors"))

        if not weight_files:
            # Try weight for back-compat
            weight_files = glob.glob(str(model_path / "weight*.safetensors"))

        if not weight_files:
            logging.error(f"No safetensors found in {model_path}")
            raise FileNotFoundError(f"No safetensors found in {model_path}")

        weights = {}
        for wf in weight_files:
            weights.update(mx.load(wf))

        model_args = ModelArgs.from_dict(config)
        model = InfiniModel(model_args)

        if hasattr(model, "sanitize"):
            weights = model.sanitize(weights)

        if (quantization := config.get("quantization", None)) is not None:
            # Handle legacy models which may not have everything quantized
            def class_predicate(p, m):
                if not hasattr(m, "to_quantized"):
                    return False
                return f"{p}.scales" in weights

            nn.quantize(
                model,
                **quantization,
                class_predicate=class_predicate,
            )
        
        has_add_gate_weights = False
        GATE_INIT_VALUE = 0
        for i in range(model_args.num_hidden_layers):
            if f"model.layers.{i}.self_attn.gate" not in weights:
                has_add_gate_weights = True
                weights[f"model.layers.{i}.self_attn.gate"] = mx.ones((1, 1, model_args.num_attention_heads, 1), dtype=mx.float32) * GATE_INIT_VALUE
        if has_add_gate_weights:
            logging.warning(f"gate weights not found in the model weights. Initializing with {GATE_INIT_VALUE}.")

        model.load_weights(list(weights.items()))

        mx.eval(model.parameters())

        model.eval()
        return model
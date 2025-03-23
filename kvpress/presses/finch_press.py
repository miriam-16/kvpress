# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import math
from contextlib import contextmanager
from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel, QuantizedCache
from transformers.models.llama.modeling_llama import repeat_kv, rotate_half

from kvpress.presses.scorer_press import ScorerPress


@dataclass
class FinchPress(ScorerPress):
    """
    Finch use the attention of the concatenated question of the user to estimate the importance
    of the context elements.
    """

    compression_ratio: float = 0.0
    condition_len: int = None
    split_size: int = 3

    @staticmethod
    def compute_normalization_factors(attention_mask, attn_weights, tol=1e-8):
        binary_mask = (torch.abs(attention_mask.to(torch.float32)) < tol).to(torch.float32)
        non_zero_counts = binary_mask.sum(dim=3, keepdim=True)
        non_zero_counts = torch.clamp_min(non_zero_counts, 1.0).to(attn_weights.dtype)

        return non_zero_counts

    def compute_finch_attention(self, module, hidden_states, keys, condition_len, position_embeddings):
        """Compute the last condition_len queries (question) and associated attention weights for the first q_len - condition_len keys (context)."""
        bsz, q_len, _ = hidden_states.shape
        num_heads = module.config.num_attention_heads
        head_dim = module.head_dim
        num_key_value_groups = num_heads // module.config.num_key_value_heads

        # print("q_len is ", q_len)
        # print("condition_len is ", condition_len)
        # print("num_heads is ", num_heads)
        # print("head_dim is ", head_dim)

        # Get question queries
        if hasattr(module, "q_proj"):
            query_states = module.q_proj(hidden_states[:, -condition_len:])
        elif hasattr(module, "qkv_proj"):
            qkv = module.qkv_proj(hidden_states[:, -condition_len:])
            query_states = qkv[..., : num_heads * head_dim]
        else:
            raise NotImplementedError(f"Finch not yet implemented for {module.__class__}.")
        # print("query_states shape is ", query_states.shape)
        query_states = query_states.view(bsz, condition_len, num_heads, head_dim).transpose(1, 2)

        # Apply RoPE, considering queries only
        cos, sin = position_embeddings
        cos, sin = cos[:, -condition_len:], sin[:, -condition_len:]
        query_states = (query_states * cos.unsqueeze(1)) + (rotate_half(query_states) * sin.unsqueeze(1))

        # Compute attention for context tokens
        key_states = repeat_kv(keys, num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
        attention_mask = torch.ones_like(attn_weights) * float("-inf")
        attention_mask = torch.triu(attention_mask, diagonal=key_states.shape[-2] - condition_len + 1)
        attn_weights += attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # Finch incorporates a normalization step, ensuring that each token’s relevance is equally evaluated.
        non_zero_counts = self.compute_normalization_factors(attention_mask, attn_weights)
        attn_weights = attn_weights * non_zero_counts

        attn_weights = attn_weights[..., :-condition_len]

        return attn_weights

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:

        bsz, num_key_value_heads, q_len, _ = keys.shape
        num_key_value_groups = module.config.num_attention_heads // num_key_value_heads

        if attentions is not None:

            attn_weights = attentions[..., -self.condition_len :, : -self.condition_len]
            non_zero_counts = self.compute_normalization_factors(
                kwargs["attention_mask"][..., -self.condition_len :, :q_len], attn_weights
            )
            if module.layer_idx == 0:
                print("attentions are: ", attentions.shape)
                print("attn_weights are: ", attn_weights.shape)
            attn_weights = attn_weights * non_zero_counts
        else:
            attn_weights = self.compute_finch_attention(
                module, hidden_states, keys, self.condition_len, kwargs["position_embeddings"]
            )
        scores = attn_weights.sum(dim=-2)
        if module.layer_idx == 0:
            print("scores sum over question: ", scores.shape)
        scores = scores.view(bsz, num_key_value_heads, num_key_value_groups, q_len - self.condition_len)
        if module.layer_idx == 0:
            print("scores reshaped: ", scores.shape)
        scores = scores.sum(dim=2)
        if module.layer_idx == 0:
            print("scores sum over heads: ", scores.shape)
        # Add back the observation window. Use max score to make sure the window is not pruned.
        scores = F.pad(scores, (0, self.condition_len), value=scores.max().item())
        if module.layer_idx == 0:
            print("scores padded: ", scores.shape)
        return scores

    @staticmethod
    def _rerotate_cos_sin(x, inv_freq, important_pos_batch):
        B, H, L = important_pos_batch.shape
        device = important_pos_batch.device
        device_type = x.device.type
        dtype = x.dtype
        idx = torch.arange(0, L, device=device)
        idx = idx.unsqueeze(0)
        inv_freq = inv_freq[None, None, :, None].float().expand(B, H, -1, 1)  # (B, H, M, 1)
        idx = idx[:, None, :].float().expand(B, H, L)  # (B, H, L)
        delta_pos = idx - important_pos_batch
        delta_pos = delta_pos.unsqueeze(2)  # (B, H, 1, L)

        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"

        with torch.autocast(device_type=device_type, enabled=False):
            freqs = delta_pos.float() * inv_freq.float()
            freqs = freqs.transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos().contiguous()
            sin = emb.sin().contiguous()
        return cos.to(dtype=dtype), sin.to(dtype=dtype)


    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.compression_ratio == 0:
            return keys, values

        # Compute scores from base press
        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)
        # Get indices of KV pairs with the lowest scores
        q_len = hidden_states.shape[1]

        context_length = kwargs["context_length"]
        #n_kept = int(q_len * (1 - self.compression_ratio)) + (scores.shape[-1] - q_len)
        n_kept = int(q_len * (1 - self.compression_ratio)) + (scores.shape[-1] - q_len) if kwargs["split_idx"]!=self.split_size-1 else int(context_length * (1 - self.compression_ratio) + self.condition_len)

        if module.layer_idx == 0:
            print("q_len is ", q_len,"n_kept is " , n_kept, "(cache + compression(chunk + question)")
        indices = scores.topk(n_kept, dim=-1).indices
        new_cos, new_sin = self._rerotate_cos_sin(keys, module.rotary_emb.inv_freq, indices)
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
        keys = keys.gather(2, indices).contiguous()
        keys = (keys * new_cos) + (rotate_half(keys) * new_sin)
        values = values.gather(2, indices).contiguous()
        return keys, values

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        """
        Default forward hook called after the forward pass of an attention layer.
        The hook calls the compress method to compress the KV cache while ensuring:
            - compression is only applied only during the pre-filling phase
            - KV cache quantization is handled correctly

        Parameters
        ----------
        module :
            Transformer attention layer.
        input :
            Input to the hook. This is the input to the forward pass of the layer.
        kwargs :
            Keyword arguments, as given to the forward pass of the layer.
        output :
            Output of the hook. This is the original output of the forward pass of the layer.

        Returns
        -------
            Modified output of the forward pass of the layer.

        """

        hidden_states = kwargs["hidden_states"]
        cache = kwargs["past_key_value"]

        if isinstance(cache, QuantizedCache):
            keys = cache._dequantize(cache._quantized_key_cache[module.layer_idx])
            values = cache._dequantize(cache._quantized_value_cache[module.layer_idx])
        else:
            keys = cache.key_cache[module.layer_idx]
            values = cache.value_cache[module.layer_idx]

        keys, values = self.compress(module, hidden_states, keys, values, output[1], kwargs)

        if isinstance(cache, QuantizedCache):
            cache._quantized_key_cache[module.layer_idx] = cache._quantize(keys, axis=cache.axis_key)
            cache._quantized_value_cache[module.layer_idx] = cache._quantize(values, axis=cache.axis_value)
            cache.key_cache[module.layer_idx] = torch.zeros(0, dtype=keys.dtype, device=keys.device)
            cache.value_cache[module.layer_idx] = torch.zeros(0, dtype=keys.dtype, device=keys.device)
            cache._seen_tokens = keys.shape[2]
        else:
            cache.key_cache[module.layer_idx] = keys
            cache.value_cache[module.layer_idx] = values

        return output

    @contextmanager
    def __call__(self, model: PreTrainedModel):
        hooks = []
        try:
            for layer in model.model.layers:
                layer.self_attn.rotary_emb = model.model.rotary_emb
                hooks.append(layer.self_attn.register_forward_hook(self.forward_hook, with_kwargs=True))
            # Save the original forward method
            original_forward = model.forward

            def chunked_forward(*args, **kwargs):
                input_ids = kwargs.get("input_ids", None)
                attention_mask = kwargs.get("attention_mask", None)

                # Split input_ids into context and question tokens.
                context_ids = input_ids[:, : -self.condition_len]
                question_ids = input_ids[:, -self.condition_len :]

                if attention_mask is not None:
                    context_attention_mask = attention_mask[:, : -self.condition_len]
                    question_attention_mask = attention_mask[:, -self.condition_len :]

                # Calculate the total number of context tokens.
                context_length = context_ids.shape[1]
                kwargs["context_length"] = context_length

                # Determine the chunk size so that we split context_ids into exactly split_size chunks.
                chunk_size = context_length // self.split_size
                last_output = None

                print("Total context length is:", context_length)
                print("Chunk size is:", chunk_size)
                print("Question length is:", self.condition_len)
                print("Number of chunks:", self.split_size)

                for i in range(self.split_size):
                    kwargs["split_idx"]=i
                    start = i * chunk_size
                    # For the last chunk, include any remaining tokens.
                    end = start + chunk_size if i < self.split_size - 1 else context_length


                    # Get the current chunk from context_ids and combine with the question tokens.
                    context_chunk = context_ids[:, start:end]

                    kwargs["input_ids"] = torch.cat((context_chunk, question_ids), dim=1)
                    print(f"Processing chunk {i} of len: ", kwargs["input_ids"].shape[1], "(context chunk + question)")

                    if attention_mask is not None:
                        context_attention_mask_chunk = context_attention_mask[:, start:end]
                        kwargs["attention_mask"] = torch.cat(
                            (context_attention_mask_chunk, question_attention_mask), dim=1
                        )
                    last_output = original_forward(*args, **kwargs)

                    # Only adjust the past key/values caches if it's not the last iteration.
                    if i < self.split_size - 1:
                        for layer_idx, _ in enumerate(model.model.layers):
                            # Adjust the past key/values caches to remove the question tokens for the next iteration
                            kwargs["past_key_values"].key_cache[layer_idx] = (
                                kwargs["past_key_values"]
                                .key_cache[layer_idx][:, :, : -self.condition_len, :]
                                .contiguous()
                            )
                            kwargs["past_key_values"].value_cache[layer_idx] = (
                                kwargs["past_key_values"]
                                .value_cache[layer_idx][:, :, : -self.condition_len, :]
                                .contiguous()
                            )

                return last_output

            # Override the model's forward with the chunked version
            model.forward = chunked_forward
            yield
        finally:
            # Remove all hooks and restore the original forward method
            for hook in hooks:
                hook.remove()
            model.forward = original_forward

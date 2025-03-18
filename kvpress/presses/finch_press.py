# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import math
from dataclasses import dataclass
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.nn import functional as F
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

    @staticmethod
    def compute_normalization_factors(attention_mask, attn_weights, tol=1e-8):
        binary_mask = (torch.abs(attention_mask.to(torch.float32)) < tol).to(torch.float32)
        non_zero_counts = binary_mask.sum(dim=3, keepdim=True)
        non_zero_counts = torch.clamp_min(non_zero_counts, 1.0).to(attn_weights.dtype)
        return non_zero_counts

    @staticmethod
    def compute_finch_attention(module, hidden_states, keys, condition_len, position_embeddings):

        """Compute the last condition_len queries (question) and associated attention weights for the first q_len - condition_len keys (context).
        """
        bsz, q_len, _ = hidden_states.shape
        num_heads = module.config.num_attention_heads
        head_dim = module.head_dim
        num_key_value_groups = num_heads // module.config.num_key_value_heads

        # Get question queries
        if hasattr(module, "q_proj"):
            query_states = module.q_proj(hidden_states[:, -condition_len:])
        elif hasattr(module, "qkv_proj"):
            qkv = module.qkv_proj(hidden_states[:, -condition_len:])
            query_states = qkv[..., : num_heads * head_dim]
        else:
            raise NotImplementedError(f"Finch not yet implemented for {module.__class__}.")

        query_states = query_states.view(bsz, condition_len, num_heads, head_dim).transpose(1, 2)

        # Apply RoPE, considering queries only
        cos, sin = position_embeddings
        cos, sin = cos[:, -condition_len:], sin[:, -condition_len:]
        query_states = (query_states * cos.unsqueeze(1)) + (rotate_half(query_states) * sin.unsqueeze(1))

        # Compute attention for context tokens
        key_states = repeat_kv(keys, num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
        attention_mask = torch.ones_like(attn_weights) * float("-inf")
        attention_mask = torch.triu(attention_mask, diagonal=q_len - condition_len+1)
        attn_weights += attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # Finch incorporates a normalization step, ensuring that each token’s relevance is equally evaluated.
        non_zero_counts = compute_normalization_factors(attention_mask, attn_weights)
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
            non_zero_counts = self.compute_normalization_factors(kwargs["attention_mask"][..., -self.condition_len :, :q_len], attn_weights)
            attn_weights = attn_weights * non_zero_counts
        else:
            attn_weights = self.compute_finch_attention(
                module, hidden_states, keys, self.condition_len, kwargs["position_embeddings"]
            )
        scores = attn_weights.sum(dim=-2)
        scores = scores.view(bsz, num_key_value_heads, num_key_value_groups, q_len - self.condition_len)
        scores = scores.sum(dim=2)
        # Add back the observation window. Use max score to make sure the window is not pruned.
        scores = F.pad(scores, (0, self.condition_len), value=scores.max().item())

        return scores

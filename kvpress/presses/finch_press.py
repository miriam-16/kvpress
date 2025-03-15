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
    SnapKV (https://arxiv.org/abs/2404.14469) use the attention of the latest window_size tokens to estimate the
    importance of the previous KV pairs. We use the default settings from:
    https://github.com/FasterDecoding/SnapKV/blob/main/snapkv/monkeypatch/snapkv_utils.py#L24
    """
<<<<<<< HEAD

    #window_size: int = 64
    #kernel_size: int = 5
    #separator_index: int = 35 # for starting, assume we have it fixed
=======
>>>>>>> e68f742 (fix question len, adjusted the question ids for generation.)
    compression_ratio: float = 0.0
    condition_len: int = None

    @staticmethod
<<<<<<< HEAD
    def compute_finch_attention(module, hidden_states, keys, condition_len, position_embeddings):

        """ apply Finch"""
=======
    def compute_finch_attention(module, hidden_states, keys, separator_index, position_embeddings):
        """apply Finch"""
>>>>>>> e68f742 (fix question len, adjusted the question ids for generation.)

        # keep it as it is
        bsz, q_len, _ = hidden_states.shape
        num_heads = module.config.num_attention_heads
        head_dim = module.head_dim
        num_key_value_groups = num_heads // module.config.num_key_value_heads

<<<<<<< HEAD




=======
>>>>>>> e68f742 (fix question len, adjusted the question ids for generation.)
        # Get question queries
        if hasattr(module, "q_proj"):
            print("inside q_proj")
            query_states = module.q_proj(hidden_states[:, -condition_len:])
            print("query states dimension after slicing: ", query_states.shape)
        elif hasattr(module, "qkv_proj"):
<<<<<<< HEAD
            #print("inside qkv_proj")
            qkv = module.qkv_proj(hidden_states[:, -condition_len:])
=======
            # print("inside qkv_proj")
            qkv = module.qkv_proj(hidden_states[:, separator_index:])
>>>>>>> e68f742 (fix question len, adjusted the question ids for generation.)
            query_states = qkv[..., : num_heads * head_dim]
        else:
            # print("inside else")
            raise NotImplementedError(f"Finch not yet implemented for {module.__class__}.")

        # TODO: still have to modify this part!!
        # query_states = query_states.view(bsz, window_size, num_heads, head_dim).transpose(1, 2)

<<<<<<< HEAD
    #   query_dimension=hidden_states.shape[1]-condition_len #adjust for excluding the separator token
    #   query_states = query_states.view(bsz, query_dimension, num_heads, head_dim).transpose(1, 2) #1x3x14x3

        query_states = query_states.view(bsz, condition_len, num_heads, head_dim).transpose(1, 2)

=======
        query_dimension = hidden_states.shape[1] - separator_index  # adjust for excluding the separator token
        query_states = query_states.view(bsz, query_dimension, num_heads, head_dim).transpose(1, 2)  # 1x3x14x3
>>>>>>> e68f742 (fix question len, adjusted the question ids for generation.)

        # print(f"query states reshaped dimension with query dim {query_dimension}: {query_states.shape}")

        # Apply RoPE, considering queries only
        cos, sin = position_embeddings
        cos, sin = cos[:, -condition_len:], sin[:, -condition_len:]
        query_states = (query_states * cos.unsqueeze(1)) + (rotate_half(query_states) * sin.unsqueeze(1))

        # print("query states after RoPE: ", query_states.shape)
        # Compute attention for context tokens
        key_states = repeat_kv(keys, num_key_value_groups)
        # print("key states dimension: ", key_states.shape)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)

        # Apply mask to avoid attending to future tokens, is it really necessary?
        attention_mask = torch.ones_like(attn_weights) * float("-inf")
        attention_mask = torch.triu(attention_mask, diagonal=q_len - condition_len+1)
        attn_weights += attention_mask
        print("attn_weights dimension after mask: ", attn_weights.shape)
        print("attn_weights after mask: ", attn_weights)

<<<<<<< HEAD
        # debug print for attention weights matrix
        """ print_matrix = attn_weights[0,0,::].numpy()
        plt.imshow(print_matrix, cmap='viridis', aspect='auto')
        plt.show() """

        print("attn_weights dimension before softmax: ", attn_weights.shape)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        print("attn_weights dimension after softmax: ", attn_weights.shape)
        attn_weights = attn_weights[..., :-condition_len]

        print("attn_weights dreturned: ", attn_weights.shape)
=======
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # print("attn_weights dimension: ", attn_weights.shape)
        attn_weights = attn_weights[..., :separator_index]

        # print("attn_weights dimension considering only context: ", attn_weights.shape)
>>>>>>> e68f742 (fix question len, adjusted the question ids for generation.)

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

        # Debus prints
        print("key states dim: ",keys.shape)
        print("values states dim: ", values.shape)
        query_states = module.q_proj(hidden_states)
        print("query states dimension: ", query_states.shape)
<<<<<<< HEAD

        #print(attentions.shape)
        #print(num_key_value_groups)
=======
        print(attentions.shape)
        # print(num_key_value_groups)
>>>>>>> e68f742 (fix question len, adjusted the question ids for generation.)

        # assert q_len > self.window_size, "Query length should be greater than the window size"

        """torch.Size([1, 3, 44, 64])
        torch.Size([1, 3, 44, 64])
        query states dimension:  torch.Size([1, 44, 576])
        torch.Size([1, 9, 44, 44])"""

        if attentions is not None:
            print("attention not none, simply slice")
            # keep only attentions between context (first part) and question (last part)
            attn_weights = attentions[..., -self.condition_len :, : -self.condition_len]
            print(attn_weights.shape)
        else:
            print("attention  none, compute")
            attn_weights = self.compute_finch_attention(
<<<<<<< HEAD
                module, hidden_states, keys,self.condition_len, kwargs["position_embeddings"]
=======
                module, hidden_states, keys, self.separator_index, kwargs["position_embeddings"]
>>>>>>> e68f742 (fix question len, adjusted the question ids for generation.)
            )

            # 1x9x14x12

        # TODO : should we normalize as Giulio did?

        scores = attn_weights.sum(dim=-2)
<<<<<<< HEAD
        print("scores not padded dim: ", scores.shape)

        #scores = F.avg_pool1d(scores, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)
        # Average per grioup (https://github.com/FasterDecoding/SnapKV/issues/22)
=======
>>>>>>> e68f742 (fix question len, adjusted the question ids for generation.)

        scores = scores.view(bsz, num_key_value_heads, num_key_value_groups, q_len - self.condition_len)
        print("scores reshaped dim: ", scores.shape)
        scores = scores.sum(2)
        print("scores sum dim: ", scores.shape)

        # Add back the condition window. Use max score to make sure the condition is not pruned.
        scores = F.pad(scores, (0, self.condition_len - 1), value=scores.max().item())

        print("scores padded dimension: ", scores.shape)

        return scores

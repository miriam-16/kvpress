# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import math
from contextlib import contextmanager
from dataclasses import dataclass
import random

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel, QuantizedCache
from transformers.models.llama.modeling_llama import repeat_kv, rotate_half

from kvpress.presses.base_press import BasePress
from kvpress.presses.snapkv_press import SnapKVPress


@dataclass
class FinchPressTSHavgPrecise(BasePress):
    """
    Finch uses the attention information between the prompt and the document chunk to dynamically
    identify the most relevant KV pairs across different layers.
    This information then is stored in the KV cache for the processing of the next input chunk
    (https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00716/125280)

    Parameters
    ----------
    condition_len : int
        The number of tokens in the prompt.
    split_size : int
        The number of chunks to split the context into.
    """

    compression_ratio: float = 0.0
    split_size: int = 1
    normalize_scores: bool = True
    condition_len: int = None  # calculate on length of question dynamically

    def score(self, module, hidden_states, keys, values, attentions, kwargs):

        bsz, num_key_value_heads, q_len, _ = keys.shape
        num_key_value_groups = module.config.num_attention_heads // num_key_value_heads

        if attentions is not None:
            attn_weights = attentions[..., -self.condition_len:, : -self.condition_len]
        else:
            attn_weights = SnapKVPress.compute_window_attention(module, hidden_states, keys, self.condition_len, kwargs["position_embeddings"])
        if self.normalize_scores:
            non_zero_counts = torch.arange(q_len - self.condition_len, q_len)[None, None, :, None]
            non_zero_counts = non_zero_counts.to(attn_weights.device)
            attn_weights = attn_weights * non_zero_counts

        if module.layer_idx == 0:
            print("attn_weights shape: ", attn_weights.shape)
        # Average per group
        scores = attn_weights.mean(dim=-2)
        scores = scores.view(bsz, num_key_value_heads, num_key_value_groups, q_len - self.condition_len)
        if module.layer_idx == 0:
            print("scores shape before group average: ", scores.shape)
        scores = scores.mean(dim=2)
        if module.layer_idx == 0:
            print("scores shape after group average: ", scores.shape)

        # Add back the observation window. Use max score to make sure the window is not pruned.
        scores = F.pad(scores, (0, self.condition_len), value=scores.max().item())

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

        context_length = kwargs["context_length"]
        last_iteration = kwargs["split_idx"] == self.split_size - 1
        q_len = hidden_states.shape[1]

        if last_iteration:
            n_kept_context = int(context_length * (1 - self.compression_ratio))
        else:
            past_cache_len = scores.shape[-1] - q_len
            n_kept_context = int((q_len - self.condition_len) * (1 - self.compression_ratio)) + past_cache_len


        #NEW PART 
        #_______________________________________________________________#

        #compute the mean of the scores across all heads
        scores_avg = scores.mean(dim=1)  # shape: (batch_size, seq_len)

        input_ids = self.current_input_ids  # shape: (seq_len,)
        tokenizer = self.tokenizer
        tuple_end_token_id = tokenizer.convert_tokens_to_ids("<tuple_end>") #retrieve the token of the separator

        #Identify token positions for each tuple
        tuple_end_positions = (input_ids == tuple_end_token_id).nonzero(as_tuple=True)[1]


        tuple_ranges = []
        start = 0
        tuple_lengths=[]
        for end in tuple_end_positions:
            tuple_ranges.append((start, end.item() + 1))  # include <tuple_end>
            #compute the length of the tuple
            tuple_lengths.append(end.item() - start + 1)  # length of the tuple
            start = end.item() + 1


        #retrieve all tokens ordered by importance
        k = scores_avg.shape[-1] - self.condition_len
        top_indices = scores_avg[:, :-self.condition_len].topk(k, dim=-1).indices
        #top_indices = scores_avg[:, :-self.condition_len].topk(n_kept_context, dim=-1).indices  #get the top indices, REMOVED 
        
        batch_top_indices=top_indices[0]

        head_kept_token_sets = set(batch_top_indices.tolist()) #store all tokens in a set to avoid duplicates
        head_kept_tuple_indices = set()  # check the tokens already stored

        important_tokens = head_kept_token_sets
        current_num_tokens = 0

        #print("important tokens: ", important_tokens)

        tuples_inserted=set()

        for token in important_tokens:
            #print("new important token: ",token)
            i=0
            for start, end in tuple_ranges: #iterate over tuples and find the tuple the token is in 
                if token in range(start, end):
                    tuple_size = end - start
                    if current_num_tokens + tuple_size <= n_kept_context and i not in tuples_inserted:
                        head_kept_tuple_indices.update(range(start, end))
                        tuples_inserted.add(i)

                        current_num_tokens += tuple_size

                    elif i not in tuples_inserted:
                        # If adding the whole tuple would exceed the max tokens, only add the necessary tokens
                        tokens_needed = n_kept_context - current_num_tokens

                        head_kept_tuple_indices.update(range(start, start + tokens_needed))
                        tuples_inserted.add(i)
                        current_num_tokens += tokens_needed
                        break  # Move to the next important token

                    #exit the for loop, move to next token
                    break 

                i+=1
            if current_num_tokens >= n_kept_context:
                break
        
        #print("final number of tokens: ",current_num_tokens)
        #print("final_len: ",len(head_kept_tuple_indices))

        
        print("FINISHED")

        #transform it back to a list of sets
        head_kept_tuple_indices = [sorted(list(head_kept_tuple_indices)) for _ in range(3)]
        
        head_kept_tuple_indices = torch.tensor(head_kept_tuple_indices)
        indices_context = head_kept_tuple_indices.unsqueeze(0).expand(self.split_size, -1, -1)

        #_______________________________________________________________#
        #END OF NEW PART


        #OLD PART

        #indices_context = scores[:, :, : -self.condition_len].topk(n_kept_context, dim=-1).indices

        indices_condition = torch.arange(scores.shape[-1] - self.condition_len, scores.shape[-1], device=scores.device)[
            None, None, :
        ].expand(scores.shape[0], scores.shape[1], -1)
        indices_context, _ = torch.sort(indices_context, dim=-1)

        print("indices_context shape: ", indices_context.shape)
        print("indices_condition shape: ", indices_condition.shape)

        # concatenate the indices
        indices = torch.cat([indices_context, indices_condition], dim=-1)

        # rerotate the positions
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
        q_len = hidden_states.shape[1]

        # Don't compress after pre-filling
        if kwargs["cache_position"][-1] > q_len + cache.get_seq_length():
            return output

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
    def __call__(self, model: PreTrainedModel, tokenizer):
        hooks = []
        self.tokenizer=tokenizer #save it globally

        try:
            for layer in model.model.layers:
                layer.self_attn.rotary_emb = model.model.rotary_emb
                hooks.append(layer.self_attn.register_forward_hook(self.forward_hook, with_kwargs=True))
            original_forward = model.forward

            def chunked_forward(*args, **kwargs):
                args = list(args)
                kwargs["input_ids"] = kwargs.get("input_ids", args.pop(0) if args else None)
                kwargs["attention_mask"] = kwargs.get("attention_mask", args.pop(0) if args else None)
                args = tuple(args)

                input_ids = kwargs["input_ids"]
                attention_mask = kwargs.get("attention_mask")

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

                for i in range(self.split_size):
                    kwargs["split_idx"] = i
                    start = i * chunk_size
                    # For the last chunk, include any remaining tokens.
                    end = start + chunk_size if i < self.split_size - 1 else context_length

                    # Get the current chunk from context_ids and combine with the question tokens.
                    context_chunk = context_ids[:, start:end]
                    kwargs["input_ids"] = torch.cat([context_chunk, question_ids], dim=1)

                    self.current_input_ids = kwargs["input_ids"] #save them in a global variable 

                    if attention_mask is not None:
                        context_attention_mask_chunk = context_attention_mask[:, start:end]
                        kwargs["attention_mask"] = torch.cat(
                            [context_attention_mask_chunk, question_attention_mask], dim=1
                        )

                    last_output = original_forward(use_cache=True, *args, **kwargs)

                    # Only adjust the past key/values caches if it's not the last iteration.
                    if i < self.split_size - 1:
                        for layer_idx, _ in enumerate(model.model.layers):
                            # Adjust the past key/values caches to remove the question tokens for the next iteration
                            last_output.past_key_values.key_cache[layer_idx] = (
                                last_output.past_key_values
                                .key_cache[layer_idx][:, :, : -self.condition_len, :]
                                .contiguous()
                            )
                            last_output.past_key_values.value_cache[layer_idx] = (
                                last_output.past_key_values
                                .value_cache[layer_idx][:, :, : -self.condition_len, :]
                                .contiguous()
                            )
                    last_output.past_key_values._seen_tokens = last_output.past_key_values.get_seq_length()
                    kwargs["past_key_values"] = last_output.past_key_values

                return last_output

            # Override the model's forward with the chunked version
            model.forward = chunked_forward
            yield
        finally:
            # Remove all hooks and restore the original forward method
            for hook in hooks:
                hook.remove()
            model.forward = original_forward

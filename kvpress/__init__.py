# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from kvpress.attention_patch import patch_attention_functions
from kvpress.pipeline import KVPressTextGenerationPipeline
from kvpress.presses.adakv_press import AdaKVPress
from kvpress.presses.base_press import BasePress
from kvpress.presses.chunk_press import ChunkPress
from kvpress.presses.chunkkv_press import ChunkKVPress
from kvpress.presses.composed_press import ComposedPress
from kvpress.presses.criticalkv_press import CriticalAdaKVPress, CriticalKVPress
from kvpress.presses.duo_attention_press import DuoAttentionPress
from kvpress.presses.expected_attention_press import ExpectedAttentionPress
from kvpress.presses.finch_press_tuple_selection_precise import FinchPress
from kvpress.presses.key_rerotation_press import KeyRerotationPress
from kvpress.presses.knorm_press import KnormPress
from kvpress.presses.observed_attention_press import ObservedAttentionPress
from kvpress.presses.per_layer_compression_press import PerLayerCompressionPress
from kvpress.presses.qfilter_press import QFilterPress
from kvpress.presses.random_press import RandomPress
from kvpress.presses.scorer_press import ScorerPress
#from kvpress.presses.simlayerkv_press import SimLayerKVPress
from kvpress.presses.snapkv_press import SnapKVPress
from kvpress.presses.streaming_llm_press import StreamingLLMPress
from kvpress.presses.think_press import ThinKPress
from kvpress.presses.tova_press import TOVAPress
from kvpress.presses.finch_press_tuple_selection_naive import FinchPressTSNaive
from kvpress.presses.finch_press_window_tuple_selection import FinchPressWTS
from kvpress.presses.finch_press_window_column_selection import FinchPressWCS
from kvpress.presses.finch_press_tuplecolumn_selection import FinchPressTCSNaive
from kvpress.presses.finch_press_heads_average_tupleselectionnaive import FinchPressTSHavg
from kvpress.presses.finch_press_heads_average_tupleselectionwindow import FinchPressTWSHavg
from kvpress.presses.finch_press_heads_average_tupleselectionprecise import FinchPressTSHavgPrecise
from kvpress.presses.finch_press_heads_average_tuplecolumnselection import FinchPressTCSNaiveHavg


# Patch the attention functions to support head-wise compression
patch_attention_functions()

__all__ = [
    "CriticalAdaKVPress",
    "CriticalKVPress",
    "AdaKVPress",
    "BasePress",
    "ComposedPress",
    "ScorerPress",
    "ExpectedAttentionPress",
    "KnormPress",
    "ObservedAttentionPress",
    "RandomPress",
    "SimLayerKVPress",
    "SnapKVPress",
    "StreamingLLMPress",
    "ThinKPress",
    "TOVAPress",
    "KVPressTextGenerationPipeline",
    "PerLayerCompressionPress",
    "KeyRerotationPress",
    "ChunkPress",
    "DuoAttentionPress",
    "FinchPress",
    "ChunkKVPress",
    "QFilterPress",
    "FinchPressTSNaive",
    "FinchPressWTS",
    "FinchPressWCS",
    "FinchPressTCSNaive",
    "FinchPressTSHavg",
    "FinchPressTWSHavg",
    "FinchPressTSHavgPrecise",
    "FinchPressTCSNaiveHavg"
]

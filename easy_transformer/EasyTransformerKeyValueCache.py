import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Union, Tuple, List, Dict, Any, Optional
from EasyTransformer import EasyTransformer


@dataclass
class EasyTransformerKeyValueCacheEntry:
    past_keys: torch.Tensor
    past_values: torch.Tensor

    @classmethod
    def init_cache_entry(cls, model: EasyTransformer):
        return cls(
            past_keys=torch.empty(1, model.cfg.n_heads, 0, model.cfg.d_head),
            past_values=torch.empty(1, model.cfg.n_heads, 0, model.cfg.d_head),
        )

    def append(self, new_keys: torch.Tensor, new_values: torch.Tensor):
        self.past_keys = torch.cat([self.past_keys, new_keys], dim=2)
        self.past_values = torch.cat([self.past_values, new_values], dim=2)
        return self.past_keys, self.past_values


@dataclass
class EasyTransformerKeyValueCache:
    entries: List[EasyTransformerKeyValueCacheEntry]

    @classmethod
    def init_cache(cls, model: EasyTransformer):
        return cls(
            entries=[
                EasyTransformerKeyValueCacheEntry.init_cache_entry(model)
                for _ in range(model.cfg.n_layers)
            ]
        )

    def __getitem__(self, idx):
        return self.entries[idx]
# %%
from mimetypes import init
from typing import Callable, Union, List, Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
import logging

from tqdm import tqdm
import random
import time

from pathlib import Path
import pickle
import os

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio

import plotly.graph_objects as go

from torch.utils.data import DataLoader

from functools import *
import pandas as pd
import gc
import collections
import copy

# import comet_ml
import itertools

from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    PreTrainedTokenizer,
)

from easy_transformer_speedy.hook_points import HookedRootModule, HookPoint
from easy_transformer_speedy.utils import (
    lm_cross_entropy_loss,
)
from easy_transformer_speedy.EasyTransformerConfig import EasyTransformerConfig

from easy_transformer_speedy.EasyTransformerKeyValueCache import (
    EasyTransformerKeyValueCache,
    EasyTransformerKeyValueCacheEntry,
)

from .parallelism_utils import *

from easy_transformer_speedy.components import *

import easy_transformer_speedy.weight_conversion as weight_conversion


# TODO: Add Bloom, GPT-J and GPT-NeoX
"""
EleutherAI/gpt-j-6B
EleutherAI/gpt-neox-20b
bloom-350m
bloom-760m
bloom-1b3
bloom-2b5
bloom-6b3
bloom (176B parameters)
https://huggingface.co/docs/transformers/model_doc/bloom
"""

# Define network architecture


# Full transformer
class EasyTransformer(HookedRootModule):
    """
    This class implements a full Transformer using the above components, with
    HookPoints on every interesting activation. It inherits from HookedRootModule.

    It can be initialised with a model name, and then will automatically load the model weights
    for that model, loads them into this model, as well as fold in LayerNorm and center
    the weights.

    It can also be initilised with an EasyTransformerConfig or a config dictionary, which can be used to instantiate a custom model without loading pretrained weights and will instead use Pytorch's default weight initialisation.
    """

    VALID_PRETRAINED_MODEL_NAMES = weight_conversion.VALID_PRETRAINED_MODEL_NAMES
    PRETRAINED_MODEL_NAMES_DICT = weight_conversion.PRETRAINED_MODEL_NAMES_DICT
    STANFORD_CRFM_CHECKPOINTS = weight_conversion.STANFORD_CRFM_CHECKPOINTS

    def __init__(
        self,
        model_name,
        cfg=None,
        use_attn_result=False,
        model=None,
        keep_original_model=False,
        checkpoint=None,
        fold_ln=True,
    ):
        """
        model_name (str: The name of the model to load, via HuggingFace. If
            "custom", then cfg must be provided.
        cfg (EasyTransformerConfig, *optional*): The config to use for the
            model. If not provided, a model name must be passed via model_name.
        tokenizer (*optional): The tokenizer to use for the model. If not
            provided, initialized to None, though the user must initialize one
            before passing strings to the model.
        use_attn_result (bool): Says whether to explicitly calculate the amount
            each head adds to the residual stream (with a hook) and THEN add it
            up, vs just calculating the sum. This can be very memory intensive
            for large models, so defaults to False
        model: The model loaded from HuggingFace or separately initialized. If
            None, it is automatically loaded from HuggingFace if model_name is
            passed - this just saves memory if the model was already loaded into
            RAM.
        keep_original_model (bool): If False, the original model is deleted,
            otherwise it's kept as a self.model attribute
        checkpoint (int, *optional): The checkpoint number of the model to load
            if it is a model with multiple possible checkpoints to load from.
        fold_ln (bool): If True, the LayerNorm weights are folded into the weights
        """
        super().__init__()
        if model_name == "custom":
            assert cfg is not None, "Must provide a config for custom model"
            assert (
                cfg.tensor_parallel_size == 1
            ), "Load ParallelEasyTransformer instead if you'd like to use tensor parallelism"
            self.cfg = cfg
            self.model_name = cfg.model_name
            self.model_type = cfg.model_type
            if self.cfg.tokenizer_name is not None:
                self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.tokenizer_name)
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                # If no tokenizer name is provided, we assume we're training on an algorithmic task and will pass in tokens directly. In this case, we don't need a tokenizer.
                self.tokenizer = None
            self.use_attn_result = use_attn_result
            self.hf_model = None
            self.keep_original_model = False
            # We're initializing a model, no need to load weights from a checkpoint
            self.checkpoint = None
        else:
            assert (
                model_name in self.VALID_PRETRAINED_MODEL_NAMES
            ), f"Invalid model name: {model_name}. Valid model names are: {self.VALID_PRETRAINED_MODEL_NAMES}"
            self.model_name = model_name
            if self.model_name in self.PRETRAINED_MODEL_NAMES_DICT:
                self.full_model_name = self.PRETRAINED_MODEL_NAMES_DICT[self.model_name]
            else:
                self.full_model_name = self.model_name
            self.model_type = self.get_model_type(self.full_model_name)
            if model is not None:
                self.hf_model = model
            else:
                if checkpoint is not None:
                    if "stanford" not in self.model_name:
                        logging.warning(
                            f"Loading checkpoints is not supported for the model {self.model_name}. Loading without checkpoints"
                        )
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.full_model_name
                        )
                    else:
                        assert (
                            checkpoint in self.STANFORD_CRFM_CHECKPOINTS
                        ), f"Checkpoint {checkpoint} is not valid. Available checkpoints are {self.STANFORD_CRFM_CHECKPOINTS}"
                        self.hf_model = AutoModelForCausalLM.from_pretrained(
                            self.full_model_name, revision=f"checkpoint-{checkpoint}"
                        )
                else:
                    self.hf_model = AutoModelForCausalLM.from_pretrained(
                        self.full_model_name
                    )

            self.cfg = self.convert_hf_config(
                self.hf_model.config, model_type=self.model_type
            )
            self.cfg.use_attn_result = use_attn_result
            self.cfg.checkpoint = checkpoint
            self.cfg.model_type = self.model_type
            self.cfg.model_name = self.model_name
            self.cfg.tokenizer_name = self.full_model_name
            self.cfg.normalization_type = "LNPre"
            self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.tokenizer_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if not self.cfg.d_vocab:
            assert (
                self.tokenizer is not None
            ), "Must provide a tokenizer if d_vocab is not provided"
            self.cfg.d_vocab = max(self.tokenizer.vocab.values()) + 1

        self.embed = Embed(self.cfg)
        self.hook_embed = HookPoint()  # [batch, pos, d_model]

        self.pos_embed = PosEmbed(self.cfg)
        self.hook_pos_embed = HookPoint()  # [batch, pos, d__dictmodel]

        if self.cfg.attn_only:
            self.blocks = nn.ModuleList(
                [
                    AttnOnlyBlock(self.cfg, block_index)
                    for block_index in range(self.cfg.n_layers)
                ]
            )
        else:
            self.blocks = nn.ModuleList(
                [
                    TransformerBlock(self.cfg, block_index)
                    for block_index in range(self.cfg.n_layers)
                ]
            )
        if self.cfg.normalization_type == "LN":
            self.ln_final = LayerNorm(self.cfg)
        elif self.cfg.normalization_type == "LNPre":
            # We've folded in LayerNorm weights, so just need the center + scale parts
            self.ln_final = LayerNormPre(self.cfg)
        elif self.cfg.normalization_type == "triton":
            self.ln_final = TritonLayerNorm(self.cfg)
        elif self.cfg.normalization_type is None:
            # If it's None, don't create either layer
            pass
        else:
            logging.warning(
                f"Invalid normalization_type passed in {self.cfg.normalization_type}"
            )
        self.unembed = Unembed(self.cfg)

        # Gives each module a parameter with its name (relative to this root module)
        # Needed for HookPoints to work
        self.setup()
        self.to(self.cfg.device)

    def forward(
        self,
        input,
        return_type: Optional[str] = "logits",
        cache: Optional[EasyTransformerKeyValueCache] = None,
    ):
        """Input is either a batch of tokens ([batch, pos]) or a text string.

        return_type Optional[str]: The type of output to return. Can be one of: None (return nothing, don't calculate logits), 'logits' (return logits), 'loss' (return cross-entropy loss), 'both' (return logits and loss)
        """
        if type(input) == str or type(input) == list:
            # If text, convert to tokens (batch_size=1)
            assert (
                self.tokenizer is not None
            ), "Must provide a tokenizer if passing a string to the model"
            tokens = self.to_tokens(input)
        else:
            tokens = input
        assert isinstance(tokens, torch.Tensor)
        B, S = tokens.shape
        if cache is None:
            pos_start = 0
        else:
            CB, CS, CN, CH = cache[0].past_keys.shape
            assert CB == B
            assert CN == self.cfg.n_heads
            assert CH == self.cfg.d_head
            assert CS == 0 or S == 1, "Pass in one token at a time after loading cache"
            pos_start = CS
        embed = self.hook_embed(self.embed(tokens))  # [batch, pos, d_model]
        pos_embed = self.hook_pos_embed(
            self.pos_embed(tokens, pos_start)
        )  # [batch, pos, d_model]
        residual = embed + pos_embed  # [batch, pos, d_model]
        for i, block in enumerate(self.blocks):
            # Note that each block includes skip connections, so we don't need
            # residual + block(residual)
            residual = block(
                residual, cache[i] if cache is not None else None
            )  # [batch, pos, d_model]
        if return_type is None:
            return None
        else:
            if self.cfg.normalization_type is not None:
                residual = self.ln_final(residual)  # [batch, pos, d_vocab]
            logits = self.unembed(residual)  # [batch, pos, d_vocab]
            if return_type == "logits":
                return logits
            else:
                loss = lm_cross_entropy_loss(logits, tokens)
                if return_type == "loss":
                    return loss
                elif return_type == "both":
                    return {"logits": logits, "loss": loss}
                else:
                    logging.warning(f"Invalid return_type passed in: {return_type}")
                    return None

    def set_tokenizer(self, tokenizer):
        """
        Sets the tokenizer to use for this model.
        tokenizer (PreTrainedTokenizer): a pretrained HuggingFace tokenizer
        """
        assert isinstance(tokenizer, PreTrainedTokenizer)
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.inference_mode()
    def generate(
        self,
        input: Union[str, list, torch.Tensor],
        max_new_tokens: int,
        device: Optional[torch.device] = None,
        stop_at_eos: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        do_sample: bool = False,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
        temperature: float = 1.0,
        freq_penalty: float = 0.0,
        num_beams: int = 1,
        num_return_sequences: int = 1,
        use_cache: bool = True,
        return_type: Optional[str] = None,
    ):
        """
        Sample tokens from the model until the model outputs eos_token or max_new_tokens is reached.
        Args:
            input (int): Either a batch of tokens ([batch, pos]) or a text string
            max_new_tokens (int): Maximum number of tokens to generate
            stop_at_eos (bool): If True, stop generating tokens when the model outputs eos_token
            pad_token_id (int, *optional*): The token ID to use for padding. If None, use the tokenizer's pad_token_id - required if using stop_at_eos
            eos_token_id (int, *optional*): The token ID to use for end of sentence. If None, use the tokenizer's eos_token_id - required if using stop_at_eos
            do_sample (bool): If True, sample from the model's output distribution. Otherwise, use beam or greedy search.
            top_k (int): Number of tokens to sample from. If None, sample from all tokens
            top_p (float): Probability mass to sample from. If 1.0, sample from all tokens
            temperature (float): Temperature for sampling. Higher values will make the model more random
            freq_penalty (float): Frequency penalty for sampling. Higher values will make the model more random
            num_beams (int): Number of beams to use for beam search. If 1, use greedy search
            num_return_sequences (int): Number of sequences to return for beam search
            use_cache (bool): If True, create and use cache to speed up generation
            return_type (str, *optional*): The type of the output to return - either a string (str), a list of strings (list), or a tensor of tokens (tensor). If None, defaults to input type.
        Returns:
            outputs (torch.Tensor): [batch, pos + max_new_tokens], generated sequence of new tokens - by default returns same type as input
        """
        if type(input) == str or type(input) == list:
            # If text, convert to tokens (batch_size=1)
            assert (
                self.tokenizer is not None
            ), "Must provide a tokenizer if passing a string to the model"
            tokens = self.to_tokens(input)
        else:
            tokens = input

        if return_type is None:
            if type(input) == str:
                return_type = "str"
            elif type(input) == list:
                return_type = "list"
            else:
                return_type = "tensor"

        assert isinstance(tokens, torch.Tensor)
        B, S = tokens.shape
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.to(device)
        tokens = tokens.to(device)
        if use_cache:
            cache = EasyTransformerKeyValueCache.init_cache(self.cfg, device, B)
        else:
            cache = None
        if stop_at_eos and pad_token_id is None:
            assert (
                self.tokenizer is not None and self.tokenizer.pad_token_id is not None
            ), "Must pass a pad_token_id if stop_at_eos is True and tokenizer is None or has no pad_token_id"
            pad_token_id = self.tokenizer.pad_token_id
        if stop_at_eos and eos_token_id is None:
            assert (
                self.tokenizer is not None and self.tokenizer.eos_token_id is not None
            ), "Must pass a eos_token_id if stop_at_eos is True and tokenizer is None or has no eos_token_id"
            eos_token_id = self.tokenizer.eos_token_id
        self.eval()
        if not do_sample and num_beams == 1:
            return self.greedy_search(
                tokens,
                max_new_tokens,
                stop_at_eos,
                pad_token_id,
                eos_token_id,
                cache,
                return_type,
            )
        elif not do_sample and num_beams > 1:
            raise NotImplementedError("Beam search not implemented yet")
            return self.beam_search(
                tokens, max_new_tokens, num_beams, num_return_sequences, cache
            )
        elif do_sample and num_beams > 1:
            raise NotImplementedError("Beam sampling not implemented yet")
            return self.beam_sample(
                tokens,
                max_new_tokens,
                num_beams,
                num_return_sequences,
                top_k,
                top_p,
                temperature,
                freq_penalty,
                cache,
            )
        else:
            return self.sample(
                tokens,
                max_new_tokens,
                stop_at_eos,
                pad_token_id,
                eos_token_id,
                top_k,
                top_p,
                temperature,
                freq_penalty,
                cache,
                return_type,
            )

    def greedy_search(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int,
        stop_at_eos: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        cache: Optional[EasyTransformerKeyValueCache] = None,
        return_type: Optional[str] = None,
    ):
        """
        Greedily sample tokens from the model until the model outputs eos_token or max_new_tokens is reached.
        Args:
            tokens (torch.Tensor): A batch of tokens ([batch, pos])
            max_new_tokens (int): Maximum number of tokens to generate
            stop_at_eos (bool): If True, stop generating tokens when the model outputs eos_token
            pad_token_id (int, *optional*): The token ID to use for padding. If None, use the tokenizer's pad_token_id - required if using stop_at_eos
            eos_token_id (int, *optional*): The token ID to use for end of sentence. If None, use the tokenizer's eos_token_id - required if using stop_at_eos
            cache (EasyTransformerKeyValueCache, *optional*): Cache to use for the model. If None, no cache is used
            return_type (str, *optional*): The type of the output to return - either a string (str), a list of strings (list), or a tensor of tokens (tensor). If None, defaults to tensor.
        Returns:
            outputs (torch.Tensor): [batch, pos + max_new_tokens], generated sequence of new tokens
        """
        B, S = tokens.shape
        outputs = tokens
        unfinished_sequences = tokens.new(tokens.shape[0]).fill_(1)

        if stop_at_eos and pad_token_id is None:
            assert (
                self.tokenizer is not None and self.tokenizer.pad_token_id is not None
            ), "Must pass a pad_token_id if stop_at_eos is True and tokenizer is None or has no pad_token_id"
            pad_token_id = self.tokenizer.pad_token_id
        if stop_at_eos and eos_token_id is None:
            assert (
                self.tokenizer is not None and self.tokenizer.eos_token_id is not None
            ), "Must pass a eos_token_id if stop_at_eos is True and tokenizer is None or has no eos_token_id"
            eos_token_id = self.tokenizer.eos_token_id

        for _ in tqdm(range(max_new_tokens)):
            logits = self(tokens, return_type="logits", cache=cache)
            next_tokens = torch.argmax(logits[:, -1, :], dim=-1)
            if stop_at_eos:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                    1 - unfinished_sequences
                )
                unfinished_sequences.mul_((next_tokens != eos_token_id).long())
            outputs = torch.cat([outputs, next_tokens.unsqueeze(-1)], dim=-1)
            if cache is not None:
                tokens = next_tokens.unsqueeze(-1)
            else:
                tokens = outputs

        if return_type is not None and return_type == "str":
            assert self.tokenizer is not None
            outputs = self.tokenizer.batch_decode(outputs)[0]
        elif return_type is not None and return_type == "list":
            assert self.tokenizer is not None
            outputs = self.tokenizer.batch_decode(outputs)

        return outputs

    def beam_search(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int,
        num_beams: int,
        num_return_sequences: int,
        cache: Optional[EasyTransformerKeyValueCache] = None,
    ):
        """
        Beam search for tokens from the model until the model outputs eos_token or max_new_tokens is reached.
        Args:
            tokens (torch.Tensor): A batch of tokens ([batch, pos])
            max_new_tokens (int): Maximum number of tokens to generate
            num_beams (int): Number of beams to use for beam search.
            num_return_sequences (int): Number of sequences to return for beam search
            cache (EasyTransformerKeyValueCache, *optional*): Cache to use for the model. If None, no cache is used
        Returns:
            outputs (torch.Tensor): [batch * num_return_sequences, pos + max_new_tokens], generated sequence of new tokens
        """
        assert num_return_sequences <= num_beams
        raise NotImplementedError("Beam search not implemented yet")

    def beam_sample(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int,
        num_beams: int,
        num_return_sequences: int,
        top_k: int,
        top_p: float,
        temperature: float,
        freq_penalty: float,
        cache: Optional[EasyTransformerKeyValueCache] = None,
    ):
        """
        Beam sampling for tokens from the model until the model outputs eos_token or max_new_tokens is reached.
        Args:
            tokens (torch.Tensor): A batch of tokens ([batch, pos])
            max_new_tokens (int): Maximum number of tokens to generate
            num_beams (int): Number of beams to use for beam search.
            num_return_sequences (int): Number of sequences to return for beam search
            top_k (int): Number of tokens to sample from. If None, sample from all tokens
            top_p (float): Probability mass to sample from. If 1.0, sample from all tokens
            temperature (float): Temperature for sampling. Higher values will make the model more random
            freq_penalty (float): Frequency penalty for sampling. Higher values will make the model more random
            cache (EasyTransformerKeyValueCache, *optional*): Cache to use for the model. If None, no cache is used
        Returns:
            outputs (torch.Tensor): [batch * num_return_sequences, pos + max_new_tokens], generated sequence of new tokens
        """
        assert num_return_sequences <= num_beams
        raise NotImplementedError("Beam sampling not implemented yet")

    def sample(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int,
        stop_at_eos: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
        temperature: float = 1.0,
        freq_penalty: float = 0.0,
        cache: Optional[EasyTransformerKeyValueCache] = None,
        return_type: Optional[str] = None,
    ):
        """
        Sample tokens from the model until the model outputs eos_token or max_new_tokens is reached.
        Args:
            tokens (torch.Tensor): A batch of tokens ([batch, pos])
            max_new_tokens (int): Maximum number of tokens to generate
            stop_at_eos (bool): If True, stop generating tokens when the model outputs eos_token
            pad_token_id (int, *optional*): The token ID to use for padding. If None, use the tokenizer's pad_token_id - required if using stop_at_eos
            eos_token_id (int, *optional*): The token ID to use for end of sentence. If None, use the tokenizer's eos_token_id - required if using stop_at_eos
            top_k (int): Number of tokens to sample from. If None, sample from all tokens
            top_p (float): Probability mass to sample from. If 1.0, sample from all tokens
            temperature (float): Temperature for sampling. Higher values will make the model more random
            freq_penalty (float): Frequency penalty for sampling. Higher values will make the model more random
            cache (EasyTransformerKeyValueCache, *optional*): Cache to use for the model. If None, no cache is used
            return_type (str, *optional*): If "str", return a string. If "list", return a list of strings. If None, return a tensor
        Returns:
            outputs (torch.Tensor): [batch, pos + max_new_tokens], generated sequence of new tokens
        """
        B, S = tokens.shape
        outputs = tokens
        unfinished_sequences = tokens.new(tokens.shape[0]).fill_(1)

        if stop_at_eos and pad_token_id is None:
            assert (
                self.tokenizer is not None and self.tokenizer.pad_token_id is not None
            ), "Must pass a pad_token_id if stop_at_eos is True and tokenizer is None or has no pad_token_id"
            pad_token_id = self.tokenizer.pad_token_id
        if stop_at_eos and eos_token_id is None:
            assert (
                self.tokenizer is not None and self.tokenizer.eos_token_id is not None
            ), "Must pass a eos_token_id if stop_at_eos is True and tokenizer is None or has no eos_token_id"
            eos_token_id = self.tokenizer.eos_token_id

        def sample_logits(input_ids, logits, top_k, top_p, temperature, freq_penalty):
            assert temperature > 0, "temperature has to be greater than 0"
            logits = logits / temperature
            if freq_penalty > 0:
                for b in range(logits.shape[0]):
                    logits[b] = logits[b] - freq_penalty * torch.bincount(
                        input_ids[b], minlength=logits.shape[-1]
                    )
            if top_k is not None:
                assert top_k > 0, "top_k has to be greater than 0"
                top_logits, top_idx = logits.topk(top_k)
                indices_to_remove = logits < top_logits[..., -1].unsqueeze(-1)
                logits = logits.masked_fill(indices_to_remove, -float("inf"))
            if top_p < 1.0:
                assert top_p > 0.0, "top_p has to be greater than 0"
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits = logits.masked_fill(indices_to_remove, -float("inf"))
            return torch.distributions.categorical.Categorical(logits=logits).sample()

        for _ in tqdm(range(max_new_tokens)):
            logits = self(tokens, return_type="logits", cache=cache)
            next_tokens = sample_logits(
                outputs,
                logits[:, -1, :],
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                freq_penalty=freq_penalty,
            )
            if stop_at_eos:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                    1 - unfinished_sequences
                )
                unfinished_sequences.mul_((next_tokens != eos_token_id).long())
            outputs = torch.cat([outputs, next_tokens.unsqueeze(-1)], dim=-1)
            if cache is not None:
                tokens = next_tokens.unsqueeze(-1)
            else:
                tokens = outputs

        if return_type is not None and return_type == "str":
            assert self.tokenizer is not None
            outputs = self.tokenizer.batch_decode(outputs)[0]
        elif return_type is not None and return_type == "list":
            assert self.tokenizer is not None
            outputs = self.tokenizer.batch_decode(outputs)

        return outputs

    def to_tokens(self, text):
        assert self.tokenizer is not None
        return self.tokenizer(text, return_tensors="pt", padding=True)["input_ids"]

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        fold_ln: bool = True,
        center_writing_weights: bool = True,
        center_unembed: bool = True,
        keep_original_model: bool = True,
        **kwargs,
    ):
        """
        Class method to load a pretrained model from HuggingFace and to automatically convert and load those weights into EasyTransformer format.

        Args:
            model_name (str): The model name - must be in VALID_MODEL_NAMES
            fold_ln (bool): Whether to fold in the LayerNorm weights to the subsequent linear layer. This does not change the computation. Defaults to True.
            center_writing_weights (bool): Whether to center weights writing to the residual stream (ie set mean to be zero). Due to LayerNorm this doesn't change the computation. Defaults to True.
            center_unembed (bool): Whether to center W_U (ie set mean to be zero). Softmax is translation invariant so this doesn't affect log probs or loss, but does change logits. Defaults to True.
            keep_original_model (bool): Whether to delete the model loaded from HuggingFace (stored as model.hf_model). Defaults to False.

        Returns:
            model: The EasyTransformer model
        """
        model = cls(model_name, fold_ln=fold_ln, **kwargs)

        # Load model weights, and fold in layer norm weights
        if model.model_type == "gpt2":
            state_dict = weight_conversion.convert_gpt2_weights(
                model.hf_model, model.cfg
            )
        elif model.model_type == "neo":
            state_dict = weight_conversion.convert_neo_weights(
                model.hf_model, model.cfg
            )
        elif model.model_type == "gptj":
            state_dict = weight_conversion.convert_gptj_weights(
                model.hf_model, model.cfg
            )
        elif model.model_type == "neox":
            state_dict = weight_conversion.convert_neox_weights(
                model.hf_model, model.cfg
            )
        elif model.model_type == "opt":
            state_dict = weight_conversion.convert_opt_weights(
                model.hf_model, model.cfg
            )
        else:
            logging.warning(
                f"Invalid model_type, no weights are stored to load: {model.model_type}, generated from model name {model.model_name}"
            )
        state_dict = model.fill_missing_keys(state_dict)
        if fold_ln:
            state_dict = model.fold_layer_norm(state_dict)
        if center_writing_weights:
            state_dict = model.center_writing_weights(state_dict)
        if center_unembed:
            state_dict = model.center_unembed(state_dict)
        # Need to delete the HuggingFace model so it isn't counted as a submodule
        del model.hf_model
        model.load_state_dict(state_dict)
        return model

    @classmethod
    def from_config(cls, cfg):
        """Used to generate a model from a config object to train from

        Args:
            cfg (EasyTransformerConfig): Config for the model

        Returns:
            model: An initialised EasyTransformer model
        """
        if isinstance(cfg, Dict):
            cfg = EasyTransformerConfig(**cfg)
        model = cls(
            "custom",
            cfg,
            use_attn_result=cfg.use_attn_result,
        )
        model.init_weights()
        return model

    def get_model_type(self, model_name):
        if "gpt2" in model_name or "stanford" in model_name:
            return "gpt2"
        elif "opt" in model_name:
            return "opt"
        elif model_name == "EleutherAI/gpt-neox-20b":
            return "neox"
        elif model_name == "EleutherAI/gpt-j-6B":
            return "gptj"
        elif "neo" in model_name:
            return "neo"
        else:
            raise ValueError(f"Invalid model name: {model_name}")

    def convert_hf_config(self, hf_config, model_type):
        cfg_dict = {}
        if model_type == "neo":
            cfg_dict = {
                "d_model": hf_config.hidden_size,
                "d_head": hf_config.hidden_size // hf_config.num_heads,
                "n_heads": hf_config.num_heads,
                "d_mlp": hf_config.hidden_size * 4,
                "n_layers": hf_config.num_layers,
                "n_ctx": hf_config.max_position_embeddings,
                "eps": hf_config.layer_norm_epsilon,
                "d_vocab": hf_config.vocab_size,
                "attn_types": hf_config.attention_layers,
                "act_fn": hf_config.activation_function,
                "use_attn_scale": False,
                "use_local_attn": True,
                "window_size": hf_config.window_size,
            }
        elif model_type == "gpt2":
            cfg_dict = {
                "d_model": hf_config.n_embd,
                "d_head": hf_config.n_embd // hf_config.n_head,
                "n_heads": hf_config.n_head,
                "d_mlp": hf_config.n_embd * 4,
                "n_layers": hf_config.n_layer,
                "n_ctx": hf_config.n_ctx,
                "eps": hf_config.layer_norm_epsilon,
                "d_vocab": hf_config.vocab_size,
                "act_fn": hf_config.activation_function,
                "use_attn_scale": True,
                "use_local_attn": False,
            }
        elif model_type == "opt":
            cfg_dict = {
                "d_model": hf_config.hidden_size,
                "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
                "n_heads": hf_config.num_attention_heads,
                "d_mlp": hf_config.ffn_dim,
                "n_layers": hf_config.num_hidden_layers,
                "n_ctx": hf_config.max_position_embeddings,
                "eps": 1e-5,
                "d_vocab": hf_config.vocab_size,
                "act_fn": hf_config.activation_function,
                "use_attn_scale": True,
                "use_local_attn": False,
            }
        elif model_type == "gptj":
            raise NotImplementedError
        elif model_type == "neox":
            raise NotImplementedError
        else:
            raise NotImplementedError
        cfg_dict["model_name"] = self.model_name
        cfg_dict["model_type"] = model_type
        cfg = EasyTransformerConfig.from_dict(cfg_dict)
        return cfg

    def init_weights(self):
        """
        Initialize weights matrices with a normal of std=initializer_range (default=0.02) and truncated between [-2, 2]. This roughly follows the GPT-2 paper's scheme (but with truncation, and not halving the std for W_pos).
        LayerNorm weights are already initialized to 1.0, and all biases are initialized to 0.0 (including LayerNorm), so this just initializes weight matrices.

        Weight matrices are set to empty by default (to save space + compute, since they're the bulk of the parameters), so it is important to call this if you are not loading in pretrained weights! Note that this function assumes that weight names being with W_
        Set seed here to ensure determinism.
        This does NOT follow the PyTorch scheme, which as far as I can tell is super out of date but no one has gotten round to updating it?
        https://github.com/pytorch/pytorch/issues/18182

        PyTorch Transformers are especially bad - TransformerEncoder initializes all layers to the same values?! https://github.com/pytorch/pytorch/issues/72253

        """

        if self.cfg.seed is not None:
            torch.manual_seed(self.cfg.seed)

        for name, param in self.named_parameters():
            if "W_" in name:
                nn.init.trunc_normal_(param, std=self.cfg.initializer_range)

    def fill_missing_keys(self, state_dict):
        """Takes in a state dict from a pretrained model, and fills in any missing keys with the default initialization.
        Args:
            state_dict (dict): State dict from a pretrained model
        Returns:
            dict: State dict with missing keys filled in
        """
        # Get the default state dict
        default_state_dict = self.state_dict()
        # Get the keys that are missing from the pretrained model
        missing_keys = set(default_state_dict.keys()) - set(state_dict.keys())
        # Fill in the missing keys with the default initialization
        for key in missing_keys:
            if "hf_model" in key:
                # Skip keys that are from the HuggingFace model, if loading from HF.
                continue
            if "W_" in key:
                logging.warning(
                    "Missing key for a weight matrix in pretrained, filled in with an empty tensor: {}".format(
                        key
                    )
                )
            state_dict[key] = default_state_dict[key]
        return state_dict

    def fold_layer_norm(self, state_dict: Dict[str, torch.Tensor]):
        """Takes in a state dict from a pretrained model, formatted to be consistent with EasyTransformer but with LayerNorm weights and biases. Folds these into the neighbouring weights.
        Args:
            state_dict (Dict[str, torch.Tensor]): State dict of pretrained model
        """
        for l in range(self.cfg.n_layers):
            # Fold ln1 into attention - it's important to fold biases first,
            # since biases depend on weights but not vice versa
            state_dict[f"blocks.{l}.attn.b_Q"] = (
                state_dict[f"blocks.{l}.attn.b_Q"]
                + state_dict[f"blocks.{l}.attn.W_Q"] @ state_dict[f"blocks.{l}.ln1.b"]
            )
            state_dict[f"blocks.{l}.attn.b_K"] = (
                state_dict[f"blocks.{l}.attn.b_K"]
                + state_dict[f"blocks.{l}.attn.W_K"] @ state_dict[f"blocks.{l}.ln1.b"]
            )
            state_dict[f"blocks.{l}.attn.b_V"] = (
                state_dict[f"blocks.{l}.attn.b_V"]
                + state_dict[f"blocks.{l}.attn.W_V"] @ state_dict[f"blocks.{l}.ln1.b"]
            )

            state_dict[f"blocks.{l}.attn.W_Q"] = (
                state_dict[f"blocks.{l}.attn.W_Q"] * state_dict[f"blocks.{l}.ln1.w"]
            )
            state_dict[f"blocks.{l}.attn.W_K"] = (
                state_dict[f"blocks.{l}.attn.W_K"] * state_dict[f"blocks.{l}.ln1.w"]
            )
            state_dict[f"blocks.{l}.attn.W_V"] = (
                state_dict[f"blocks.{l}.attn.W_V"] * state_dict[f"blocks.{l}.ln1.w"]
            )

            # Fold ln2 into MLP
            state_dict[f"blocks.{l}.mlp.b_in"] = (
                state_dict[f"blocks.{l}.mlp.b_in"]
                + state_dict[f"blocks.{l}.mlp.W_in"] @ state_dict[f"blocks.{l}.ln2.b"]
            )
            state_dict[f"blocks.{l}.mlp.W_in"] = (
                state_dict[f"blocks.{l}.mlp.W_in"] * state_dict[f"blocks.{l}.ln2.w"]
            )
            del (
                state_dict[f"blocks.{l}.ln1.w"],
                state_dict[f"blocks.{l}.ln1.b"],
                state_dict[f"blocks.{l}.ln2.w"],
                state_dict[f"blocks.{l}.ln2.b"],
            )
        # Fold ln_final into Unembed
        state_dict[f"unembed.b_U"] = (
            state_dict[f"unembed.W_U"] @ state_dict[f"ln_final.b"]
        )
        state_dict[f"unembed.W_U"] = (
            state_dict[f"unembed.W_U"] * state_dict[f"ln_final.w"]
        )
        del state_dict[f"ln_final.w"], state_dict[f"ln_final.b"]
        return state_dict

    def center_writing_weights(self, state_dict: Dict[str, torch.Tensor]):
        """Centers the weights of the model that write to the residual stream - W_out, W_E, W_pos and W_out. This is done by subtracting the mean of the weights from the weights themselves. This is done in-place. As LayerNorm centers before reading from the residual stream, this doesn't change the computation."""
        state_dict["embed.W_E"] = state_dict["embed.W_E"] - state_dict[
            "embed.W_E"
        ].mean(0, keepdim=True)
        state_dict["pos_embed.W_pos"] = state_dict["pos_embed.W_pos"] - state_dict[
            "pos_embed.W_pos"
        ].mean(0, keepdim=True)
        for l in range(self.cfg.n_layers):
            state_dict[f"blocks.{l}.attn.W_O"] = state_dict[
                f"blocks.{l}.attn.W_O"
            ] - state_dict[f"blocks.{l}.attn.W_O"].mean(
                1, keepdim=True
            )  # W_O is [head_index, d_model, d_head]
            state_dict[f"blocks.{l}.attn.b_O"] = (
                state_dict[f"blocks.{l}.attn.b_O"]
                - state_dict[f"blocks.{l}.attn.b_O"].mean()
            )  # b_O is [d_model]
            state_dict[f"blocks.{l}.mlp.W_out"] = state_dict[
                f"blocks.{l}.mlp.W_out"
            ] - state_dict[f"blocks.{l}.mlp.W_out"].mean(0, keepdim=True)
            state_dict[f"blocks.{l}.mlp.b_out"] = (
                state_dict[f"blocks.{l}.mlp.b_out"]
                - state_dict[f"blocks.{l}.mlp.b_out"].mean()
            )
        return state_dict

    def center_unembed(self, state_dict: Dict[str, torch.Tensor]):
        """Centers the unembedding weights W_U. This is done by subtracting the mean of the weights from the weights themselves. This is done in-place. As softmax is translation invariant, this changes the logits but not the log probs, and makes the model logits more interpretable."""
        state_dict["unembed.W_U"] = state_dict["unembed.W_U"] - state_dict[
            "unembed.W_U"
        ].mean(0, keepdim=True)
        state_dict["unembed.b_U"] = (
            state_dict["unembed.b_U"] - state_dict["unembed.b_U"].mean()
        )
        return state_dict


class ParallelEasyTransformer(HookedRootModule):
    def __init__(self, model_name: str, cfg: Optional[EasyTransformerConfig] = None):
        super().__init__()
        if model_name == "custom":
            assert cfg is not None, "Must provide a config for custom model"
            assert (
                cfg.tensor_parallel_size > 1 or cfg.pipeline_parallel_size > 1
            ), "Must use either tensor or pipeline parallelism"
            self.cfg = cfg

            initialize_parallel_groups(
                self.cfg.tensor_parallel_size, self.cfg.pipeline_parallel_size
            )

            self.model_name = cfg.model_name
            self.model_type = cfg.model_type
            if self.cfg.tokenizer_name is not None:
                self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.tokenizer_name)
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                # If no tokenizer name is provided, we assume we're training on an algorithmic task and will pass in tokens directly. In this case, we don't need a tokenizer.
                self.tokenizer = None
        else:
            raise NotImplementedError(
                "Only custom models are supported for now for parallelism"
            )

        if not self.cfg.d_vocab:
            assert (
                self.tokenizer is not None
            ), "Must provide a tokenizer if d_vocab is not provided"
            self.cfg.d_vocab = max(self.tokenizer.vocab.values()) + 1

        self.embed = EmbedParallelSplitVocab(self.cfg)
        self.hook_embed = HookPoint()  # [batch, pos, d_model]

        self.pos_embed = PosEmbed(self.cfg)
        self.hook_pos_embed = HookPoint()  # [batch, pos, d__dictmodel]

        if self.cfg.attn_only:
            self.blocks = nn.ModuleList(
                [
                    AttnOnlyParallelBlock(self.cfg, block_index)
                    for block_index in range(self.cfg.n_layers)
                ]
            )
        else:
            self.blocks = nn.ModuleList(
                [
                    TransformerParallelBlock(self.cfg, block_index)
                    for block_index in range(self.cfg.n_layers)
                ]
            )
        if self.cfg.normalization_type == "LN":
            self.ln_final = LayerNorm(self.cfg)
        elif self.cfg.normalization_type == "LNPre":
            # We've folded in LayerNorm weights, so just need the center + scale parts
            self.ln_final = LayerNormPre(self.cfg)
        elif self.cfg.normalization_type == "triton":
            self.ln_final = TritonLayerNorm(self.cfg)
        elif self.cfg.normalization_type is None:
            # If it's None, don't create either layer
            pass
        else:
            logging.warning(
                f"Invalid normalization_type passed in {self.cfg.normalization_type}"
            )
        self.unembed = UnembedParallelSplitVocab(self.cfg)

        # Gives each module a parameter with its name (relative to this root module)
        # Needed for HookPoints to work
        self.setup()
        self.to(self.cfg.device)

    @classmethod
    def from_config(cls, cfg: EasyTransformerConfig):
        if isinstance(cfg, Dict):
            cfg = EasyTransformerConfig(**cfg)
        model = cls(
            "custom",
            cfg,
        )
        model.init_weights()
        return model

    def init_weights(self):
        """
        Initialize weights matrices with a normal of std=initializer_range (default=0.02) and truncated between [-2, 2]. This roughly follows the GPT-2 paper's scheme (but with truncation, and not halving the std for W_pos).
        LayerNorm weights are already initialized to 1.0, and all biases are initialized to 0.0 (including LayerNorm), so this just initializes weight matrices.

        Weight matrices are set to empty by default (to save space + compute, since they're the bulk of the parameters), so it is important to call this if you are not loading in pretrained weights! Note that this function assumes that weight names being with W_
        Set seed here to ensure determinism.
        This does NOT follow the PyTorch scheme, which as far as I can tell is super out of date but no one has gotten round to updating it?
        https://github.com/pytorch/pytorch/issues/18182

        PyTorch Transformers are especially bad - TransformerEncoder initializes all layers to the same values?! https://github.com/pytorch/pytorch/issues/72253

        """

        if self.cfg.seed is not None:
            torch.manual_seed(self.cfg.seed)

        for name, param in self.named_parameters():
            if "W_" in name:
                nn.init.trunc_normal_(param, std=self.cfg.initializer_range)

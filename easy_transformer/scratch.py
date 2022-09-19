# %%
# Import stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
import tqdm.notebook as tqdm

import random
import time

# from google.colab import drive
from pathlib import Path
import pickle
import os


import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "colab"
import plotly.graph_objects as go

from torch.utils.data import DataLoader

from functools import *
import pandas as pd
import gc
import collections
import copy

# import comet_ml
import itertools
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import dataclasses
import datasets
# %%
from easy_transformer.utils import gelu_new, to_numpy, get_corner, tokenize_and_concatenate #helper functions
from easy_transformer.hook_points import HookedRootModule, HookPoint
from easy_transformer.EasyTransformer import EasyTransformer,TransformerBlock, MLP, Attention, LayerNormPre, PosEmbed, Unembed, Embed
from easy_transformer.experiments import ExperimentMetric, AblationConfig, EasyAblation, EasyPatching, PatchingConfig
from easy_transformer.EasyTransformerConfig import EasyTransformerConfig
from easy_transformer.train import train, EasyTransformerTrainConfig

# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# %%
micro_gpt_cfg = EasyTransformerConfig(
    d_model = 384,
    d_head = 64,
    n_heads = 6,
    d_mlp = 1536,
    n_layers=6,
    n_ctx = 512,
    act_fn='gelu_new',
    normalization_type='LN',
    tokenizer_name='gpt2',
    )
micro_gpt = EasyTransformer.from_config(micro_gpt_cfg)
# %%
dataset = datasets.load_dataset("stas/openwebtext-10k", split="train")
assert isinstance(dataset, datasets.arrow_dataset.Dataset)
dataset = tokenize_and_concatenate(dataset, micro_gpt.tokenizer, max_length=micro_gpt.cfg.n_ctx, add_bos_token=False)
# %%
training_cfg = EasyTransformerTrainConfig(
    num_epochs = 1,
    batch_size = 16,
    weight_decay = 0.01,
    lr=1e-2,
    max_grad_norm=1.0,
    optimizer_name = 'TritonAdam',
    print_every=320,
    device="cuda",
)
assert isinstance(dataset, datasets.arrow_dataset.Dataset)
micro_gpt = train(micro_gpt, training_cfg, dataset)
# %%
micro_gpt.tokenizer.batch_decode(micro_gpt.generate("The red-tailed hawk", max_new_tokens=50, do_sample=True, temperature=2))
#%%

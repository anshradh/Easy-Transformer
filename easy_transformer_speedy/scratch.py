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
from easy_transformer_speedy.utils import (
    gelu_new,
    to_numpy,
    get_corner,
    tokenize_and_concatenate,
)  # helper functions
from easy_transformer_speedy.hook_points import HookedRootModule, HookPoint
from easy_transformer_speedy.EasyTransformer import (
    EasyTransformer,
    TransformerBlock,
    MLP,
    Attention,
    LayerNormPre,
    PosEmbed,
    Unembed,
    Embed,
)
from easy_transformer_speedy.experiments import (
    ExperimentMetric,
    AblationConfig,
    EasyAblation,
    EasyPatching,
    PatchingConfig,
)
from easy_transformer_speedy.EasyTransformerConfig import EasyTransformerConfig
from easy_transformer_speedy.train import run_train, EasyTransformerTrainConfig

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
# %%
small_gpt_cfg = EasyTransformerConfig(
    d_model=128,
    d_head=16,
    n_heads=8,
    d_mlp=4 * 128,
    n_layers=4,
    n_ctx=512,
    act_fn="gelu_new",
    normalization_type="LN",
    tokenizer_name="gpt2",
    use_triton=True,
)
small_gpt = EasyTransformer.from_config(small_gpt_cfg)
# %%
dataset = datasets.load_dataset("stas/openwebtext-10k", split="train")
assert isinstance(dataset, datasets.arrow_dataset.Dataset)
dataset = tokenize_and_concatenate(
    dataset, small_gpt.tokenizer, max_length=small_gpt.cfg.n_ctx, add_bos_token=False
)
# %%
training_cfg = EasyTransformerTrainConfig(
    num_epochs=1,
    batch_size=16,
    weight_decay=0.01,
    lr=1e-3,
    max_grad_norm=1.0,
    optimizer_name="TritonAdam",
    print_every=128,
    device="cuda",
    wandb=True,
    wandb_project_name="TritonAdamTesting",
    max_steps=640,
    # warmup_steps=128,
)
assert isinstance(dataset, datasets.arrow_dataset.Dataset)
run_train(small_gpt, training_cfg, dataset)
# %%
small_gpt.generate(
    "The red-tailed hawk", max_new_tokens=50, do_sample=True, temperature=0.7
)
#%%

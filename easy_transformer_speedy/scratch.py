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
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    small_gpt_cfg = EasyTransformerConfig(
        d_model=768,
        d_head=64,
        n_heads=12,
        d_mlp=4 * 768,
        n_layers=12,
        n_ctx=1024,
        act_fn="gelu_new",
        normalization_type="LN",
        tokenizer_name="gpt2",
    )
    small_gpt = EasyTransformer.from_config(small_gpt_cfg)

    dataset = datasets.load_dataset("stas/openwebtext-10k", split="train")
    assert isinstance(dataset, datasets.arrow_dataset.Dataset)
    dataset = tokenize_and_concatenate(
        dataset,
        small_gpt.tokenizer,
        max_length=small_gpt.cfg.n_ctx,
        add_bos_token=False,
    )

    training_cfg = EasyTransformerTrainConfig(
        num_epochs=20,
        batch_size=8,
        weight_decay=0.01,
        lr=5e-4,
        max_grad_norm=1.0,
        optimizer_name="TritonAdam",
        print_every=128,
        device="cuda",
        wandb=True,
        wandb_project_name="DDPTesting",
        num_devices=8,
        save_every=640,
        save_dir="models"
        # max_steps=640,
        # warmup_steps=128,
    )
    assert isinstance(dataset, datasets.arrow_dataset.Dataset)
    run_train(small_gpt, training_cfg, dataset)

    trained_model = EasyTransformer.from_config(small_gpt_cfg)
    trained_model.load_state_dict(torch.load("models/final_model.pt"))
    trained_model.cuda()
# %%
if __name__ == "__main__":
    print(
        trained_model.generate(
            "The German government released a study this week that found that one in five Germans has anti-Semitic attitudes.", max_new_tokens=50, do_sample=True, temperature=0.7
        )
    )
#%%

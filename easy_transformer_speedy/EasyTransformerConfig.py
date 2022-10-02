from dataclasses import dataclass
from typing import Union, Tuple, List, Dict, Any, Optional
import torch
import torch.nn as nn
import numpy as np
import random


@dataclass
class EasyTransformerConfig:
    """
    Configuration class to store the configuration of a EasyTransformer model.
    Args:
        d_model (int): The dimensionality of the embeddings.
        d_head (int): The dimensionality of each attention head.
        n_heads (int): The number of attention heads.
        n_layers (int): The number of attention layers.
        n_ctx (int): The maximum sequence length.
        d_mlp (int, *optional): The dimensionality of the feedforward mlp network. Must be set unless using an attn-only model.
        act_fn (str): The activation function to use. Always lowercase. Supports ['relu', 'gelu', 'silu', 'glu', 'gelu_new', 'solu_ln']. Must be set unless using an attn-only model.
        d_vocab (int): The size of the vocabulary. If not set, will be automatically set
            from the tokenizer's vocab size.
        eps (float): The epsilon value to use for layer normalization. Defaults to 1e-5
        use_attn_result (bool): whether to explicitly calculate the amount
            each head adds to the residual stream (with a hook) and THEN add it
            up, vs just calculating the sum. This can be very memory intensive
            for large models, so defaults to False
        use_attn_scale (bool): whether to scale the attention weights by
        1/sqrt(d_head)
        use_local_attn (bool): whether to use local attention
        model_name (str, *optional*): the name of the model, used to load
            weights from HuggingFace or initialized to "custom" if not passed
        model_type (str, *optional*): the type of the model, used to help load
            weights from HuggingFace or initialized to "custom" if not passed
        full_model_name (str, *optional*): the full name of the model,
            initialized to "custom" if not passed
        tokenizer_name (str, *optional*): the full name of the model, passed into
            HuggingFace to access the tokenizer. Only used when passing in custom
            config, if loading from pretrained then this is not needed.
        window_size (int, *optional*): the size of the window for local
            attention
        attn_types (List[str], *optional*): the types of attention to use for
            local attention
        weight_init_mode (str): the initialization mode to use for the
            weights. Only relevant for custom models, ignored for pre-trained. Options
            are 'pytorch' (for PyTorch defaults) and 'gpt2' (for GPT-2 defaults),
            defaults to 'gpt2
        normalization_type (str, *optional*): the type of normalization to use. Options
            are None (no normalization), 'LN' (use LayerNorm, including weights &
            biases) and 'LNPre' (use LayerNorm, but no weights & biases). Defaults to
            None
        device(str): The device to use for the model. Defaults to 'cuda' if available,
            else 'cpu
        attention_dir (str): Whether to use causal (aka unidirectional aka GPT-2
            style) or bidirectional attention. Options are 'causal' and 'bidirectional'.
            Defaults to 'causal'
        attn_only (bool): Whether to only use attention layers, no feedforward
            layers. Defaults to False
        seed (int, *optional*): The seed to use for the model. Defaults to 42. Used to set sources of randomness (Python, PyTorch and NumPy) and to initialize weights. If set to None, does nothing.
        initializer_range (float): The standard deviation of the truncated normal used to initialise the weights.
        tensor_parallel_size (int): The number of tensor parallel groups to use. Defaults to 1.
        use_triton (bool): Whether to use custom triton kernels. Defaults to False. If true, overwrites normalization_type to "triton"
    """

    d_model: int
    d_head: int
    n_heads: int
    n_layers: int
    n_ctx: int
    d_mlp: Optional[int] = None
    act_fn: Optional[str] = None
    d_vocab: Optional[int] = None
    eps: float = 1e-5
    use_attn_result: bool = False
    use_attn_scale: bool = True
    use_local_attn: bool = False
    model_name: Optional[str] = None
    model_type: Optional[str] = None
    checkpoint: Optional[int] = None
    full_model_name: Optional[str] = None
    tokenizer_name: Optional[str] = None
    window_size: Optional[int] = None
    attn_types: Optional[List] = None
    init_mode: str = "gpt2"
    normalization_type: Optional[str] = None
    gated_act_fn: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    attention_dir: str = "causal"
    attn_only: bool = False
    seed: int = 42
    initializer_range: float = 0.02
    tensor_parallel_size: int = 1
    use_triton: bool = False

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        if self.seed is not None:
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
        if self.use_local_attn:
            assert (
                self.window_size is not None
            ), "window_size must be specified for local attention"
            assert (
                self.attn_types is not None
            ), "attn_types must be specified for local attention"
        if self.model_name is None:
            self.model_name = "custom"
            self.model_type = "custom"
            self.full_model_name = "custom"
        if not self.attn_only:
            assert (
                self.d_mlp is not None
            ), "d_mlp must be specified for non-attn-only models"
            assert (
                self.act_fn is not None
            ), "act_fn must be specified for non-attn-only models"
        if self.use_triton:
            self.normalization_type = "triton"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """
        Instantiates a `EasyTransformerConfig` from a Python dictionary of parameters.
        """
        return cls(**config_dict)

from dataclasses import dataclass
from typing import Union, Tuple, List, Dict, Any, Optional
import torch
import torch.nn as nn


@dataclass
class EasyTransformerConfig:
    """
    Configuration class to store the configuration of a EasyTransformer model.
    Args:
        d_model (int): The dimensionality of the embeddings.
        d_head (int): The dimensionality of each attention head.
        n_heads (int): The number of attention heads.
        d_mlp (int): The dimensionality of the feedforward mlp network.
        n_layers (int): The number of attention layers.
        n_ctx (int): The maximum sequence length.
        eps (float): The epsilon value to use for layer normalization.
        d_vocab (int): The size of the vocabulary.
        act_fn (str): The activation function to use.
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
        window_size (int, *optional*): the size of the window for local
            attention
        attn_types (List[str], *optional*): the types of attention to use for
            local attention
    """

    d_model: int
    d_head: int
    n_heads: int
    d_mlp: int
    n_layers: int
    n_ctx: int
    eps: float
    d_vocab: int
    act_fn: str
    use_attn_result: bool = False
    use_attn_scale: bool = True
    use_local_attn: bool = False
    model_name: Optional[str] = None
    model_type: Optional[str] = None
    checkpoint: Optional[int] = None
    full_model_name: Optional[str] = None
    window_size: Optional[int] = None
    attn_types: Optional[List] = None

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
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

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """
        Instantiates a `EasyTransformerConfig` from a Python dictionary of parameters.
        """
        return cls(**config_dict)

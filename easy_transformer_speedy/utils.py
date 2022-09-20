import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import datasets
import einops
from transformers import AutoTokenizer


def get_sample_from_dataset(sequences, nb_sample=2, print_len=10):
    rd_idx = np.random.randint(0, len(sequences), 3)
    return "\n".join([str(sequences[k][:print_len]) + " ... " for k in rd_idx])


def print_gpu_mem(step_name=""):
    print(
        f"{step_name} ~ {np.round(torch.cuda.memory_allocated()/2e30, 2)} GiB allocated on GPU."
    )


def get_corner(tensor, n=2):
    # Prints the top left corner of the tensor
    if len(tensor.shape) == 0:
        return tensor
    elif len(tensor.shape) == 1:
        return tensor[:n]
    elif len(tensor.shape) == 2:
        return tensor[:n, :n]
    elif len(tensor.shape) == 3:
        return tensor[:n, :n, :n]
    elif len(tensor.shape) == 4:
        return tensor[:n, :n, :n, :n]
    elif len(tensor.shape) == 5:
        return tensor[:n, :n, :n, :n, :n]
    elif len(tensor.shape) == 6:
        return tensor[:n, :n, :n, :n, :n, :n]
    else:
        # I never need tensors of rank > 6
        raise ValueError(f"Tensor of shape {tensor.shape} is too big")


def to_numpy(tensor, flat=False):
    if (type(tensor) != torch.Tensor) and (
        type(tensor) != torch.nn.parameter.Parameter
    ):
        return tensor
    if flat:
        return tensor.flatten().detach().cpu().numpy()
    else:
        return tensor.detach().cpu().numpy()


def gelu_new(input):
    # Implementation of GeLU used by GPT2 - subtly different from PyTorch's
    return (
        0.5
        * input
        * (
            1.0
            + torch.tanh(
                np.sqrt(2.0 / np.pi) * (input + 0.044715 * torch.pow(input, 3.0))
            )
        )
    )


def solu(input):
    """
    SoLU activation function as described by
    https://transformer-circuits.pub/2022/solu/index.html.
    
    LayerNorm implemented by the MLP class.
    """
    return input * F.softmax(input, dim=-1)


def reglu(input, gate):
    """
    ReGLU activation function as described by
    https://arxiv.org/pdf/2002.05202.pdf.
    """
    return F.relu(gate) * input


def geglu(input, gate, use_gelu_new=False):
    """
    GeGLU activation function as described by
    https://arxiv.org/pdf/2002.05202.pdf.
    """
    if use_gelu_new:
        return gelu_new(gate) * input
    else:
        return F.gelu(gate) * input


def swiglu(input, gate):
    """
    SwiGLU activation function as described by
    https://arxiv.org/pdf/2002.05202.pdf.
    """
    return F.silu(gate) * input

def tokenize_and_concatenate(dataset: datasets.arrow_dataset.Dataset, 
                             tokenizer: AutoTokenizer, 
                             streaming=False, 
                             max_length=1024, 
                             column_name='text', 
                             add_bos_token=True):
    """Helper function to tokenizer and concatenate a dataset of text. This converts the text to tokens, concatenates them (separated by EOS tokens) and then reshapes them into a 2D array of shape (____, sequence_length), dropping the last batch. Tokenizers are much faster if parallelised, so we chop the string into 20, feed it into the tokenizer, in parallel with padding, then remove padding at the end. 
    
    This tokenization is useful for training language models, as it allows us to efficiently train on a large corpus of text of varying lengths (without, eg, a lot of truncation or padding). Further, for models with absolute positional encodings, this avoids privileging early tokens (eg, news articles often begin with CNN, and models may learn to use early positional encodings to predict these)

    Args:
        dataset (datasets.arrow_dataset.Dataset): The dataset to tokenize, assumed to be a HuggingFace text dataset.
        tokenizer (AutoTokenizer): The tokenizer. Assumed to have a bos_token_id and an eos_token_id.
        streaming (bool, optional): Whether the dataset is being streamed. If True, avoids using parallelism. Defaults to False.
        max_length (int, optional): The length of the context window of the sequence. Defaults to 1024.
        column_name (str, optional): The name of the text column in the dataset. Defaults to 'text'.
        add_bos_token (bool, optional): . Defaults to True.

    Returns:
        datasets.arrow_dataset.Dataset: Returns the tokenized dataset, as a dataset of tensors, with a single column called "tokens"
    """
    if tokenizer.pad_token is None:
        # We add a padding token, purely to implement the tokenizer. This will be removed before inputting tokens to the model, so we do not need to increment d_vocab in the model.
        tokenizer.add_special_tokens({'pad_token': "<PAD>"})
    # Define the length to chop things up into - leaving space for a bos_token if required
    if add_bos_token:
        seq_len = max_length - 1
    else:
        seq_len = max_length
    
    def tokenize_function(examples):
        text = examples[column_name]
        # Concatenate it all into an enormous string, separated by eos_tokens
        full_text = tokenizer.bos_token.join(text)
        # Divide into 20 chunks of ~ equal length
        num_chunks = 20
        chunk_length = (len(full_text)-1)//num_chunks + 1
        chunks = [full_text[i*chunk_length:(i+1)*chunk_length] for i in range(num_chunks)]
        # Tokenize the chunks in parallel. Uses NumPy because HuggingFace map doesn't want tensors returned
        tokens = tokenizer(chunks, return_tensors='np', padding=True)['input_ids'].flatten()
        # Drop padding tokens
        tokens = tokens[tokens!=tokenizer.pad_token_id]
        num_tokens = len(tokens)
        num_batches = num_tokens//(seq_len)
        # Drop the final tokens if not enough to make a full sequence
        tokens = tokens[:seq_len*num_batches]
        tokens = einops.rearrange(tokens, '(batch seq) -> batch seq', batch=num_batches, seq=seq_len)
        if add_bos_token:
            prefix = np.full((num_batches, 1), tokenizer.bos_token_id)
            tokens = np.concatenate([prefix, tokens], axis=1)
        return {'tokens':tokens}
    tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=4 if not streaming else None, remove_columns=[column_name])
    tokenized_dataset.set_format(type='torch', columns=['tokens'])
    return tokenized_dataset
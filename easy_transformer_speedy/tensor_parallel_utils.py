import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


def partition(n: int, rank: int, world_size: int, even_split: bool = True):
    """
    Partitions a sequence of n elements into a slice object representing that rank's partition. This
    is used for Tensor Parallel Modules, to help split them across GPUs.

    Args:
        n (int): The number of elements in the sequence to partition.
        rank (int): The rank of the current process.
        world_size (int): The number of processes.
        even_split (bool, optional): Whether the sequence should split evenly. Defaults to True.

    Returns:
        slice: A slice object representing the partition of the sequence
    """
    start, mod = divmod(n * rank, world_size)
    if even_split and mod != 0:
        raise ValueError(f"{n* rank} is not evenly divisible by {world_size}")
    end = n * (rank + 1) // world_size
    return slice(start, end)


def reduce(x: torch.Tensor, group: dist.ProcessGroup):
    """
    Reduces the tensor x across all processes. This is used for Tensor Parallel Modules, to help
    sum the parameters across GPUs.

    Args:
        x (torch.Tensor): The tensor to reduce.
        group (dist.ProcessGroup): The process group to reduce across.

    Returns:
        torch.Tensor: The reduced tensor.
    """
    if dist.get_world_size() == 1:
        return x
    dist.all_reduce(x, group=group)
    return x


def split(x: torch.Tensor):
    """
    Splits the tensor x along its last dimension across all processes. This is used for Tensor Parallel Modules, to help
    split the input across GPUs.

    Args:
        x (torch.Tensor): The tensor to split.
        group (dist.ProcessGroup): The process group to split across.

    Returns:
        torch.Tensor: The split tensor.
    """
    if dist.get_world_size() == 1:
        return x
    return x.chunk(dist.get_world_size(), dim=-1)[dist.get_rank()].contiguous()


def gather(x: torch.Tensor, group: dist.ProcessGroup):
    """
    Gathers the tensor x across all processes. This is used for Tensor Parallel Modules, to help
    gather the output across GPUs.

    Args:
        x (torch.Tensor): The tensor to gather.
        group (dist.ProcessGroup): The process group to gather across.

    Returns:
        torch.Tensor: The gathered tensor.
    """
    if dist.get_world_size() == 1:
        return x
    tensors = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    tensors[dist.get_rank()] = x
    dist.all_gather(tensors, x, group=group)
    return torch.cat(tensors, dim=-1).contiguous()


class Copy(torch.autograd.Function):
    """
    Copy the input tensor to the output tensor.
    """

    @staticmethod
    def symbolic(graph, x):
        return x

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, dy):
        return reduce(dy)

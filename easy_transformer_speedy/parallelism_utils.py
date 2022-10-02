import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

DATA_PARALLEL_GROUP = None
TENSOR_PARALLEL_GROUP = None
PIPELINE_PARALLEL_GROUP = None
EMBEDDING_GROUP = None


def initialize_parallel_groups(
    tensor_parallel_size: int = 1, pipeline_parallel_size: int = 1
):
    """
    Created the process groups for parallelized training/inference.

    Args:
        tensor_parallel_size (int, optional): The number of GPUs to use for tensor parallelism. Defaults to 1.
        pipeline_parallel_size (int, optional): The number of GPUs to use for pipeline parallelism. Defaults to 1.
    """
    assert dist.is_initialized(), "torch.distributed is not initialized"
    rank = dist.get_rank()
    if rank == 0:
        print(
            f"Initializing Parallel Groups: {tensor_parallel_size} GPUs for Tensor Parallelism, {pipeline_parallel_size} GPUs for Pipeline Parallelism"
        )

    world_size = dist.get_world_size()
    assert (
        world_size % tensor_parallel_size == 0
    ), f"world_size ({world_size}) must be divisible by tensor_parallel_size ({tensor_parallel_size})"
    assert (
        world_size % pipeline_parallel_size == 0
    ), f"world_size ({world_size}) must be divisible by pipeline_parallel_size ({pipeline_parallel_size})"

    data_parallel_size = world_size // (tensor_parallel_size * pipeline_parallel_size)

    num_data_parallel_groups = world_size // data_parallel_size
    num_tensor_parallel_groups = world_size // tensor_parallel_size
    num_pipeline_parallel_groups = world_size // pipeline_parallel_size

    global DATA_PARALLEL_GROUP, TENSOR_PARALLEL_GROUP, PIPELINE_PARALLEL_GROUP, EMBEDDING_GROUP

    data_parallel_group_ranks = []
    for t in range(tensor_parallel_size):
        start = t * num_tensor_parallel_groups
        end = start + num_tensor_parallel_groups
        for p in range(pipeline_parallel_size):
            group_ranks = list(range(start + p, end, pipeline_parallel_size))
            group = dist.new_group(group_ranks)
            data_parallel_group_ranks.append(group_ranks)
            if rank in group_ranks:
                DATA_PARALLEL_GROUP = group

    tensor_parallel_group_ranks = []
    for t in range(num_tensor_parallel_groups):
        group_ranks = list(
            range(t * tensor_parallel_size, (t + 1) * tensor_parallel_size)
        )
        group = dist.new_group(group_ranks)
        tensor_parallel_group_ranks.append(group_ranks)
        if rank in group_ranks:
            TENSOR_PARALLEL_GROUP = group

    pipeline_parallel_group_ranks = []
    for p in range(num_pipeline_parallel_groups):
        group_ranks = list(range(p, world_size, num_pipeline_parallel_groups))
        group = dist.new_group(group_ranks)
        pipeline_parallel_group_ranks.append(group_ranks)
        if rank in group_ranks:
            PIPELINE_PARALLEL_GROUP = group
        if len(group_ranks) > 1:
            embedding_ranks = [group_ranks[0], group_ranks[-1]]
        else:
            embedding_ranks = group_ranks
        group = dist.new_group(embedding_ranks)
        if rank in embedding_ranks:
            EMBEDDING_GROUP = group


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

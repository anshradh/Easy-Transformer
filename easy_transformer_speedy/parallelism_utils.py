import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

DATA_PARALLEL_GROUP = None
TENSOR_PARALLEL_GROUP = None
PIPELINE_PARALLEL_GROUP = None
EMBEDDING_GROUP = None
PROCESS_GROUPS_INITIALIZED = False


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
    global PROCESS_GROUPS_INITIALIZED
    PROCESS_GROUPS_INITIALIZED = True


def parallel_groups_initialized():
    """
    Returns whether or not the process groups have been initialized.

    Returns:
        bool: True if the process groups have been initialized, False otherwise.
    """
    return PROCESS_GROUPS_INITIALIZED


def get_data_parallel_rank():
    """
    Returns the rank of the current process in the data parallel group.

    Returns:
        int: The rank of the current process in the data parallel group.
    """
    global PROCESS_GROUPS_INITIALIZED, DATA_PARALLEL_GROUP
    assert PROCESS_GROUPS_INITIALIZED, "Process groups must be initialized"
    assert DATA_PARALLEL_GROUP is not None, "Data Parallel Group must be initialized"
    return dist.get_rank(group=DATA_PARALLEL_GROUP)


def get_tensor_parallel_rank():
    """
    Returns the rank of the current process in the tensor parallel group.

    Returns:
        int: The rank of the current process in the tensor parallel group.
    """
    global PROCESS_GROUPS_INITIALIZED, TENSOR_PARALLEL_GROUP
    assert PROCESS_GROUPS_INITIALIZED, "Process groups must be initialized"
    assert (
        TENSOR_PARALLEL_GROUP is not None
    ), "Tensor Parallel Group must be initialized"
    return dist.get_rank(group=TENSOR_PARALLEL_GROUP)


def get_pipeline_parallel_rank():
    """
    Returns the rank of the current process in the pipeline parallel group.

    Returns:
        int: The rank of the current process in the pipeline parallel group.
    """
    global PROCESS_GROUPS_INITIALIZED, PIPELINE_PARALLEL_GROUP
    assert PROCESS_GROUPS_INITIALIZED, "Process groups must be initialized"
    assert (
        PIPELINE_PARALLEL_GROUP is not None
    ), "Pipeline Parallel Group must be initialized"
    return dist.get_rank(group=PIPELINE_PARALLEL_GROUP)


def get_data_parallel_world_size():
    """
    Returns the world size of the data parallel group.

    Returns:
        int: The world size of the data parallel group.
    """
    global PROCESS_GROUPS_INITIALIZED, DATA_PARALLEL_GROUP
    assert PROCESS_GROUPS_INITIALIZED, "Process groups must be initialized"
    assert DATA_PARALLEL_GROUP is not None, "Data Parallel Group must be initialized"
    return dist.get_world_size(group=DATA_PARALLEL_GROUP)


def get_tensor_parallel_world_size():
    """
    Returns the world size of the tensor parallel group.

    Returns:
        int: The world size of the tensor parallel group.
    """
    global PROCESS_GROUPS_INITIALIZED, TENSOR_PARALLEL_GROUP
    assert PROCESS_GROUPS_INITIALIZED, "Process groups must be initialized"
    assert (
        TENSOR_PARALLEL_GROUP is not None
    ), "Tensor Parallel Group must be initialized"
    return dist.get_world_size(group=TENSOR_PARALLEL_GROUP)


def get_pipeline_parallel_world_size():
    """
    Returns the world size of the pipeline parallel group.

    Returns:
        int: The world size of the pipeline parallel group.
    """
    global PROCESS_GROUPS_INITIALIZED, PIPELINE_PARALLEL_GROUP
    assert PROCESS_GROUPS_INITIALIZED, "Process groups must be initialized"
    assert (
        PIPELINE_PARALLEL_GROUP is not None
    ), "Pipeline Parallel Group must be initialized"
    return dist.get_world_size(group=PIPELINE_PARALLEL_GROUP)


def get_data_parallel_group():
    """
    Returns the data parallel group for the current process.

    Returns:
        torch.distributed.ProcessGroup: The data parallel group.
    """
    global PROCESS_GROUPS_INITIALIZED, DATA_PARALLEL_GROUP
    assert PROCESS_GROUPS_INITIALIZED, "Process groups must be initialized"
    assert DATA_PARALLEL_GROUP is not None, "Data Parallel Group must be initialized"
    return DATA_PARALLEL_GROUP


def get_tensor_parallel_group():
    """
    Returns the tensor parallel group for the current process.

    Returns:
        torch.distributed.ProcessGroup: The tensor parallel group.
    """
    global PROCESS_GROUPS_INITIALIZED, TENSOR_PARALLEL_GROUP
    assert PROCESS_GROUPS_INITIALIZED, "Process groups must be initialized"
    assert (
        TENSOR_PARALLEL_GROUP is not None
    ), "Tensor Parallel Group must be initialized"
    return TENSOR_PARALLEL_GROUP


def get_pipeline_parallel_group():
    """
    Returns the pipeline parallel group for the current process.

    Returns:
        torch.distributed.ProcessGroup: The pipeline parallel group.
    """
    global PROCESS_GROUPS_INITIALIZED, PIPELINE_PARALLEL_GROUP
    assert PROCESS_GROUPS_INITIALIZED, "Process groups must be initialized"
    assert (
        PIPELINE_PARALLEL_GROUP is not None
    ), "Pipeline Parallel Group must be initialized"
    return PIPELINE_PARALLEL_GROUP


def get_embedding_group():
    """
    Returns the embedding group for the current process.

    Returns:
        torch.distributed.ProcessGroup: The embedding group.
    """
    global PROCESS_GROUPS_INITIALIZED, EMBEDDING_GROUP
    assert PROCESS_GROUPS_INITIALIZED, "Process groups must be initialized"
    assert EMBEDDING_GROUP is not None, "Embedding Group must be initialized"
    return EMBEDDING_GROUP


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


def reduce(x: torch.Tensor):
    """
    Reduces the tensor x across all processes. This is used for Tensor Parallel Modules, to help
    sum the parameters across GPUs.

    Args:
        x (torch.Tensor): The tensor to reduce.

    Returns:
        torch.Tensor: The reduced tensor.
    """
    global PROCESS_GROUPS_INITIALIZED, TENSOR_PARALLEL_GROUP
    assert PROCESS_GROUPS_INITIALIZED, "Process groups must be initialized"
    assert (
        TENSOR_PARALLEL_GROUP is not None
    ), "Tensor Parallel Group must be initialized"
    if dist.get_world_size() == 1:
        return x
    dist.all_reduce(x, group=TENSOR_PARALLEL_GROUP)
    return x


def split(x: torch.Tensor):
    """
    Splits the tensor x along its last dimension across all processes. This is used for Tensor Parallel Modules, to help
    split the input across GPUs.

    Args:
        x (torch.Tensor): The tensor to split.

    Returns:
        torch.Tensor: The split tensor.
    """
    global PROCESS_GROUPS_INITIALIZED, TENSOR_PARALLEL_GROUP
    assert PROCESS_GROUPS_INITIALIZED, "Process groups must be initialized"
    assert (
        TENSOR_PARALLEL_GROUP is not None
    ), "Tensor Parallel Group must be initialized"
    if dist.get_world_size() == 1:
        return x
    return x.chunk(dist.get_world_size(group=TENSOR_PARALLEL_GROUP), dim=-1)[
        dist.get_rank(group=TENSOR_PARALLEL_GROUP)
    ].contiguous()


def gather(x: torch.Tensor):
    """
    Gathers the tensor x across all processes. This is used for Tensor Parallel Modules, to help
    gather the output across GPUs.

    Args:
        x (torch.Tensor): The tensor to gather.

    Returns:
        torch.Tensor: The gathered tensor.
    """
    global PROCESS_GROUPS_INITIALIZED, TENSOR_PARALLEL_GROUP
    assert PROCESS_GROUPS_INITIALIZED, "Process groups must be initialized"
    assert (
        TENSOR_PARALLEL_GROUP is not None
    ), "Tensor Parallel Group must be initialized"
    if dist.get_world_size() == 1:
        return x
    tensors = [
        torch.empty_like(x)
        for _ in range(dist.get_world_size(group=TENSOR_PARALLEL_GROUP))
    ]
    tensors[dist.get_rank(group=TENSOR_PARALLEL_GROUP)] = x
    dist.all_gather(tensors, x, group=TENSOR_PARALLEL_GROUP)
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


class Reduce(torch.autograd.Function):
    """
    Reduce the input tensor across all processes.
    """

    @staticmethod
    def symbolic(graph, x):
        return reduce(x)

    @staticmethod
    def forward(ctx, x):
        return reduce(x)

    @staticmethod
    def backward(ctx, dy):
        return dy


class Split(torch.autograd.Function):
    """
    Split the input tensor across all processes.
    """

    @staticmethod
    def symbolic(graph, x):
        return split(x)

    @staticmethod
    def forward(ctx, x):
        return split(x)

    @staticmethod
    def backward(ctx, dy):
        return gather(dy)


class Gather(torch.autograd.Function):
    """
    Gather the input tensor across all processes.
    """

    @staticmethod
    def symbolic(graph, x):
        return gather(x)

    @staticmethod
    def forward(ctx, x):
        return gather(x)

    @staticmethod
    def backward(ctx, dy):
        return split(dy)

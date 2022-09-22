import torch
from torch import autograd
import torch.nn.functional as F

import triton
import triton.language as tl

from triton_modules.utils import num_warps_from_block_size


# Credit to https://github.com/lucidrains/triton-transformer/blob/main/triton_transformer/layernorm.py

GAMMA_BETA_BLOCK_SIZE = 64
GAMMA_BETA_ROW_BLOCK_SIZE = 64


@triton.jit
def layernorm_kernel_forward_training(
    output_ptr,
    mean_centered_ptr,
    normed_ptr,
    input_ptr,
    gamma_ptr,
    beta_ptr,
    input_row_stride,
    gamma_row_stride,
    beta_row_stride,
    output_row_stride,
    mean_centered_row_stride,
    normed_row_stride,
    n_cols,
    stable,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)

    row_start_ptr = input_ptr + row_idx * input_row_stride
    gamma_row_start_ptr = gamma_ptr + row_idx * gamma_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    gamma_ptrs = gamma_row_start_ptr + col_offsets
    beta_row_start_ptr = beta_ptr + row_idx * beta_row_stride

    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=0.0)
    gammas = tl.load(gamma_ptrs, mask=mask, other=0.0)
    betas = tl.load(beta_row_start_ptr, mask=mask, other=0.0)

    if stable:
        row_max = tl.max(tl.where(mask, row, float("-inf")), axis=0)
        row /= row_max

    row_mean = tl.sum(row, axis=0) / n_cols
    row_mean_centered = tl.where(mask, row - row_mean, 0.0)
    row_var = tl.sum(row_mean_centered * row_mean_centered, axis=0) / n_cols
    inv_var = 1.0 / tl.sqrt(row_var + eps)
    normed = row_mean_centered * inv_var

    output = normed * gammas + betas

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, output, mask=mask)

    mean_centered_row_start_ptr = mean_centered_ptr + row_idx * mean_centered_row_stride
    mean_centered_ptrs = mean_centered_row_start_ptr + col_offsets
    tl.store(mean_centered_ptrs, row_mean_centered, mask=mask)

    normed_row_start_ptr = normed_ptr + row_idx * normed_row_stride
    normed_ptrs = normed_row_start_ptr + col_offsets
    tl.store(normed_ptrs, normed, mask=mask)


@triton.jit
def layernorm_kernel_forward_inference(
    output_ptr,
    input_ptr,
    gamma_ptr,
    beta_ptr,
    input_row_stride,
    gamma_row_stride,
    beta_row_stride,
    output_row_stride,
    n_cols,
    stable,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)

    row_start_ptr = input_ptr + row_idx * input_row_stride
    gamma_row_start_ptr = gamma_ptr + row_idx * gamma_row_stride
    beta_row_start_ptr = beta_ptr + row_idx * beta_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    gamma_ptrs = gamma_row_start_ptr + col_offsets
    beta_ptrs = beta_row_start_ptr + col_offsets

    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=0.0)
    gammas = tl.load(gamma_ptrs, mask=mask, other=0.0)
    betas = tl.load(beta_ptrs, mask=mask, other=0.0)

    if stable:
        row_max = tl.max(tl.where(mask, row, float("-inf")), axis=0)
        row /= row_max

    row_mean = tl.sum(row, axis=0) / n_cols
    row_mean_centered = tl.where(mask, row - row_mean, 0.0)
    row_var = tl.sum(row_mean_centered * row_mean_centered, axis=0) / n_cols
    inv_var = 1.0 / tl.sqrt(row_var + eps)
    normed = row_mean_centered * inv_var

    output = normed * gammas + betas

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, output, mask=mask)


@triton.jit
def layernorm_kernel_backward(
    output_ptr,
    dy_ptr,
    mean_centered_ptr,
    output_row_stride,
    dy_row_stride,
    mean_centered_row_stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)

    dy_row_start_ptr = dy_ptr + row_idx * dy_row_stride
    mean_centered_row_start_ptr = mean_centered_ptr + row_idx * mean_centered_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    dy_ptrs = dy_row_start_ptr + col_offsets
    mean_centered_ptrs = mean_centered_row_start_ptr + col_offsets

    mask = col_offsets < n_cols

    dy = tl.load(dy_ptrs, mask=mask, other=0.0)
    mean_centered = tl.load(mean_centered_ptrs, mask=mask, other=0.0)

    row_var = tl.sum(mean_centered * mean_centered, axis=0) / n_cols
    inv_var = 1.0 / tl.sqrt(row_var + eps)
    normed = mean_centered * inv_var

    output = (
        1.0
        / n_cols
        * inv_var
        * (n_cols * dy - tl.sum(dy, axis=0) - normed * tl.sum(dy * normed, axis=0))
    )

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, output, mask=mask)


@triton.jit
def layernorm_gamma_beta_kernel_backward(
    dbeta_ptr,
    dgamma_ptr,
    norm_ptr,
    dy_ptr,
    norm_stride,
    dy_stride,
    dgamma_row_stride,
    dbeta_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_ROW: tl.constexpr,
):
    col_idx = tl.program_id(0)
    row_idx = tl.program_id(1)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    row_offsets = tl.arange(0, BLOCK_SIZE_ROW)

    col_range = col_idx * BLOCK_SIZE + col_offsets
    row_range = row_idx * BLOCK_SIZE_ROW + row_offsets

    col_mask = col_range < n_cols
    mask = (row_range < n_rows)[:, None] & col_mask[None, :]

    dy_ptr += row_range[:, None] * dy_stride + col_range[None, :]
    norm_ptr += row_range[:, None] * norm_stride + col_range[None, :]

    dy = tl.load(dy_ptr, mask=mask, other=0.0)
    norm = tl.load(norm_ptr, mask=mask, other=0.0)

    dbeta = tl.sum(dy, axis=0)
    dgamma = tl.sum(dy * norm, axis=0)

    dgamma_ptr += row_idx * dgamma_row_stride + col_range
    dbeta_ptr += row_idx * dbeta_row_stride + col_range

    tl.store(dgamma_ptr, dgamma, mask=col_mask)
    tl.store(dbeta_ptr, dbeta, mask=col_mask)


class _layernorm(autograd.Function):
    @classmethod
    def forward(cls, ctx, x, gamma, beta, eps, stable):
        shape = x.shape
        dim = shape[-1]
        x = x.view(-1, dim)
        n_rows, n_cols = x.shape

        expanded_gamma = gamma[None, :].expand(n_rows, -1)
        expanded_beta = beta[None, :].expand(n_rows, -1)

        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        num_warps = num_warps_from_block_size(BLOCK_SIZE)

        out = torch.empty_like(x)

        ctx.eps = eps

        if x.requires_grad:
            scaled_x = torch.empty_like(x)
            normed = torch.empty_like(x)

            layernorm_kernel_forward_training[(n_rows,)](  # type: ignore
                out,
                scaled_x,
                normed,
                x,
                expanded_gamma,
                expanded_beta,
                x.stride(0),
                expanded_gamma.stride(0),
                expanded_beta.stride(0),
                out.stride(0),
                scaled_x.stride(0),
                normed.stride(0),
                n_cols,
                stable,
                eps,
                num_warps=num_warps,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            ctx.save_for_backward(scaled_x, gamma, out)
        else:
            layernorm_kernel_forward_inference[(n_rows,)](  # type: ignore
                out,
                x,
                expanded_gamma,
                expanded_beta,
                x.stride(0),
                expanded_gamma.stride(0),
                expanded_beta.stride(0),
                out.stride(0),
                n_cols,
                stable,
                eps,
                num_warps=num_warps,
                BLOCK_SIZE=BLOCK_SIZE,
            )

        return out.view(*shape)

    @classmethod
    def backward(cls, ctx, dy):
        shape, device = dy.shape, dy.device
        dim = shape[-1]
        dy = dy.view(-1, dim)

        scaled_x, gamma, normed = ctx.saved_tensors

        n_rows, n_cols = dy.shape

        num_col_programs = triton.cdiv(n_cols, GAMMA_BETA_BLOCK_SIZE)
        num_row_programs = triton.cdiv(n_rows, GAMMA_BETA_ROW_BLOCK_SIZE)

        dgamma = torch.empty((num_row_programs, n_cols), device=device)
        dbeta = torch.empty((num_row_programs, n_cols), device=device)

        layernorm_gamma_kernel_backward[(num_col_programs, num_row_programs)](  # type: ignore
            dbeta,
            dgamma,
            normed,
            dy,
            normed.stride(0),
            dy.stride(0),
            dgamma.stride(0),
            n_rows,
            n_cols,
            num_warps=4,
            BLOCK_SIZE=GAMMA_BETA_BLOCK_SIZE,
            ROW_BLOCK_SIZE=GAMMA_BETA_ROW_BLOCK_SIZE,
        )

        dbeta = dbeta.sum(dim=0)
        dgamma = dgamma.sum(dim=0)

        dxhat = dy * gamma
        dx = torch.empty_like(dy)

        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        num_warps = num_warps_from_block_size(BLOCK_SIZE)

        layernorm_kernel_backward[(n_rows,)](  # type: ignore
            dx,
            dxhat,
            scaled_x,
            dx.stride(0),
            dxhat.stride(0),
            scaled_x.stride(0),
            n_cols,
            ctx.eps,
            num_warps=num_warps,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        dx = dx.view(*shape)
        return dx, dgamma, dbeta, None, None


def triton_layernorm(x, gamma, beta, eps=1e-5, stable=False):
    """
    Wrapper to return layernorm of x using triton kernels.

    Args:
        x (torch.Tensor): input tensor
        gamma (torch.Tensor): scale tensor
        beta (torch.Tensor): bias tensor
        eps (float): epsilon value for numerical stability
        use_triton (bool): whether to use triton or torch layernorm
        stable (bool): whether to use numerically stable layernorm

    Returns:
        torch.Tensor: layernorm of x
    """
    return _layernorm.apply(x, gamma, beta, eps, stable)

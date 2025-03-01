import torch
from torch.optim.optimizer import Optimizer
from typing import Optional, Tuple, Iterable
import triton
import triton.language as tl

from triton_modules.utils import num_warps_from_block_size


@triton.jit
def adam_update_kernel(
    params_ptr,
    params_grad_ptr,
    m_ptr,
    v_ptr,
    lr,
    beta1,
    beta2,
    beta1_t,
    beta2_t,
    eps,
    wd,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    This kernel implements the Adam(W) update rule in-place on the given parameters.

    Args:
        params_ptr (torch.Tensor): Pointer to the parameters.
        params_grad_ptr (torch.Tensor): Pointer to the parameters gradients.
        m_ptr (torch.Tensor): Pointer to the first moment.
        v_ptr (torch.Tensor): Pointer to the second moment.
        lr (float): Learning rate.
        beta1 (float): Exponential decay rate for the first moment estimates.
        beta2 (float): Exponential decay rate for the second moment estimates.
        beta1_t (float): beta1^t.
        beta2_t (float): beta2^t.
        eps (float): Term added to the denominator to improve numerical stability.
        wd (float): Weight decay. If > 0, weight decay is applied and AdamW is implemented.
        n_elements (int): Number of elements in the parameters.
        BLOCK_SIZE (int): Triton block size.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    param = tl.load(params_ptr + offsets, mask=mask)
    grad = tl.load(params_grad_ptr + offsets, mask=mask)
    m = tl.load(m_ptr + offsets, mask=mask)
    v = tl.load(v_ptr + offsets, mask=mask)

    g = grad

    # If weight_decay, implement AdamW update instead
    if wd != 0.0:
        param_ = param - lr * wd * param
    else:
        param_ = param

    m_output = beta1 * m + (1.0 - beta1) * g
    tl.store(m_ptr + offsets, m_output, mask)
    v_output = beta2 * v + (1.0 - beta2) * (g * g)
    tl.store(v_ptr + offsets, v_output, mask)

    m_hat = m_output / (1.0 - beta1_t)
    v_hat = v_output / (1.0 - beta2_t)

    param_output = param_ - lr * (m_hat / (tl.sqrt(v_hat) + eps))
    tl.store(params_ptr + offsets, param_output, mask)


class TritonAdam(Optimizer):
    """
    Implements Adam (or AdamW) using Triton (https://github.com/openai/triton).

    We unroll all parameters into a single buffer, and use a single kernel to update all parameters.
    Credit to Connor Kissane for this idea (https://ckkissane.github.io/fused_adam_post.html)

    Like the PyTorch versions, but assumes amsgrad=False and maximize=False

    If weight_decay != 0.0, identical to:
        https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
    Else:
        https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
    """

    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: Optional[float] = None,
    ):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        if weight_decay is None:
            weight_decay = 0.0
        self.wd = weight_decay
        self.t = 0

        self.total_n_elements = sum(p.numel() for p in self.params)

        self.param_buffer = self.params[0].new_zeros((self.total_n_elements))
        self.grad_buffer = self.params[0].new_zeros((self.total_n_elements))

        buffer_index = 0
        for p in self.params:
            n = p.numel()
            self.param_buffer[buffer_index : buffer_index + n] = p.data.view(-1)
            p.data = self.param_buffer[buffer_index : buffer_index + n].view(
                p.data.shape
            )
            p.grad = self.grad_buffer[buffer_index : buffer_index + n].view(
                p.data.shape
            )

            buffer_index += n

        self.param_buffer.grad = self.grad_buffer

        self.m = torch.zeros_like(self.grad_buffer)
        self.v = torch.zeros_like(self.grad_buffer)

    def zero_grad(self):
        if self.param_buffer.grad is not None:
            if self.param_buffer.grad.grad_fn is not None:
                self.param_buffer.grad.detach_()
            else:
                self.param_buffer.grad.requires_grad_(False)
            self.param_buffer.grad.zero_()

    @torch.inference_mode()
    def step(self):
        self.t += 1

        grid = lambda meta: (triton.cdiv(self.total_n_elements, meta["BLOCK_SIZE"]),)
        block_size = min(triton.next_power_of_2(self.total_n_elements), 2**12)
        adam_update_kernel[grid](  # type: ignore
            self.param_buffer,
            self.grad_buffer,
            self.m,
            self.v,
            self.lr,
            self.beta1,
            self.beta2,
            self.beta1**self.t,
            self.beta2**self.t,
            self.eps,
            self.wd,
            self.total_n_elements,
            num_warps=num_warps_from_block_size(block_size),
            BLOCK_SIZE=block_size,
        )

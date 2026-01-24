import torch
import triton
import triton.language as tl


@triton.jit
def silu_kernel(m_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(m_ptr + offsets, mask=mask)
    silu = x / (1 + tl.exp(-x))
    tl.store(output_ptr + offsets, silu, mask=mask)
    


def silu_triton(input):
    output = torch.empty_like(input)
    n_elements = input.numel()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    silu_kernel[grid](input, output, n_elements, 64)
    return output
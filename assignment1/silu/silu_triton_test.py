import torch
from silu_triton_kernel import silu_triton
import time

# Test the Triton kernel
torch.manual_seed(0)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

# 8192 x 8192 tensor
input_tensor = torch.randn(8192, 8192, device='cuda')

output_torch = torch.nn.functional.silu(input_tensor)

# warm up
output_triton = silu_triton(input_tensor)

start.record()
output_triton = silu_triton(input_tensor)
end.record()

# Verify correctness
if torch.allclose(output_torch, output_triton, atol=1e-6):
    print("Triton SiLU kernel is correct!")
else:
    print("Triton SiLU kernel is incorrect!")

elapsed_sec = start.elapsed_time(end) / 1000

print(f"Triton SiLU kernel time: {elapsed_sec:.6f} seconds")
print("Bandwidth: ", 2 * input_tensor.element_size() * input_tensor.numel() / elapsed_sec / 1e9, "GB/s")



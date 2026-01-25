import torch 
from silu_torch import silu_activation

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
matrix = torch.rand(8192, 8192, device="cpu")
local_matrix = matrix.to(device="cuda")




# warm-up
_ = silu_activation(local_matrix)
torch.cuda.synchronize()


start.record()
result = silu_activation(local_matrix)

end.record()
torch.cuda.synchronize()


print(result.cpu())

elapsed_sec = start.elapsed_time(end) / 1000

print(f"Triton SiLU kernel time: {elapsed_sec:.6f} seconds")
print("Bandwidth: ", 2 * local_matrix.element_size() * local_matrix.numel() / elapsed_sec / 1e9, "GB/s")




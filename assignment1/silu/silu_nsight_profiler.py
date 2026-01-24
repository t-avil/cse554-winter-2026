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
print("time", start.elapsed_time(end))

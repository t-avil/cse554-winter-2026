import torch
from torch.profiler import profile, record_function, ProfilerActivity
from silu_torch import silu_activation

if __name__ == "__main__":

    # Create the large matrix
    input_tensor = torch.randn(8192, 8192, device="cpu")

    input_tensor = input_tensor.to("cuda")
    
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        with_stack=True,
        record_shapes=True,
    ) as prof:
        with record_function("silu_activation"):
            silu_activation(input_tensor)
    
    prof.export_chrome_trace("torch_silu.json")


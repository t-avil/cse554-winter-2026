import numpy as np
import matplotlib.pyplot as plt
import torch
import flashinfer

#Reference config for each model
llama3_1b_config = {
    "hidden_size": 2048,
    "num_attention_heads": 32,
    "num_key_value_heads": 8
}

llama3_3b_config = {
    "hidden_size": 3072,
    "num_attention_heads": 24,
    "num_key_value_heads": 8
}

llama3_8b_config = {
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 8
}

# Sequence lengths (powers of 2)
SEQ_LEN = 1024
BATCH_SIZES = [2 ** x for x in range(7)]


def measure_tflops_sdpa_decode(seq_len, config, bs):
    # Safety check for memory limits
    if config['hidden_size'] == 4096 and bs == 64:
        return float('nan')
        
    d_h = config['hidden_size'] // config['num_attention_heads']
    h_qo = config['num_attention_heads']
    h_kv = config['num_key_value_heads']
    
    gen = torch.Generator(device='cuda')
    gen.manual_seed(42)
    
    # In decode, the query length is always 1
    # Shape: (batch_size, num_heads, seq_len_q, head_dim)
    q = torch.randn(bs, h_qo, 1, d_h, dtype=torch.float16, device='cuda', generator=gen)
    
    # K and V contain the full context of length seq_len
    # Shape: (batch_size, num_kv_heads, seq_len_kv, head_dim)
    k = torch.randn(bs, h_kv, seq_len, d_h, dtype=torch.float16, device='cuda', generator=gen)
    v = torch.randn(bs, h_kv, seq_len, d_h, dtype=torch.float16, device='cuda', generator=gen)

    num_iters = 50 # Decode is fast; more iters improve measurement stability
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(num_iters):
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v, enable_gqa=True)
    end.record()
    torch.cuda.synchronize()
    
    elapsed_time = start.elapsed_time(end) / 1000 / num_iters

    # Decode FLOPs Calculation:
    total_flop = 4 * bs * h_qo * seq_len * d_h
    tflops = total_flop / elapsed_time / 1e12
    
    return tflops


def measure_tflops_flashinfer_single_decode(seq_len, config, bs):
    if config['hidden_size'] == 4096 and bs == 64:
        return float('nan')
    print(bs)
    # Model configuration
    d_h = config['hidden_size'] // config['num_attention_heads']
    h_qo = config['num_attention_heads']
    h_kv = config['num_key_value_heads']
    
    gen = torch.Generator(device='cuda')
    gen.manual_seed(42)

    # In decode, Q is a single token: [num_qo_heads, head_dim]
    q = torch.randn(h_qo * bs, d_h, dtype=torch.float16, device='cuda', generator=gen)
    k = torch.randn(seq_len, h_kv * bs, d_h, dtype=torch.float16, device='cuda', generator=gen)
    v = torch.randn(seq_len, h_kv * bs, d_h, dtype=torch.float16, device='cuda', generator=gen)

    num_iters = 10
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(num_iters):
        # Result shape: [num_qo_heads, head_dim]
        o = flashinfer.decode.single_decode_with_kv_cache(q, k, v, kv_layout='NHD')
    end.record()
    
    torch.cuda.synchronize()
    elapsed_time = start.elapsed_time(end) / 1000 / num_iters # In seconds

    # Decode FLOPs calculation:
    total_flop = 4 * h_qo * seq_len * d_h * bs
    tflops = total_flop / elapsed_time / 1e12
    
    return tflops


# Generate fake compute utilization data
llama3_1b_sdpa = [measure_tflops_sdpa_decode(SEQ_LEN, llama3_1b_config, bs) for bs in BATCH_SIZES]
llama3_1b_flashinfer = [measure_tflops_flashinfer_single_decode(SEQ_LEN, llama3_1b_config, bs) for bs in BATCH_SIZES]

llama3_3b_sdpa = [measure_tflops_sdpa_decode(SEQ_LEN, llama3_3b_config, bs) for bs in BATCH_SIZES]
llama3_3b_flashinfer = [measure_tflops_flashinfer_single_decode(SEQ_LEN, llama3_3b_config, bs) for bs in BATCH_SIZES]

llama3_8b_sdpa = [measure_tflops_sdpa_decode(SEQ_LEN, llama3_8b_config, bs) for bs in BATCH_SIZES]
llama3_8b_flashinfer = [measure_tflops_flashinfer_single_decode(SEQ_LEN, llama3_8b_config, bs) for bs in BATCH_SIZES]

print("plotting")
# Plotting setup
fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
models = ['LLaMA3-1B', 'LLaMA3-3B', 'LLaMA3-8B']

# LLaMA2-7B plot
axs[0].plot(BATCH_SIZES, llama3_1b_sdpa, label='PyTorch SDPA', marker='o')
axs[0].plot(BATCH_SIZES, llama3_1b_flashinfer, label='FlashInfer', marker='x')
axs[0].set_xscale('log', base=2)
axs[0].set_title(models[0])
axs[0].set_xlabel('Batch Size')
axs[0].set_ylabel('Compute Utilization (TFLOPs)')
axs[0].set_xticks(BATCH_SIZES)
axs[0].set_xticklabels([str(p) for p in BATCH_SIZES])
axs[0].legend()
axs[0].grid(True, which='both')

# LLaMA3-8B plot
axs[1].plot(BATCH_SIZES, llama3_3b_sdpa, label='PyTorch SDPA', marker='o')
axs[1].plot(BATCH_SIZES, llama3_3b_flashinfer, label='FlashInfer', marker='x')
axs[1].set_xscale('log', base=2)
axs[1].set_title(models[1])
axs[1].set_xlabel('Batch Size')
axs[1].set_xticks(BATCH_SIZES)
axs[1].set_xticklabels([str(p) for p in BATCH_SIZES])
axs[1].legend()
axs[1].grid(True, which='both')

# LLaMA3-70B plot
axs[2].plot(BATCH_SIZES, llama3_8b_sdpa, label='PyTorch SDPA', marker='o')
axs[2].plot(BATCH_SIZES, llama3_8b_flashinfer, label='FlashInfer', marker='x')
axs[2].set_xscale('log', base=2)
axs[2].set_title(models[2])
axs[2].set_xlabel('Batch Size')
axs[2].set_xticks(BATCH_SIZES)
axs[2].set_xticklabels([str(p) for p in BATCH_SIZES])
axs[2].legend()
axs[2].grid(True, which='both')

# for i in range(3):
#     axs[i].set_yscale('log', base=2)

# Overall figure title and layout
fig.suptitle('Decode Compute Utilization by Batch Size', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('assignment3/Section2/viz/decode_attention_by_batch_size.png', dpi=300)

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
BATCH_SIZE = 128
PAGE_SIZES = [1, 2, 4, 8, 16]


def measure_tflops_flashinfer_paged_decode(seq_len, config, bs, page_size=16):
    print(page_size)
    d_h = config['hidden_size'] // config['num_attention_heads']
    h_qo = config['num_attention_heads']
    h_kv = config['num_key_value_heads']
    
    # 1. Calculate Paging Logic
    # Number of pages needed for one sequence
    num_pages_per_seq = (seq_len + page_size - 1) // page_size
    # Total pages across the whole batch
    total_pages = num_pages_per_seq * bs
    
    # 2. Setup metadata tensors (Must be int32 on CPU/GPU as per FlashInfer docs)
    # indptr: [batch_size + 1] -> [0, pages_per_seq, 2*pages_per_seq, ...]
    indptr = torch.arange(0, bs + 1, dtype=torch.int32) * num_pages_per_seq
    # indices: [total_pages] -> Physical IDs [0, 1, 2, ..., total_pages-1]
    indices = torch.arange(total_pages, dtype=torch.int32)
    # last_page_len: [batch_size] -> How many tokens in the very last page
    remainder = seq_len % page_size
    last_page_val = remainder if remainder > 0 else page_size
    last_page_len = torch.full((bs,), last_page_val, dtype=torch.int32)

    # 3. Initialize Tensors
    q = torch.randn(bs, h_qo, d_h, dtype=torch.float16, device='cuda')
    # FlashInfer Paged KV layout: [max_num_pages, 2, num_kv_heads, page_size, head_dim]
    kv_cache = torch.randn(total_pages, 2, h_kv, page_size, d_h, dtype=torch.float16, device='cuda')
    
    # 4. Initialize Wrapper
    workspace_buffer = torch.empty(128 * 1024 * 1024 * 8, dtype=torch.uint8, device='cuda')
    decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, "HND")
    
    # plan() expects indptr, indices, and last_page_len as tensors
    decode_wrapper.plan(
        indptr,
        indices,
        last_page_len,
        h_qo,
        h_kv,
        d_h,
        page_size,
        pos_encoding_mode="NONE",
        data_type=torch.float16
    )

    num_iters = 100
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(num_iters):
        o = decode_wrapper.run(q, kv_cache)
    end.record()
    
    torch.cuda.synchronize()
    elapsed_time = start.elapsed_time(end) / 1000 / num_iters

    # Decode FLOPs: 4 * batch * heads * seq_len * head_dim
    total_flop = 4 * bs * h_qo * seq_len * d_h
    tflops = total_flop / elapsed_time / 1e12
    
    return tflops


# Generate fake compute utilization data
llama3_1b_flashinfer = [measure_tflops_flashinfer_paged_decode(SEQ_LEN, llama3_1b_config, BATCH_SIZE, page_size=page_size) for page_size in PAGE_SIZES]

llama3_3b_flashinfer = [measure_tflops_flashinfer_paged_decode(SEQ_LEN, llama3_3b_config, BATCH_SIZE, page_size=page_size) for page_size in PAGE_SIZES]

llama3_8b_flashinfer = [measure_tflops_flashinfer_paged_decode(SEQ_LEN, llama3_8b_config, BATCH_SIZE, page_size=page_size) for page_size in PAGE_SIZES]


print("plotting")
# Plotting setup
fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
models = ['LLaMA3-1B', 'LLaMA3-3B', 'LLaMA3-8B']

# LLaMA2-7B plot
axs[0].plot(PAGE_SIZES, llama3_1b_flashinfer, label='FlashInfer', marker='x')
axs[0].set_xscale('log', base=2)
axs[0].set_title(models[0])
axs[0].set_xlabel('Page Size')
axs[0].set_ylabel('Compute Utilization (TFLOPs)')
axs[0].set_xticks(PAGE_SIZES)
axs[0].set_xticklabels([str(p) for p in PAGE_SIZES])
axs[0].legend()
axs[0].grid(True, which='both')

# LLaMA3-8B plot
axs[1].plot(PAGE_SIZES, llama3_3b_flashinfer, label='FlashInfer', marker='x')
axs[1].set_xscale('log', base=2)
axs[1].set_title(models[1])
axs[1].set_xlabel('Page Size')
axs[1].set_xticks(PAGE_SIZES)
axs[1].set_xticklabels([str(p) for p in PAGE_SIZES])
axs[1].legend()
axs[1].grid(True, which='both')

# LLaMA3-70B plot
axs[2].plot(PAGE_SIZES, llama3_8b_flashinfer, label='FlashInfer', marker='x')
axs[2].set_xscale('log', base=2)
axs[2].set_title(models[2])
axs[2].set_xlabel('Page Size')
axs[2].set_xticks(PAGE_SIZES)
axs[2].set_xticklabels([str(p) for p in PAGE_SIZES])
axs[2].legend()
axs[2].grid(True, which='both')

# for i in range(3):
#     axs[i].set_yscale('log', base=2)

# Overall figure title and layout
fig.suptitle('Decode Compute Utilization by Page Size', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('assignment3/Section2/viz/decode_attention_by_page_size.png', dpi=300)
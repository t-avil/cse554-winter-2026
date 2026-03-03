import numpy as np
import matplotlib.pyplot as plt

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

for model_name, config in zip(['LLaMA3-1B', 'LLaMA3-3B', 'LLaMA3-8B'],
                              [llama3_1b_config, llama3_3b_config, llama3_8b_config]):
    print(f"{model_name} Config:")
    xs = []
    ys = []
    for prefill_length in [2 ** i for i in range(7, 16)]:
        prefill_operational_intensity = (2 * config['num_attention_heads'] * prefill_length) / (config['num_attention_heads'] + config['num_key_value_heads'])
        xs.append(prefill_length)
        ys.append(prefill_operational_intensity)
    fig, ax = plt.subplots()
    ax.plot(xs, ys, marker='o')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    ax.set_title(f"{model_name} Prefill Operational Intensity")
    ax.set_xlabel('p')
    ax.set_ylabel('Operational Intensity (FLOPs/Byte)')
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{y:.2f}" for y in ys])
    ax.grid(True, which='both')
    fig.savefig(f"assignment3/Section2/viz/{model_name}_prefill_operational_intensity.png")


for model_name, config in zip(['LLaMA3-1B', 'LLaMA3-3B', 'LLaMA3-8B'],
                              [llama3_1b_config, llama3_3b_config, llama3_8b_config]):
    print(f"{model_name} Config:")
    xs = []
    ys = []
    for prefill_length in [2 ** i for i in range(7, 16)]:
        prefill_operational_intensity = (2 * config['num_attention_heads'] * prefill_length) / (config['num_key_value_heads'] * prefill_length + config['num_attention_heads'])
        xs.append(prefill_length)
        ys.append(prefill_operational_intensity)
    fig, ax = plt.subplots()
    ax.plot(xs, ys, marker='o')
    ax.set_xscale('log', base=2)
    ax.set_title(f"{model_name} Decode Operational Intensity")
    ax.set_xlabel('p')
    ax.set_ylabel('Operational Intensity (FLOPs/Byte)')
    # ax.set_yticks(ys)
    # ax.set_yticklabels([f"{y:.2f}" for y in ys])
    ax.grid(True, which='both')
    fig.savefig(f"assignment3/Section2/viz/{model_name}_decode_operational_intensity.png")



import matplotlib.pyplot as plt
import time
import torch
import sys
import os

try:
    from no_kv import Engine as NoKVEngine
    from single_batch import Engine as KVEngine
except ImportError:
    sys.path.append("assignment2")
    from no_kv import Engine as NoKVEngine
    from single_batch import Engine as KVEngine


def benchmark():
    input_len = 1024
    output_lengths = list(range(128, 2049, 128))
    
    print(f"Benchmarking with Input Length: {input_len}")
    
    # Initialize Engines
    print("Loading KV Cache Engine...")
    kv_engine = KVEngine()
    print("Loading No-KV Engine...")
    no_kv_engine = NoKVEngine()
    
    # Pre-compute a prompt with 1024 tokens
    input_prompt = "Hi, who are you and how are you doing today?" * 85 + "CSECE"
    
    # Get tokens
    tokens = kv_engine.tokenizer.encode(input_prompt)
    
    # print token number
    print(f"Token Number: {len(tokens)}")
    
    times_no_kv = []
    times_kv = []
    
    for out_len in output_lengths:
        print(f"Testing Output Length: {out_len} ...")
        
        # --- No KV Cache ---
        start_time = time.time()
        
        no_kv_engine.generate(input_prompt, rounds=out_len)
            
        times_no_kv.append(time.time() - start_time)
        print(f"  No KV: {times_no_kv[-1]:.4f}s")
        
        # --- With KV Cache ---
        start_time = time.time()
        
        # KV Engine resets/uses kv_cache internally in run(prefill=True)
        kv_engine.generate(input_prompt, rounds=out_len)
            
        times_kv.append(time.time() - start_time)
        print(f"  KV:    {times_kv[-1]:.4f}s")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(output_lengths, times_no_kv, marker='o', label='Without KV Cache')
    plt.plot(output_lengths, times_kv, marker='s', label='With KV Cache')
    plt.xlabel("Output Length (tokens)")
    plt.ylabel("Generation Time (seconds)")
    plt.title("End-to-End Generation Time: KV Cache vs No KV Cache\n(Input Length ≈ 1024)")
    plt.legend()
    plt.grid(True)
    
    output_file = "benchmark_comparison.png"
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    benchmark()

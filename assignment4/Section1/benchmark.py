import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import time
import random
import numpy as np
import torch

from continous_engine import Engine as ContEngine
from continous_scheduler import Scheduler as ContScheduler, InputRequest as ContInputRequest

from chunked_engine import Engine as ChunkedEngine
from chunked_scheduler import Scheduler as ChunkedScheduler, InputRequest as ChunkedInputRequest

def generate_requests(num_reqs):
    reqs = []
    # input lognormal with u=6.0, sigma=0.7
    il_samples = np.random.lognormal(6.0, 0.7, size=num_reqs)
    
    for il in il_samples:
        il = max(1, int(round(il)))
        # output uniform [1, 512]
        ol = random.randint(1, 512)
        
        il = min(il, 5000) 
        
        prompt = " ".join(["A"] * il)
        reqs.append((prompt, ol))
    return reqs

def run_continuous(reqs, req_batch_size):
    import gc
    engine = ContEngine()
    scheduler = ContScheduler(engine, req_batch_size=req_batch_size)
    iteration_times = []
    start_time = time.time()
    for prompt, ol in reqs:
        scheduler.add_req(ContInputRequest(prompt, output_len=ol))
        iter_start = time.time()
        scheduler.run()
        iteration_times.append(time.time() - iter_start)

    while not scheduler.finished():
        iter_start = time.time()
        scheduler.run()
        iteration_times.append(time.time() - iter_start)
    elapsed = time.time() - start_time
    del scheduler
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return elapsed, iteration_times

def run_chunked(reqs, token_batch_size):
    import gc
    engine = ChunkedEngine()
    scheduler = ChunkedScheduler(engine, token_batch_size=token_batch_size)
    iteration_times = []
    start_time = time.time()
    for prompt, ol in reqs:
        scheduler.add_req(ChunkedInputRequest(prompt, output_len=ol))
        iter_start = time.time()
        scheduler.run()
        iteration_times.append(time.time() - iter_start)
    while not scheduler.finished():
        iter_start = time.time()
        scheduler.run()
        iteration_times.append(time.time() - iter_start)
    elapsed = time.time() - start_time
    del scheduler
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return elapsed, iteration_times

def main():
    num_requests = 100
    
    cont_times = []
    chunked_times = []
    
    for run in range(3):
        print(f"\n--- Run {run+1} ---")
        random.seed(42 + run)
        np.random.seed(42 + run)
        
        reqs = generate_requests(num_requests)
        
        # Run Chunked Prefill
        print("Running Chunked Prefill...")
        try:
            ch_time, ch_iters = run_chunked(reqs, token_batch_size=512)
            print(f"Chunked Prefill Time: {ch_time:.4f} s")
            chunked_times.append(ch_time)
        except Exception as e:
            print(f"Chunked Failed: {e}")
            chunked_times.append(float('nan'))
            ch_iters = []
            
        # Run Continuous Batching
        print("Running Continuous Batching...")
        try:
            cb_time, cb_iters = run_continuous(reqs, req_batch_size=128)
            print(f"Continuous Batching Time: {cb_time:.4f} s")
            cont_times.append(cb_time)
        except Exception as e:
            print(f"Continuous Failed: {e}")
            cont_times.append(float('nan'))
            cb_iters = []

        if run == 2: # Plot on the last run
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.scatter(range(len(ch_iters)), ch_iters, label='Chunked Prefill', alpha=0.6, s=10)
            plt.scatter(range(len(cb_iters)), cb_iters, label='Continuous Batching', alpha=0.6, s=10)
            plt.title('Iteration Time: Continuous Batching vs Chunked Prefill')
            plt.xlabel('Iteration ID')
            plt.ylabel('Time (s)')
            plt.legend()
            plt.grid(True)
            plt.savefig('iteration_times.png')
            print("Iteration time plot saved to iteration_times.png")

    avg_chunked = float(np.nanmean(chunked_times))
    avg_cont = float(np.nanmean(cont_times))
    
    print("\n=== SUMMARY ===")
    print(f"Chunked Prefill Times: {chunked_times}")
    print(f"Continuous Batching Times: {cont_times}")
    print(f"Average Chunked Prefill Time: {avg_chunked:.4f} s")
    print(f"Average Continuous Batching Time: {avg_cont:.4f} s")
    if avg_cont > 0 and not np.isnan(avg_cont) and not np.isnan(avg_chunked):
        print(f"Chunked Speedup: {avg_cont / avg_chunked:.4f}x")

if __name__ == '__main__':
    main()

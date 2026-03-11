import time
import random
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from continous_engine import Engine
from continous_scheduler import Scheduler, InputRequest

def generate_requests(num_reqs):
    reqs = []
    # input uniformly [1, 10], output uniformly [1, 128]
    for _ in range(num_reqs):
        il = random.randint(1, 10)
        ol = random.randint(1, 128)
        # prompt string of `il` words
        prompt = " ".join(["A"] * il)
        reqs.append((prompt, ol))
    return reqs

def run_naive(engine, reqs, batch_size=10):
    scheduler = Scheduler(engine, req_batch_size=batch_size)
    
    start_time = time.time()
    # In naive scheduling, we wait for a batch to completely finish before pulling the next
    for i in range(0, len(reqs), batch_size):
        batch_reqs = reqs[i:i+batch_size]
        for prompt, ol in batch_reqs:
            scheduler.add_req(InputRequest(prompt, output_len=ol))
        
        while not scheduler.finished():
            scheduler.run()
            
    end_time = time.time()
    return end_time - start_time

def run_continuous(engine, reqs, batch_size=10):
    scheduler = Scheduler(engine, req_batch_size=batch_size)
    
    start_time = time.time()
    # In continuous batching, we add all requests to pending; scheduler pulls them in eagerly
    for prompt, ol in reqs:
        scheduler.add_req(InputRequest(prompt, output_len=ol))
        
    while not scheduler.finished():
        scheduler.run()
        
    end_time = time.time()
    return end_time - start_time

def main():
    engine = Engine()
    
    num_requests = 100
    batch_size = 10
    
    # Reproducible requests so comparisons are perfectly aligned
    random.seed(42)
    print("Generating requests...")
    reqs = generate_requests(num_requests)
    
    print("Warming up engine...")
    scheduler = Scheduler(engine, req_batch_size=2)
    scheduler.add_req(InputRequest("test warm up", output_len=5))
    while not scheduler.finished():
        scheduler.run()

    print("Profiling Naive Scheduling...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        with_stack=False, # Keep false to avoid crashing/hanging the profiler trace write
        record_shapes=True,
    ) as prof_naive:
        with record_function("naive_scheduling"):
            naive_time = run_naive(engine, reqs, batch_size)
    
    print(f"Naive Time: {naive_time:.4f} seconds")

    print("Profiling Continuous Scheduling...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        with_stack=False,
        record_shapes=True,
    ) as prof_cont:
        with record_function("continuous_scheduling"):
            continuous_time = run_continuous(engine, reqs, batch_size)
            
    print(f"Continuous Time: {continuous_time:.4f} seconds")
    
    speedup = naive_time / continuous_time
    print(f"Speedup: {speedup:.4f}x")

if __name__ == "__main__":
    main()

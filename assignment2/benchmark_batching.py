from uniform_prefill import Engine as UniformPrefillEngine
import matplotlib.pyplot as plt
import torch

def main():
    engine = UniformPrefillEngine()
    prompt = """CSE 554: Systems for Machine Learning - Assignment 1
In this assignment, you will practice GPU programming using PyTorch, Triton, and CUDA, along with profiling techniques. The assignment consists of three parts: SiLU, RMS Norm, and Host-GPU Communication. Your submission will be in the form of a .zip file uploaded to Canvas and must include a short report with answers to the following questions (.pdf), code, and profiling results. For performance-related questions, you will receive 0 points if your kernel is incorrect. If your kernel is correct, your score will be determined linearly based on its performance. You will receive a full grade if your kernel meets or exceeds the performance standard, which is around 60-70% performance of our reference kernels.
The template code can be found at https://github.com/efeslab/cse554-winter-2026
FAQ:
Profiling:
Memory accesses are defined as the number of loads and stores to the GPU global memory (HBM) needed to carry out the computation. We expect you to theoretically infer the minimum number of memory accesses needed from the input and output tensor shape, datatype, and the operation being performed. This excludes any memcpy (host-to-device, device-to-host, device-to-device) operations.


To compute memory bandwidth utilisation, determine the minimum number of memory accesses needed for the computation and divide it by total execution time. If your implementation calls multiple kernels to run the computation, the execution time should include the total time from the first kernel's launch until the last kernel's execution. This would account for the kernel launch overhead (and overhead of any additional memory copies / allocations needed).


For accurate kernel execution time measurements use torch.cuda.Event() and torch.cuda.synchronize(). 


Reference kernels are our own implementation which we use as a baseline for grading. You do not need to match the reference kernel's performance exactly. The target bandwidths provided in each part are intended to be guidelines to indicate what an optimized implementation can achieve. Achieving performance in the expected range (that is, close to the target bandwidth) along with correctness is sufficient for credit.


Since CSE servers are being shared among groups, it is possible that the GPU resources are contended, leading to inaccurate measurements. Make sure to check nvidia-smi and make adjustments to CUDA_VISIBLE_DEVICES as needed to use the right GPUs assigned to your group.


For torch profiler, collecting profiles from the last cycle should be sufficient.


Correctness:
To determine kernel's correctness, measure"""
    NUM_ITERS = 20
    xs = [2 ** i for i in range(7)]
    ys_time = []
    ys_throughput = []
    for x in xs:
        cur_prompts = [prompt for _ in range(x)]
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(NUM_ITERS):
            engine.generate_batched(cur_prompts, rounds=128)
        end.record()
        torch.cuda.synchronize()
        each_iter_time = start.elapsed_time(end) / 1000 / NUM_ITERS
        print(f"Batch size {x}, time taken {each_iter_time} second.")
        ys_time.append(each_iter_time)
        throughput = x * 128 / each_iter_time
        print(f"Batch size {x}, throughput {throughput} token/s.")
        ys_throughput.append(throughput)
    
    # -------- Plot: Time vs Batch Size --------
    plt.figure()
    plt.plot(xs, ys_time, marker='o')
    plt.xscale('log', base=2)
    plt.xlabel("Batch Size")
    plt.ylabel("Average Generation Time per Call (seconds)")
    plt.title("LLM Generation Latency vs Batch Size")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("time_vs_batch.png")
    plt.close()

    # -------- Plot: Throughput vs Batch Size --------
    plt.figure()
    plt.plot(xs, ys_throughput, marker='o')
    plt.xscale('log', base=2)
    plt.xlabel("Batch Size")
    plt.ylabel("Throughput (tokens / second)")
    plt.title("LLM Generation Throughput vs Batch Size")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("throughput_vs_batch.png")
    plt.close()

if __name__ == '__main__':
    main()

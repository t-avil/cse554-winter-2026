import torch
import numpy as np
import matplotlib.pyplot as plt
from flashinfer_pipeline import Engine

# used gpt to help out with the charts here

def cuda_time_ms(func):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    func()
    end.record()

    torch.cuda.synchronize()
    return start.elapsed_time(end)


def build_fixed_length_prompts(batch_size, token_length):
    base = "hello "
    prompts = []
    for _ in range(batch_size):
        text = base * token_length
        prompts.append(text)
    return prompts


def experiment_decode_scaling():
    batch_size = 32
    prefill_len = 256
    decode_lengths = [2 ** i for i in range(5, 11)]

    total_times = []

    for decode_len in decode_lengths:
        engine = Engine()

        prompts = build_fixed_length_prompts(batch_size, prefill_len)

        def run():
            engine.generate_batched(prompts, rounds=decode_len)

        total_time = cuda_time_ms(run)
        total_times.append(total_time)

        print(f"[Part1] Decode {decode_len}: {total_time:.2f} ms")
        del engine
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    plt.figure()
    plt.plot(np.log2(decode_lengths), total_times, marker="o")
    plt.xlabel("log2(Decode Length)")
    plt.ylabel("End-to-End Time (ms)")
    plt.title("Q2 Part1: Decode Scaling")
    plt.grid(True)
    plt.savefig("fig_q2_part1_end2end.png", dpi=300)
    plt.close()


def experiment_prefill_scaling():
    batch_size = 1
    prefill_lengths = [2 ** i for i in range(8, 15)]

    prefill_times = []

    for prefill_len in prefill_lengths:
        engine = Engine()

        prompts = build_fixed_length_prompts(batch_size, prefill_len)

        def run():
            engine.generate_batched(prompts, rounds=1)

        total_time = cuda_time_ms(run)
        prefill_times.append(total_time)

        print(f"[Part2] Prefill {prefill_len}: {total_time:.2f} ms")
        del engine
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    plt.figure()
    plt.plot(np.log2(prefill_lengths), prefill_times, marker="o")
    plt.xlabel("log2(Prefill Length)")
    plt.ylabel("Prefill Time (ms)")
    plt.title("Q2 Part2: Prefill Scaling")
    plt.grid(True)
    plt.savefig("fig_q2_part2_prefill_scaling.png", dpi=300)
    plt.close()


def experiment_batch_scaling():
    batch_sizes = [2 ** i for i in range(0, 9)]
    prefill_len = 128
    decode_len = 128

    total_times = []
    throughputs = []

    for batch_size in batch_sizes:
        engine = Engine()

        prompts = build_fixed_length_prompts(batch_size, prefill_len)

        def run():
            engine.generate_batched(prompts, rounds=decode_len)

        total_time = cuda_time_ms(run)
        total_times.append(total_time)

        total_tokens = batch_size * (prefill_len + decode_len)
        throughput = total_tokens / (total_time / 1000.0)
        throughputs.append(throughput)

        print(f"[Part3] Batch {batch_size}: {total_time:.2f} ms")
        del engine
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    plt.figure()
    plt.plot(np.log2(batch_sizes), total_times, marker="o")
    plt.xlabel("log2(Batch Size)")
    plt.ylabel("End-to-End Time (ms)")
    plt.title("Q2 Part3: Batch Scaling (Latency)")
    plt.grid(True)
    plt.savefig("fig_q2_part3_latency_scaling.png", dpi=300)
    plt.close()

    plt.figure()
    plt.plot(np.log2(batch_sizes), throughputs, marker="o")
    plt.xlabel("log2(Batch Size)")
    plt.ylabel("Throughput (tokens/sec)")
    plt.title("Q2 Part3: Batch Scaling (Throughput)")
    plt.grid(True)
    plt.savefig("fig_q2_part3_throughput_scaling.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    experiment_decode_scaling()
    experiment_prefill_scaling()
    experiment_batch_scaling()
    print("All figures saved.")
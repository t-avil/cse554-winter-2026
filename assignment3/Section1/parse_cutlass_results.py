import pandas as pd
import os
import glob

rows = []
result_dir = "cutlass_results"

for csv_file in sorted(glob.glob(os.path.join(result_dir, "*.gemm.csv"))):
    try:
        df = pd.read_csv(csv_file, dtype=str)
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        continue

    df.columns = df.columns.str.strip()
    print(f"File: {csv_file}, {len(df)} rows, columns: {list(df.columns[:5])}")

    # Filter only passed runs
    if 'Disposition' in df.columns:
        df = df[df['Disposition'].astype(str).str.strip().str.lower() == 'passed']
        print(f"  After filtering: {len(df)} passed rows")

    for _, row in df.iterrows():
        try:
            M = int(row['m'])
            N = int(row['n'])
            K = int(row['k'])
            split_k = int(row['split_k_slices'])
            kernel = str(row['Operation'])
            gflops = float(row['GFLOPs'])
            tflops = gflops / 1000.0

            if tflops > 0:
                rows.append({'batch_size': M, 'N': N, 'K': K,
                             'split_k': split_k, 'kernel': kernel, 'tflops': tflops})
        except (ValueError, KeyError) as e:
            print(f"  Skipping row: {e}")
            continue

if len(rows) == 0:
    print("No valid results parsed!")
    exit(1)

result_df = pd.DataFrame(rows)
print(f"\nTotal parsed rows: {len(result_df)}")

# Pick best TFLOPS per (M, N, K) across all kernels and split_k values
best_idx = result_df.groupby(['batch_size', 'N', 'K'])['tflops'].idxmax()
best_df = result_df.loc[best_idx].copy()

print("\n=== Best kernels per shape ===")
for _, row in best_df.iterrows():
    print(f"  M={int(row['batch_size'])}, N={int(row['N'])}, K={int(row['K'])} "
          f"-> split_k={int(row['split_k'])}, tflops={row['tflops']:.3f}, kernel={row['kernel']}")

# Append cutlass rows to gemm_perf.csv
best_df['library'] = 'cutlass'
out_df = best_df[['batch_size', 'N', 'K', 'library', 'tflops']]
out_df.to_csv('gemm_perf.csv', mode='a', header=False, index=False)

print(f"\nAppended {len(out_df)} cutlass rows to gemm_perf.csv")
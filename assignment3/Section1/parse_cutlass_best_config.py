import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

rows = []
result_dir = "cutlass_results"

for csv_file in sorted(glob.glob(os.path.join(result_dir, "*.gemm.csv"))):
    try:
        df = pd.read_csv(csv_file, dtype=str)
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        continue

    df.columns = df.columns.str.strip()
    print(f"File: {csv_file}, {len(df)} rows")

    if 'Disposition' in df.columns:
        df = df[df['Disposition'].astype(str).str.strip().str.lower() == 'passed']

    for _, row in df.iterrows():
        try:
            rows.append({
                'batch_size': int(row['m']),
                'N': int(row['n']),
                'K': int(row['k']),
                'split_k': int(row['split_k_slices']),
                'kernel': str(row['Operation']),
                'op_class': str(row['op_class']),
                'cta_m': int(row['cta_m']),
                'cta_n': int(row['cta_n']),
                'cta_k': int(row['cta_k']),
                'gflops': float(row['GFLOPs']),
                'tflops': float(row['GFLOPs']) / 1000.0,
            })
        except (ValueError, KeyError):
            continue

result_df = pd.DataFrame(rows)

# Pick best TFLOPS per (M, N, K) across all kernels and split_k values
best_idx = result_df.groupby(['batch_size', 'N', 'K'])['tflops'].idxmax()
best_df = result_df.loc[best_idx].copy()

# Compute number of output tiles for best kernel
best_df['num_tiles'] = ((best_df['batch_size'] + best_df['cta_m'] - 1) // best_df['cta_m']) * \
                       ((best_df['N'] + best_df['cta_n'] - 1) // best_df['cta_n'])

print("\n=== Best kernel configs ===")
for _, row in best_df.sort_values(['N', 'K', 'batch_size']).iterrows():
    print(f"  M={int(row['batch_size']):>5}, N={int(row['N']):>5}, K={int(row['K']):>5} "
          f"| tile={int(row['cta_m'])}x{int(row['cta_n'])}x{int(row['cta_k'])} "
          f"| split_k={int(row['split_k'])} "
          f"| tiles={int(row['num_tiles'])} "
          f"| tflops={row['tflops']:.3f} "
          f"| {row['kernel']}")

# Get unique (N, K) shapes
shapes = best_df[['N', 'K']].drop_duplicates().values.tolist()
os.makedirs('plots', exist_ok=True)

colors = {1: 'blue', 2: 'orange', 4: 'green', 8: 'red'}

for N, K in shapes:
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    shape_all = result_df[(result_df['N'] == N) & (result_df['K'] == K)]
    shape_best = best_df[(best_df['N'] == N) & (best_df['K'] == K)].sort_values('batch_size')

    # --- Top: TFLOPS by split_k, best marked ---
    for sk in [1, 2, 4, 8]:
        sk_df = shape_all[shape_all['split_k'] == sk]
        if len(sk_df) == 0:
            continue
        sk_best_idx = sk_df.groupby('batch_size')['tflops'].idxmax()
        sk_best = sk_df.loc[sk_best_idx].sort_values('batch_size')
        ax1.plot(sk_best['batch_size'], sk_best['tflops'],
                 marker='o', color=colors[sk], label=f'split_k={sk}', linewidth=2, alpha=0.7)
    ax1.scatter(shape_best['batch_size'], shape_best['tflops'],
                marker='*', s=200, color='black', zorder=5, label='Best overall')
    ax1.set_ylabel('TFLOPS')
    ax1.set_title(f'CUTLASS Kernel Analysis: N={int(N)}, K={int(K)}')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- Middle: Best split_k chosen ---
    bar_width = 50
    ax2.bar(shape_best['batch_size'], shape_best['split_k'], width=bar_width,
            color='steelblue', edgecolor='black')
    ax2.set_ylabel('Best split_k')
    ax2.set_yticks([1, 2, 4, 8])
    ax2.grid(True, alpha=0.3)

    # --- Bottom: Number of output tiles and tiles*split_k ---
    x = np.array(shape_best['batch_size'].values, dtype=float)
    num_bars = len(x)
    # Compute bar width based on spacing between x values
    bar_w = min(np.diff(x).min(), 128) * 0.35 if num_bars > 1 else 50
    offset = bar_w / 2 + 2

    ax3.bar(x - offset, shape_best['num_tiles'].values, width=bar_w,
            color='lightblue', edgecolor='black', label='Output tiles')
    ax3.bar(x + offset, (shape_best['num_tiles'] * shape_best['split_k']).values, width=bar_w,
            color='lightsalmon', edgecolor='black', label='Tiles × split_k')
    ax3.set_xlabel('M (batch_size)')
    ax3.set_ylabel('Number of tiles')
    ax3.set_xticks(range(128, 2049, 128))
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(f'plots/cutlass_analysis_N{int(N)}_K{int(K)}.png', dpi=150)
    plt.close(fig)
    print(f"Saved plots/cutlass_analysis_N{int(N)}_K{int(K)}.png")

# Save best config CSV
best_df.sort_values(['N', 'K', 'batch_size']).to_csv('cutlass_best_configs.csv', index=False)
print("Saved cutlass_best_configs.csv")
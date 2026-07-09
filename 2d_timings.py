"""
Benchmarks the local eigenvalue solve time in `ed.run_fno_data_generation`
for the bubble_coeff test case across several combinations of the
fine-mesh resolution `ny`, overlap `ol`, and oversampling `os`.

For each combination the per-sample eigsh solve time is collected over
`num_samples` samples and stored as a numpy array
    eig_solve_times_{ny}_{os}_{ol}.npy
inside `OUT_DIR/times`.
"""

import experiments_driver as ed
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Fixed configuration
store_tag = "bubble_coeff"
num_samples = 50
nloc = 5
deg = 1
Ny = 4
rho = 0.0
subdom_idx = 5

# Output location
BASE_DIR = Path(
    "/fs/scratch/rb_bd_dlp_rng_dl01_cr_MSO_employees/students/nuj7rng/msgfem_data/fenics_out_data"
)
OUT_DIR = BASE_DIR / store_tag / f"samples_{num_samples}"
TIMES_DIR = OUT_DIR / "times"
TIMES_DIR.mkdir(parents=True, exist_ok=True)

# Directory for plots (workspace-relative)
PLOTS_DIR = Path(__file__).resolve().parent / "plots" / store_tag / f"samples_{num_samples}"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# (ny, ol, os) combinations to benchmark
ny_ol_os_combinations = [
    [2**8, 2, 2],
    [2**8, 4, 4],
    [2**8, 6, 6],
    [2**9, 2, 2],
    [2**9, 4, 4],
    [2**9, 6, 6],
    [2**10, 2, 2],
    [2**10, 4, 4],
    [2**10, 6, 6],
]

# Bubble parameter sampling (same as in ex_FNO.py so seeds line up)
x0 = 0.234375  # lower right corner of subdomain
y1 = 0.515625  # upper left corner
eps = 1e-08
np.random.seed(42)

num_bubbles = 5
side = y1 - x0
r_min = side / 50
r_max = side / 6
h_min = 1.0
h_max = 100.0

parameters = np.zeros((num_samples, 2 + num_bubbles * 4))
parameters[:, 0] = x0
parameters[:, 1] = y1
for s in range(num_samples):
    for b in range(num_bubbles):
        r = np.random.uniform(r_min, r_max)
        cx = np.random.uniform(x0 + r + eps, y1 - r - eps)
        cy = np.random.uniform(x0 + r + eps, y1 - r - eps)
        h = np.random.uniform(h_min, h_max)
        parameters[s, 2 + 4 * b : 2 + 4 * (b + 1)] = [cx, cy, r, h]

print(f"Starting 2D timing benchmark for case: {store_tag}")
print(f"Number of samples per config: {num_samples}")
print(f"Combinations (ny, ol, os): {ny_ol_os_combinations}")

for ny, ol, os_ in ny_ol_os_combinations:
    print(f"\n=== Config ny={ny}, ol={ol}, os={os_} ===", flush=True)
    eig_solve_times = np.zeros(num_samples)
    for idx, parameter in enumerate(parameters):
        print(f"Sample Nr.: {idx}", flush=True)
        _, _, _, _, _, eig_solve_time = ed.run_fno_data_generation(
            deg, Ny, ny, ol, os_, nloc, rho, store_tag, parameter, subdom_idx
        )
        eig_solve_times[idx] = eig_solve_time

    out_path = TIMES_DIR / f"eig_solve_times_{ny}_{os_}_{ol}.npy"
    np.save(out_path, eig_solve_times)
    print(
        f"Saved {out_path} | mean={eig_solve_times.mean():.4f}s "
        f"std={eig_solve_times.std():.4f}s",
        flush=True,
    )

print("\n2D timing benchmark completed!")

# --- Plotting: load stored timings and show mean +/- std per configuration ---
means = np.zeros(len(ny_ol_os_combinations))
stds = np.zeros(len(ny_ol_os_combinations))
labels = []
for i, (ny, ol, os_) in enumerate(ny_ol_os_combinations):
    times = np.load(TIMES_DIR / f"eig_solve_times_{ny}_{os_}_{ol}.npy")
    means[i] = times.mean()
    stds[i] = times.std()
    labels.append(f"ny={ny}, ol={ol}, os={os_}")

x = np.arange(len(ny_ol_os_combinations))
fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(x, means, yerr=stds, fmt="o-", capsize=5, capthick=1.2, linewidth=1.2)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_ylabel("eigsh solve time [s]")
ax.set_xlabel("configuration")
ax.set_title(
    f"Local eigsh solve time (mean $\\pm$ std) — {store_tag}, {num_samples} samples"
)
ax.grid(True, linestyle="--", alpha=0.5)
fig.tight_layout()

plot_path = PLOTS_DIR / "eig_solve_times_summary.png"
fig.savefig(plot_path, dpi=150)
print(f"Saved plot to {plot_path}")
plt.show()

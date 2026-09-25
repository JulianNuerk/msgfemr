"""
Benchmarks the local eigsh solve time for the 3D subdomain eigenproblem
implemented in `eigenproblem3d.computeSubdomain`, using the same bubble
coefficient as the 2D benchmark in 2d_timings.py.

The global cube is partitioned 3 x 3 x 3 and the interior subdomain (1,1,1) is
solved; the fine mesh has `nx = 3 * n_part` cells per direction, so `n_part` is
derived from the requested `nx`.

Runs as an LSF job array: each task handles one entry of `NX_LIST` and writes
`eig_solve_times_3d_nx{nx}.npy` of shape (num_samples,). A final `--merge` pass
stacks these into a (num_samples, len(NX_LIST)) array.

    python 3d_timings.py --index $LSB_JOBINDEX
    python 3d_timings.py --merge
"""

import argparse
from pathlib import Path

import numpy as np

NX_LIST = [2**4, 2**5, 2**6]

STORE_TAG = "bubble_coeff"
NUM_SAMPLES = 10
NLOC = 5
N_COARSE = 3
OL = 2
OS = 2
BOOL_RING = False

OUT_DIR = Path(__file__).resolve().parent / "times" / STORE_TAG


def build_parameters(num_samples, n_sub=4, seed=42):
    """One bubble per subdomain, layout (cx_local, cy_local, radius, height).

    Identical to 2d_timings.build_parameters so both benchmarks see the same
    coefficient realisations; in 3D the bubbles are extruded along z.
    """
    side = 1.0 / n_sub
    eps = 1e-08
    r_min, r_max = side / 50.0, side / 4.0
    h_min, h_max = 1.0, 1000.0

    rng = np.random.RandomState(seed)
    parameters = np.zeros((num_samples, 4 * n_sub * n_sub))
    for s in range(num_samples):
        for b in range(n_sub * n_sub):
            r = rng.uniform(r_min, r_max)
            cx_local = rng.uniform(r + eps, side - r - eps)
            cy_local = rng.uniform(r + eps, side - r - eps)
            h = rng.uniform(h_min, h_max)
            parameters[s, 4 * b : 4 * (b + 1)] = [cx_local, cy_local, r, h]
    return parameters


def timings_path(nx):
    return OUT_DIR / f"eig_solve_times_3d_nx{nx}.npy"


def run(nx, num_samples):
    import eigenproblem3d as ep3d
    import setup

    n_part = max(1, int(round(nx / N_COARSE)))
    nx_actual = N_COARSE * n_part

    parameters = build_parameters(num_samples)
    out_path = timings_path(nx)
    times = np.full(num_samples, np.nan)
    sizes = np.full(num_samples, np.nan)
    omega_os_shapes = np.full((num_samples, 3), np.nan)

    print(
        f"=== nx={nx} (realised {nx_actual}, n_part={n_part}), ol={OL}, os={OS}, "
        f"nloc={NLOC}, samples={num_samples} ===",
        flush=True,
    )
    for idx, parameter in enumerate(parameters):
        print(f"Sample Nr.: {idx}", flush=True)
        # bubble_coeff reads x[0], x[1] only, so the 2D bubbles extrude along z.
        coeff_A_function = lambda x, p=parameter: setup.bubble_coeff(x, p)
        timing, size_eigenproblem, [nx_real, ny_real, nz_real], _ = ep3d.computeSubdomain(
            n_part,
            NLOC,
            BOOL_RING,
            OL,
            OS,
            coeff_A_function=coeff_A_function,
            timing_iterations=1,
        )
        times[idx] = timing
        sizes[idx] = size_eigenproblem
        omega_os_shapes[idx] = [nx_real, ny_real, nz_real]
        # Checkpoint every sample so a wallclock/OOM kill does not lose the run.
        np.save(out_path, times)
    np.save(OUT_DIR / f"omega_os_shapes.npy", omega_os_shapes)
    np.save(OUT_DIR / f"eig_solve_MA_sizes.npy", sizes)

    print(
        f"Saved {out_path} | n={np.shape(times)[0]} MA size={np.nanmax(sizes):.0f} "
        f"mean={np.nanmean(times):.4f}s std={np.nanstd(times):.4f}s",
        flush=True,
    )


def merge():
    columns = [
        np.load(timings_path(nx)) if timings_path(nx).exists() else None
        for nx in NX_LIST
    ]
    n_rows = max((c.size for c in columns if c is not None), default=0)
    merged = np.full((n_rows, len(NX_LIST)), np.nan)
    for j, column in enumerate(columns):
        if column is None:
            print(f"Missing timings for nx={NX_LIST[j]}")
            continue
        merged[: column.size, j] = column

    out_path = OUT_DIR / "eig_solve_times_3d.npy"
    np.save(out_path, merged)
    print(f"Saved {out_path} with shape {merged.shape}")
    for j, nx in enumerate(NX_LIST):
        print(
            f"  nx={nx:5d}  mean={np.nanmean(merged[:, j]):10.4f}s"
            f"  std={np.nanstd(merged[:, j]):10.4f}s"
        )


def main():
    parser = argparse.ArgumentParser(description="3D eigsh timing benchmark")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--index", type=int, help="1-based LSF array index into NX_LIST")
    group.add_argument("--nx", type=int, choices=NX_LIST, help="run a single nx")
    group.add_argument("--merge", action="store_true", help="stack the per-nx files")
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.merge:
        merge()
        return

    if args.nx is not None:
        nx = args.nx
    elif 1 <= args.index <= len(NX_LIST):
        nx = NX_LIST[args.index - 1]
    else:
        parser.error(f"--index must be in 1..{len(NX_LIST)}, got {args.index}")

    run(nx, args.num_samples)


if __name__ == "__main__":
    main()

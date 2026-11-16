"""
ex_gmsfem.py
============

Command line entry point for GMsFEM experiments; run from the msgfemr root:

    python -m GMsFEM.ex_gmsfem --problem channel_rotated_coeff --num_samples 10 \
        --n_bf 5 10 --compare_msgfem --nloc 10

The parameters of the coefficient ``--problem`` (a store_tag of
``setup.FNO_coeffs``) are drawn with ``setup.sample_FNO_parameters``.
The errors are stored in ``--out`` (default ``errors/gmsfem/<problem>/errors.npz``).
"""

import os

# Every Ray task requests a single CPU; limit BLAS threads before numpy is imported.
for _thread_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_thread_var, "1")

import argparse

import numpy as np
import ray

import helper
import setup
from GMsFEM.gmsfem import DEFAULT_SIGMA
from GMsFEM.gmsfem_driver import COEFF_ELEMENTS, compute_errors_gmsfem


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GMsFEM solutions and energy errors for msgfemr problems.")
    parser.add_argument("--problem", type=str, required=True,
                        help="Coefficient, i.e. store_tag of setup.FNO_coeffs (e.g. bubble_coeff).")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of parameter samples.")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the parameter sampling.")
    parser.add_argument("--n_bf", type=int, nargs="+", default=[5],
                        help="Numbers of basis functions per local domain.")
    parser.add_argument("--deg", type=int, default=1)
    parser.add_argument("--Ny", type=int, default=4, help="Coarse cells per direction.")
    parser.add_argument("--ny", type=int, default=2 ** 8, help="Fine cells per direction.")
    parser.add_argument("--coeff_space", type=str, default="lagrange", choices=sorted(COEFF_ELEMENTS))
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA,
                        help="Shift of the local shift-invert eigensolver (< 0).")
    parser.add_argument("--compare_msgfem", action="store_true",
                        help="Also compute the MS-GFEM solution of every sample.")
    parser.add_argument("--nloc", type=int, default=5, help="MS-GFEM local basis functions.")
    parser.add_argument("--ol", type=int, default=2, help="MS-GFEM overlap.")
    parser.add_argument("--os", type=int, default=2, help="MS-GFEM oversampling.")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--plot_root", type=str, default=None,
                        help="Root for XDMF visualisations; no plots if omitted.")
    parser.add_argument("--num_plot_samples", type=int, default=3)
    args = parser.parse_args()

    if args.sigma >= 0:
        parser.error("--sigma must be negative (the Neumann matrix is singular).")

    np.random.seed(args.seed)
    parameters = setup.sample_FNO_parameters(args.problem, args.num_samples)
    print(f"Sampled {parameters.shape[0]} parameter samples for {args.problem}.", flush=True)

    if args.out is None:
        args.out = os.path.join("errors", "gmsfem", args.problem, "errors.npz")

    result = compute_errors_gmsfem(
        problem=args.problem,
        n_bf_list=args.n_bf,
        parameters=parameters,
        deg=args.deg,
        Ny=args.Ny,
        ny=args.ny,
        coeff_space=args.coeff_space,
        sigma=args.sigma,
        compare_msgfem=args.compare_msgfem,
        nloc=args.nloc,
        ol=args.ol,
        os_=args.os,
        plot_root=args.plot_root,
        num_plot_samples=args.num_plot_samples,
    )
    result["commit_hash"] = helper.get_commit_hash()
    result["parameters"] = parameters
    result["seed"] = args.seed

    out_parent = os.path.dirname(args.out)
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)
    np.savez(args.out, **result)
    print(f"Wrote errors to {args.out}")
    ray.shutdown()

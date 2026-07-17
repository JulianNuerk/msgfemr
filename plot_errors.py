"""
plot_errors.py
==============

Loads ``errors/<case>/errors.npz`` (as written by ``integrate_predictions.py``)
and produces three side-by-side histograms of the relative energy errors:

    err(uG, uh),  err(uG_pred, uh),  err(uG_pred, uG).

The figure is written to ``errors/<case>/error_histograms.png``.
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt


ERRORS = [
    ("error_uG_vs_uh",      r"$\mathrm{err}(u_G, u_h)$"),
    ("error_uG_pred_vs_uh", r"$\mathrm{err}(u_G^{\mathrm{pred}}, u_h)$"),
    ("error_uG_pred_vs_uG", r"$\mathrm{err}(u_G^{\mathrm{pred}}, u_G)$"),
]


def main():
    parser = argparse.ArgumentParser(
        description="Plot histograms of MS-GFEM prediction errors.",
    )
    parser.add_argument("--case", type=str, required=True,
                        help="Case tag; expects errors/<case>/errors.npz.")
    parser.add_argument("--bins", type=int, default=30)
    parser.add_argument("--log", action="store_true",
                        help="Use log-scaled x axis (recommended).")
    args = parser.parse_args()

    case_dir = os.path.join("errors", args.case)
    in_path = os.path.join(case_dir, "errors.npz")
    out_path = os.path.join(case_dir, "error_histograms.png")

    with np.load(in_path, allow_pickle=True) as d:
        arrays = {k: np.asarray(d[k]) for k, _ in ERRORS}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, (key, label) in zip(axes, ERRORS):
        vals = arrays[key]
        if args.log:
            vals = vals[vals > 0]
            bins = np.logspace(np.log10(vals.min()), np.log10(vals.max()),
                               args.bins)
            ax.set_xscale("log")
        else:
            bins = args.bins
        ax.hist(vals, bins=bins, edgecolor="black")
        ax.set_title(label)
        ax.set_xlabel("relative energy error")
        ax.set_ylabel("count")
        ax.axvline(np.median(vals), color="red", linestyle="--",
                   label=f"median = {np.median(vals):.2e}")
        ax.legend(loc="upper right", fontsize=9)

    fig.suptitle(f"case: {args.case}  (N = {len(arrays[ERRORS[0][0]])})")
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

"""
plot_gmsfem_errors.py
=====================

Mean relative energy errors of MS-GFEM and GMsFEM over nloc (= n_bf) for all
cases in a run directory written by GMsFEM/gmsfem.bsub, i.e.

    <run_dir>/<case>/nloc_<n>.npz

Run from the msgfemr root:

    python -m GMsFEM.plot_gmsfem_errors --run_dir errors/gmsfem/nloc_5_10
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

YLABEL = (r"$\mathbb{E}_{A_\xi}\left[ \frac{\| u_\xi - \hat{u}_\xi \|_{a, \Omega}}"
          r"{\|u_\xi\|_{a, \Omega}}\right]$")
MARKERS = {"Ms-GFEM": "o", "GMsFEM": "s"}
# Case directory name -> coefficient name used in the paper.
CASE_LABELS = {
    "bubble_coeff": "bubble",
    "channel_rotated_coeff": "channel",
    "channel_rotated_small_contrast_coeff": "channel-small",
    "channel_rotated_medium_contrast_coeff": "channel-medium",
    "channel_long_coeff": "channel-long",
    "multiscale_sincos_coeff": "sincos",
    "multiscale_sincos_hc_coeff": "sincos-hc",
}
# Cases not shown in the paper.
EXCLUDE = {"bubble_coeff"}


def load_case(case_dir):
    """nloc and (mean, std) of the errors of both methods, sorted by nloc."""
    nloc, err_msgfem, err_gmsfem = [], [], []
    for f in case_dir.glob("nloc_*.npz"):
        d = np.load(f, allow_pickle=True)
        nloc.append(int(d["nloc_msgfem"]))
        e_ms, e_gms = d["error_uG_vs_uh"], d["error_uGMs_vs_uh"][:, 0]
        err_msgfem.append((np.mean(e_ms), np.std(e_ms, ddof=1)))
        err_gmsfem.append((np.mean(e_gms), np.std(e_gms, ddof=1)))
    order = np.argsort(nloc)
    return np.array(nloc)[order], np.array(err_msgfem)[order], np.array(err_gmsfem)[order]


def plot_band(ax, nloc, err, color, marker, linestyle):
    """Mean line with a mean +/- std band; the lower edge is clipped for the log axis."""
    mean, std = err[:, 0], err[:, 1]
    ax.fill_between(nloc, np.maximum(mean - std, 0.1 * mean), mean + std,
                    color=color, alpha=0.15, linewidth=0)
    ax.semilogy(nloc, mean, color=color, marker=marker, linestyle=linestyle)


def plot_errors(run_dir, out=None):
    run_dir = Path(run_dir)
    cases = sorted(p for p in run_dir.iterdir()
                   if p.is_dir() and p.name not in EXCLUDE and any(p.glob("nloc_*.npz")))
    if not cases:
        raise FileNotFoundError(f"No <case>/nloc_*.npz files in {run_dir}")

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    case_handles = []
    all_nloc = set()
    for color, case_dir in zip(plt.cm.tab10.colors, cases):
        nloc, err_msgfem, err_gmsfem = load_case(case_dir)
        all_nloc.update(nloc.tolist())
        plot_band(ax, nloc, err_msgfem, color, MARKERS["Ms-GFEM"], "-")
        plot_band(ax, nloc, err_gmsfem, color, MARKERS["GMsFEM"], "--")
        case_handles.append(Line2D([], [], color=color, label=CASE_LABELS.get(case_dir.name, case_dir.name)))

    method_handles = [Line2D([], [], color="k", marker=MARKERS["Ms-GFEM"], linestyle="-", label="Ms-GFEM"),
                      Line2D([], [], color="k", marker=MARKERS["GMsFEM"], linestyle="--", label="GMsFEM")]
    case_legend = ax.legend(handles=case_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                            fontsize="small", title="coefficient")
    ax.add_artist(case_legend)
    ax.legend(handles=method_handles, loc="lower left", fontsize="small")

    ax.set_xticks(sorted(all_nloc))
    ax.set_xlabel(r"$n_{loc}$")
    ax.set_ylabel(YLABEL, fontsize=14)
    ax.set_title("Ms-GFEM & GMsFEM error decay")
    ax.grid(True, which="both", alpha=0.3)

    out = Path(out) if out else run_dir / "error_decay.pdf"
    fig.savefig(out, bbox_inches="tight", bbox_extra_artists=[case_legend])
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot MS-GFEM and GMsFEM error decay over nloc.")
    parser.add_argument("--run_dir", type=str, required=True, help="Directory containing <case>/nloc_<n>.npz.")
    parser.add_argument("--out", type=str, default=None, help="Output file (default <run_dir>/error_decay.pdf).")
    args = parser.parse_args()
    plot_errors(args.run_dir, args.out)

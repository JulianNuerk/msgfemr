"""
read_errors.py
==============

Collect the per-sample errors written by ``integrate_predictions.py`` and emit
a LaTeX table: one row per case, mean +- standard deviation over the samples.

Columns (keys as stored in ``errors.npz``):

    error_uG_vs_uh                -> e_{MsGFEM}      classical MS-GFEM vs. the
                                                     fine reference; independent
                                                     of the network, so it is
                                                     the floor the predicted
                                                     basis is judged against
    error_uG_pred_vs_uh           -> e_{pred}        hybrid solve vs. reference
    error_uG_pred_vs_uh_rel_L2    -> rel. L^2
    error_uG_pred_vs_uh_rel_H1    -> rel. H^1

Example::

    python read_errors.py --cases multiscale_sincos_hc_coeff \
        channel_rotation_circle_coeff --out tables/errors.tex
"""

import argparse
from pathlib import Path

import numpy as np

COLUMNS = [
    ("error_uG_vs_uh", r"$e_{\mathrm{MsGFEM}}$"),
    ("error_uG_pred_vs_uh", r"$e_{\mathrm{pred}}$"),
    ("error_uG_pred_vs_uh_rel_L2", r"rel. $L^2$"),
    ("error_uG_pred_vs_uh_rel_H1", r"rel. $H^1$"),
]

DEFAULT_CASES = [
    "multiscale_sincos_hc_coeff",
    "channel_rotation_circle_coeff",
    "channel_rotated_small_contrast_coeff",
    "channel_rotated_medium_contrast_coeff",
]


def _escape(name: str) -> str:
    """Underscores are subscript operators in LaTeX maths mode."""
    return name.replace("_", r"\_")


def load_case(errors_root: Path, case: str):
    """``(stats, n_samples)`` with ``stats[key] = (mean, std)``."""
    path = errors_root / case / "errors.npz"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found -- run integrate_pred.bsub for {case}")
    stats, n = {}, None
    with np.load(path, allow_pickle=True) as z:
        for key, _ in COLUMNS:
            if key not in z.files:
                raise KeyError(
                    f"{path} has no {key!r}; it was written by an older "
                    f"integrate_predictions.py. Available: {sorted(z.files)}"
                )
            v = np.asarray(z[key], dtype=float).ravel()
            # NaNs mean a sample failed to solve; drop them rather than letting
            # one failure turn the whole row into NaN.
            finite = v[np.isfinite(v)]
            if finite.size == 0:
                raise ValueError(f"{path}: {key} has no finite entries")
            stats[key] = (float(finite.mean()), float(finite.std()))
            n = finite.size if n is None else min(n, finite.size)
    return stats, n


def fmt(mean: float, std: float, digits: int, std_digits: int) -> str:
    return f"${mean:.{digits}f} \\pm {std:.{std_digits}f}$"


def build_table(rows, digits: int, std_digits: int, caption: str, label: str) -> str:
    header = " & ".join(["Case", r"$N$"] + [t for _, t in COLUMNS]) + r" \\"
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        rf"  \caption{{{caption}}}",
        rf"  \label{{{label}}}",
        r"  \begin{tabular}{l r " + " ".join(["c"] * len(COLUMNS)) + r"}",
        r"    \toprule",
        "    " + header,
        r"    \midrule",
    ]
    for case, stats, n in rows:
        cells = [fmt(*stats[key], digits, std_digits) for key, _ in COLUMNS]
        lines.append("    " + " & ".join([_escape(case), str(n)] + cells) + r" \\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="LaTeX table of MS-GFEM / FNO errors")
    p.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    p.add_argument("--errors_root", default="errors",
                   help="directory holding <case>/errors.npz")
    p.add_argument("--out", default=None, help="write the table here (default: stdout only)")
    p.add_argument("--digits", type=int, default=6, help="decimals on the mean")
    p.add_argument("--std_digits", type=int, default=4, help="decimals on the std")
    p.add_argument("--caption", default="Relative energy errors of the MS-GFEM solution "
                                        "with predicted local bases, mean $\\pm$ std over "
                                        "the test samples.")
    p.add_argument("--label", default="tab:fno_msgfem_errors")
    args = p.parse_args()

    errors_root = Path(args.errors_root)
    rows = []
    for case in args.cases:
        stats, n = load_case(errors_root, case)
        rows.append((case, stats, n))
        summary = "  ".join(
            f"{title}={stats[key][0]:.{args.digits}f}+-{stats[key][1]:.{args.std_digits}f}"
            for key, title in
            [(k, k.replace("error_", "")) for k, _ in COLUMNS]
        )
        print(f"[{case}] N={n}  {summary}", flush=True)

    table = build_table(rows, args.digits, args.std_digits, args.caption, args.label)
    print("\n" + table)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(table + "\n")
        print(f"\n[out] {out}")


if __name__ == "__main__":
    main()

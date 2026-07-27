"""
make_error_tables.py
====================

Loads ``errors/<case>/errors.npz`` (as written by ``integrate_predictions.py``)
and creates a LaTeX table of the relative energy errors with columns

    err(uG, uh),  err(uG_pred, uh),  err(uG_pred, uG),

one row per sample plus a final row holding the per-column means.

The table is written next to the error file as ``errors/<case>/errors.tex``.
"""

import argparse
import os

import numpy as np


ERRORS = [
    ("error_uG_vs_uh",      r"$\mathrm{err}(u_G, u_h)$"),
    ("error_uG_pred_vs_uh", r"$\mathrm{err}(u_G^{\mathrm{pred}}, u_h)$"),
    ("error_uG_pred_vs_uG", r"$\mathrm{err}(u_G^{\mathrm{pred}}, u_G)$"),
]


def main():
    parser = argparse.ArgumentParser(
        description="Build a LaTeX table of MS-GFEM prediction errors.",
    )
    parser.add_argument("--case", type=str, required=True,
                        help="Case tag; expects errors/<case>/errors.npz.")
    args = parser.parse_args()

    case_dir = os.path.join("errors", args.case)
    in_path = os.path.join(case_dir, "errors.npz")
    out_path = os.path.join(case_dir, "errors.tex")

    with np.load(in_path, allow_pickle=True) as d:
        columns = [np.asarray(d[key]).ravel() for key, _ in ERRORS]

    num_samples = len(columns[0])
    means = [col.mean() for col in columns]

    headers = " & ".join(label for _, label in ERRORS)
    lines = [
        r"\begin{tabular}{r" + "c" * len(ERRORS) + "}",
        r"\toprule",
        r"sample & " + headers + r" \\",
        r"\midrule",
    ]
    for i in range(num_samples):
        row = " & ".join(f"{col[i]:.6e}" for col in columns)
        lines.append(f"{i} & {row} " + r"\\")
    lines.append(r"\midrule")
    mean_row = " & ".join(f"{m:.6e}" for m in means)
    lines.append(r"mean & " + mean_row + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

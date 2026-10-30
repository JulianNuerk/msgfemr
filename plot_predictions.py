"""
Plot the FNO-predicted local basis and the local coefficient of one subdomain.

Reads the prediction bundles written by ``msgfem.inference_all_subdomains``::

    <pred_root>/<case>/samples_<N>/<split_leaf>/
        manifest.json
        predictions/sub_dom_<idx>.npz

Each ``sub_dom_<idx>.npz`` holds ``pred`` of shape ``(num_samples, k, nx, ny)``,
the source sample ``indices``, and -- when inference ran with
``save_coeff``/``save_target`` -- the matching ``coeff`` and ``target``.
The arrays are already cropped to the true subdomain grid, so the halo padding
recorded in ``pads`` must not be sliced off again.

One figure per subdomain: one row per sample, the first column shows the
coefficient, the remaining columns the predicted basis functions. With
``--show_target`` every sample gets a second row holding the ground truth on a
shared colour scale.

Usage
-----
    python plot_predictions.py --case channel_rotation_circle_coeff \
        --sub_dom_idx_list 0 5 15 [--num_samples 100] [--show_target]
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm

PRED_ROOT = Path("/fs/scratch/rb_bd_dlp_rng_dl01_cr_MSO_employees/students/"
                 "nuj7rng/msgfem_data/FNO_predictions")


def subdomain_extent(subdom_idx, Ny, ny, ol, os):
    """Physical extent (x0, x1, y0, y1) of the oversampling domain omega^*."""
    h = 1.0 / ny
    side = 1.0 / Ny
    i, j = subdom_idx % Ny, subdom_idx // Ny
    pad = (ol + os) * h
    return (max(i * side - pad, 0.0), min((i + 1) * side + pad, 1.0),
            max(j * side - pad, 0.0), min((j + 1) * side + pad, 1.0))


def load_subdomain(pred_dir, subdom_idx, num_plot_samples):
    """First ``num_plot_samples`` rows of ``sub_dom_<idx>.npz`` as a dict."""
    path = pred_dir / f"sub_dom_{subdom_idx}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"No prediction file {path}")

    with np.load(path) as z:
        keys = set(z.files)
        n = min(num_plot_samples, z["pred"].shape[0])
        out = {
            "pred": z["pred"][:n],
            "indices": z["indices"][:n] if "indices" in keys else np.arange(n),
            "subdomain_type": str(z["subdomain_type"]) if "subdomain_type" in keys else "?",
            "seen_in_training": bool(z["seen_in_training"]) if "seen_in_training" in keys else None,
        }
        # Optional payloads: inference writes them only when asked to.
        out["coeff"] = z["coeff"][:n] if "coeff" in keys else None
        out["target"] = z["target"][:n] if "target" in keys else None
        # `arccos` is stored but left empty when the bundle has no ground truth.
        arccos = z["arccos"] if "arccos" in keys else np.empty(0)
        out["arccos"] = arccos[:n] if arccos.size else None
    return out


def align_sign(pred, target):
    """Flip each predicted mode onto the target; the basis is sign-ambiguous."""
    s = np.sign(np.sum(pred * target, axis=(-2, -1), keepdims=True))
    s[s == 0] = 1.0
    return pred * s


def draw_coefficient(ax, A, label):
    pos = A[A > 0]
    norm = (LogNorm(vmin=pos.min(), vmax=A.max())
            if pos.size == A.size and A.max() > A.min() else Normalize())
    im = ax.imshow(A.T, origin="lower", extent=ax._msgfem_extent, cmap="viridis", norm=norm)
    contrast = A.max() / pos.min() if pos.size else np.nan
    ax.set_title(fr"$A_{{\xi_{{{label}}}}}$, contrast $={contrast:.2f}$", fontsize=9)
    return im


def draw_mode(ax, z, title, vmax):
    im = ax.imshow(z.T, origin="lower", extent=ax._msgfem_extent, cmap="RdBu_r",
                   norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax))
    ax.set_title(title, fontsize=9)
    return im


def plot_subdomain(case, subdom_idx, pred_dir, args):
    data = load_subdomain(pred_dir, subdom_idx, args.num_plot_samples)
    pred, target, coeff = data["pred"], data["target"], data["coeff"]
    sample_ids = data["indices"]

    show_target = args.show_target and target is not None
    if args.show_target and target is None:
        print(f"  [warn] sub_dom {subdom_idx}: no 'target' in the bundle "
              f"(inference.save_target=false); plotting predictions only")
    if coeff is None:
        print(f"  [warn] sub_dom {subdom_idx}: no 'coeff' in the bundle "
              f"(inference.save_coeff=false); skipping the coefficient column")

    n_samples, n_modes = pred.shape[0], pred.shape[1]
    rows_per_sample = 2 if show_target else 1
    n_rows = n_samples * rows_per_sample
    has_coeff = coeff is not None
    n_cols = n_modes + (1 if has_coeff else 0)

    # Arrays were cropped back to the true subdomain grid, so the physical
    # extent is that of omega^* -- but only if the stored grid agrees with it.
    extent = subdomain_extent(subdom_idx, args.Ny, args.ny, args.ol, args.os)
    h = 1.0 / args.ny
    expected = (round((extent[1] - extent[0]) / h) + 1, round((extent[3] - extent[2]) / h) + 1)
    if tuple(pred.shape[-2:]) != expected:
        print(f"  [warn] sub_dom {subdom_idx}: grid {tuple(pred.shape[-2:])} != expected "
              f"{expected} for Ny={args.Ny}, ny={args.ny}, ol={args.ol}, os={args.os}; "
              f"axes fall back to pixel indices")
        extent = None

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.9 * n_cols, 3.0 * n_rows),
                             squeeze=False)
    for ax in axes.ravel():
        ax._msgfem_extent = extent

    for r in range(n_samples):
        s = int(sample_ids[r])
        p = align_sign(pred[r], target[r]) if show_target else pred[r]
        blocks = [("pred", p)] + ([("truth", target[r])] if show_target else [])

        for b, (kind, phi) in enumerate(blocks):
            row = axes[r * rows_per_sample + b]
            c0 = 0
            if has_coeff:
                c0 = 1
                if b == 0:
                    # Channel 0 is A; channel 1 is the domain mask.
                    im = draw_coefficient(row[0], coeff[r, 0], s)
                    fig.colorbar(im, ax=row[0], fraction=0.046, pad=0.04)
                else:
                    row[0].axis("off")

            for k in range(n_modes):
                # Shared scale so prediction and truth are directly comparable.
                vmax = (max(np.abs(p[k]).max(), np.abs(target[r, k]).max())
                        if show_target else np.abs(pred[r, k]).max())
                vmax = vmax if vmax > 0 else 1.0
                im = draw_mode(row[c0 + k], phi[k],
                               fr"$\Phi_{{\xi_{{{s}}}}}^{{{k}}}$ ({kind})", vmax)
                fig.colorbar(im, ax=row[c0 + k], fraction=0.046, pad=0.04)

            for ax in row:
                ax.set_xlabel(r"$x$", fontsize=8)
                ax.tick_params(labelsize=7)
            ylabel = r"$y$"
            if kind == "pred" and data["arccos"] is not None:
                ylabel = fr"$y$   (arccos $={float(data['arccos'][r]):.3e}$)"
            row[0].set_ylabel(ylabel, fontsize=8)

    seen = data["seen_in_training"]
    seen_txt = "" if seen is None else f", {'seen' if seen else 'unseen'} in training"
    fig.suptitle(f"{case} — subdomain {subdom_idx} "
                 f"({data['subdomain_type']}{seen_txt})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_dir = Path(args.out_dir) / case
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "pred_vs_truth" if show_target else "predictions"
    out_file = out_dir / f"{case}_sub_dom_{subdom_idx}_{suffix}.png"
    fig.savefig(out_file, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_file}  (samples {[int(i) for i in sample_ids]})")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", type=str, required=True,
                        help="case / store tag, i.e. the bundle directory name")
    parser.add_argument("--sub_dom_idx_list", type=int, nargs="+", default=[5],
                        help="subdomains to plot; one figure each")
    parser.add_argument("--bundle", type=str, default=None,
                        help="bundle directory; overrides --pred_root/--num_samples/--split_leaf")
    parser.add_argument("--pred_root", type=str, default=str(PRED_ROOT),
                        help="root holding <case>/samples_<N>/<split_leaf>/")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="selects the directory samples_<num_samples>")
    parser.add_argument("--split_leaf", type=str, default="all_samples",
                        help="split folder inside samples_<N> ('test', 'all_samples', ...)")
    parser.add_argument("--num_plot_samples", type=int, default=5,
                        help="number of samples (rows) to plot")
    parser.add_argument("--show_target", action="store_true",
                        help="add a ground-truth row per sample, sign-aligned to the prediction")
    parser.add_argument("--Ny", type=int, default=4, help="coarse cells per direction")
    parser.add_argument("--ny", type=int, default=2 ** 8, help="fine cells per direction")
    parser.add_argument("--ol", type=int, default=2, help="overlap in fine cells")
    parser.add_argument("--os", type=int, default=2, help="oversampling in fine cells")
    parser.add_argument("--out_dir", type=str, default="plots/predictions",
                        help="directory in which the figures are stored")
    args = parser.parse_args()

    bundle = (Path(args.bundle) if args.bundle else
              Path(args.pred_root) / args.case / f"samples_{args.num_samples}" / args.split_leaf)
    pred_dir = bundle / "predictions"
    if not pred_dir.is_dir():
        raise FileNotFoundError(
            f"No prediction directory {pred_dir} -- run the evaluation first, "
            f"or point --bundle at the bundle written by inference_all_subdomains"
        )
    print(f"bundle = {bundle}")

    for subdom_idx in args.sub_dom_idx_list:
        plot_subdomain(args.case, subdom_idx, pred_dir, args)


if __name__ == "__main__":
    main()

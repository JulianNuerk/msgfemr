"""Load FNO prediction bundles written by ``msgfem.inference_all_subdomains``.

The bundle is the unit of trust here: predictions, the parameter rows that
generated the coefficients they were computed from, and a manifest recording
where both came from all live in one directory, written in a single pass::

    <root>/<case>/samples_<N>/test/
        manifest.json
        input_paras.pt          (num_samples, P) -- row r pairs with row r of every pred
        indices.npy             original sample indices in the source tensors
        predictions/sub_dom_<id>.npz
        metrics/summary.json

That pairing is exactly what ``compute_overall_errors`` assumes and cannot
check for itself: it consumes ``parameters`` and ``predictions_data``
positionally, so a mismatched parameter file does not raise, it just yields
errors computed against the wrong coefficient. Hence every consistency check
below raises instead of warning.

Lives next to ``integrate_predictions.py`` so the verification can reach the
``setup.FNO_coeffs`` that actually generated the datasets.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_fno_predictions(bundle_dir, subdomain_ids=None, verify_source=True):
    """``(store_tag, parameters, predictions_data)`` ready for ``compute_overall_errors``.

    ``subdomain_ids`` restricts the returned mapping (default: every
    ``sub_dom_*.npz`` present). ``verify_source=False`` skips re-hashing the
    original ``input_paras*.pt``, which is only appropriate when the source
    tree is no longer reachable.
    """
    bundle = Path(bundle_dir)
    manifest_path = bundle / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{manifest_path} missing -- not a bundle written by "
            f"msgfem.inference_all_subdomains with inference.output_root set"
        )
    manifest = json.loads(manifest_path.read_text())

    paras_path = bundle / manifest["files"]["input_paras"]
    parameters = torch.load(paras_path.as_posix(), map_location="cpu", weights_only=False)
    parameters = np.asarray(parameters, dtype=np.float64)
    if parameters.ndim == 1:
        parameters = parameters[None, :]

    indices = np.load(bundle / manifest["files"]["indices"])
    n = int(manifest["num_samples"])
    if parameters.shape[0] != n or indices.shape[0] != n:
        raise ValueError(
            f"{bundle}: manifest says {n} samples but input_paras has "
            f"{parameters.shape[0]} and indices has {indices.shape[0]}"
        )

    # The source file is the one that generated the coefficients; if it changed
    # on disk the stored parameters may no longer describe this data.
    if verify_source:
        src = Path(manifest["source_input_paras"]["path"])
        if src.is_file():
            digest = _sha256(src)
            if digest != manifest["source_input_paras"]["sha256"]:
                raise ValueError(
                    f"{src} has changed since the predictions were written "
                    f"(sha256 {digest[:12]}... != {manifest['source_input_paras']['sha256'][:12]}...). "
                    f"Re-run inference, or pass verify_source=False to override."
                )
        else:
            print(f"[warn] source parameter file not reachable, skipping hash check: {src}")

    pred_dir = bundle / "predictions"
    available = sorted(int(p.stem.split("_")[-1]) for p in pred_dir.glob("sub_dom_*.npz"))
    if not available:
        raise FileNotFoundError(f"No 'sub_dom_*.npz' under {pred_dir}")
    wanted = available if subdomain_ids is None else [int(i) for i in subdomain_ids]
    missing = sorted(set(wanted) - set(available))
    if missing:
        raise FileNotFoundError(f"{pred_dir}: no predictions for subdomains {missing}")

    predictions_data = {}
    for sd_id in wanted:
        with np.load(pred_dir / f"sub_dom_{sd_id}.npz") as z:
            pred = z["pred"]
            if pred.shape[0] != n:
                raise ValueError(
                    f"sub_dom_{sd_id}: {pred.shape[0]} samples but parameters has {n}"
                )
            if pred.ndim != 4:
                raise ValueError(
                    f"sub_dom_{sd_id}: expected (num_samples, k, nx+1, ny+1), got {pred.shape}"
                )
            if "indices" in z and not np.array_equal(z["indices"], indices):
                raise ValueError(
                    f"sub_dom_{sd_id}: per-file indices differ from {bundle.name}/indices.npy; "
                    f"the bundle is inconsistent and rows would not align with parameters"
                )
        predictions_data[sd_id] = pred

    return manifest["store_tag"], parameters, predictions_data


def verify_against_coefficients(bundle_dir, n_check=3, rtol=1e-8, atol=1e-10, seed=0,
                                deg=1, Ny=4, ny=2 ** 8, ol=2, os_=2, store_tag=None):
    """Prove the stored parameters generated the stored coefficients.

    Regenerates A on one subdomain from ``parameters[r]`` for a few random rows
    and compares it to the ``coeff`` channel saved alongside the predictions.
    This is the only check that catches a parameter file that is internally
    consistent but simply belongs to a different dataset. Requires the dolfinx
    environment and ``inference.save_coeff=true`` at inference time.

    ``store_tag`` overrides the manifest entry, which is needed whenever the
    dataset directory is named differently from the coefficient family in
    ``setup.FNO_coeffs``.
    """
    from generate_coeff_on_subdomains import coeff_on_subdomain  # noqa: PLC0415 - dolfinx-side

    bundle = Path(bundle_dir)
    manifest = json.loads((bundle / "manifest.json").read_text())
    tag, parameters, _ = load_fno_predictions(bundle, verify_source=False)
    tag = store_tag or tag

    pred_dir = bundle / "predictions"
    sd_id = sorted(int(p.stem.split("_")[-1]) for p in pred_dir.glob("sub_dom_*.npz"))[0]
    with np.load(pred_dir / f"sub_dom_{sd_id}.npz") as z:
        if "coeff" not in z:
            raise KeyError(
                f"sub_dom_{sd_id}.npz has no 'coeff'; re-run inference with "
                f"inference.save_coeff=true to enable this check"
            )
        coeff = z["coeff"]

    rng = np.random.default_rng(seed)
    rows = rng.choice(parameters.shape[0], size=min(n_check, parameters.shape[0]),
                      replace=False)
    regenerated, _ = coeff_on_subdomain(
        sd_id, parameters[rows], tag, deg, Ny, ny, ol, os_
    )
    for k, r in enumerate(rows):
        stored = coeff[r, 0]
        got = regenerated[k, 0]
        if got.shape != stored.shape:
            raise ValueError(
                f"{bundle}: regenerated A for sub_dom_{sd_id} has shape {got.shape} "
                f"but the stored coeff is {stored.shape}; the decomposition "
                f"parameters (deg/Ny/ny/ol/os) do not match the dataset"
            )
        if not np.allclose(got, stored, rtol=rtol, atol=atol):
            raise ValueError(
                f"{bundle}: coefficient regenerated from parameters[{r}] does not match "
                f"the stored coeff of sub_dom_{sd_id}. input_paras.pt does not belong "
                f"to this prediction set."
            )
    print(f"[ok] {len(rows)} coefficients regenerate exactly from "
          f"{manifest['files']['input_paras']} (sub_dom_{sd_id})")
    return True

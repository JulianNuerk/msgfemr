"""
generate_coeff_on_subdomains.py
===============================

Generate the PDE coefficient A on EVERY subdomain of the coarse decomposition,
for the test-split samples of an existing FNO dataset.

Training data is only produced for the 9 representative subdomains, so a model
trained there cannot be evaluated on the remaining positions of Omega: the
loader reads A from disk and there is no file for them. This fills that gap
without touching the eigenproblem -- predicting a basis needs only A, and the
ground-truth basis for the overall MS-GFEM error is recomputed inside
``integrate_predictions.compute_overall_errors`` anyway.

Per requested subdomain it

* rebuilds the global mesh ``ex_FNO.py`` used (``deg``, ``Ny``, ``ny``),
* rebuilds the oversampling region ``omega^*`` with the geometry
  ``msgfem_parallel.computeSubdomainFNO`` derives from ``(ol, os)``, so the
  emitted grid matches the training files cell for cell (73x73 interior,
  69x69 corner, 73x69 / 69x73 edge for the usual 4x4, ol=os=2 setup),
* interpolates A onto that submesh for every selected parameter row.

Parameter rows come from the dataset's own ``input_paras*`` file and are sliced
with the partition the training run computed, so row r of the output is the
same global realization as row ``source_indices[r]`` of the source tensors.

TWO ENVIRONMENTS
----------------
The dolfinx env has no torch and the training env has no dolfinx, so this runs
in stages -- the same split ``ex_FNO.py`` already lives with (it writes
``.npy``; the ``.pt`` tensors are packed afterwards). Run from this directory,
whose ``setup.FNO_coeffs`` is the one that generated the datasets::

    # 0) torch env: export the test parameter rows as .npy
    conda run -n msgfemno python generate_coeff_on_subdomains.py --export_paras \
        --case multiscale_sincos_hc_coeff --total_samples 10000

    # 1) dolfinx env: generate the coefficients as .npy
    conda run -n msgfemr python generate_coeff_on_subdomains.py \
        --case multiscale_sincos_hc_coeff \
        --total_samples 10000 --subdom_idx_list 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15

    # 2) torch env: pack .npy -> .pt in the layout the loader globs for
    conda run -n msgfemno python generate_coeff_on_subdomains.py --pack_only \
        --case multiscale_sincos_hc_coeff --total_samples 10000 --nloc 5

``--store_tag`` defaults to ``--case`` and the dataset directories are named
after the coefficient families here, so it only needs setting when the two
diverge. Verified against the shipped data: regenerating A reproduces the
stored coeff tensors exactly for bubble_coeff and channel_rotation_circle_coeff,
and to 4e-13 for multiscale_sincos_hc_coeff.

Output lands in ``<fno_data_root>/<out_subdir>/<case>/samples_<n_test>/``, which
is what ``inference.data_subdir=<out_subdir>`` resolves to, so the evaluation
reads it with ``data.total_samples=<n_test>`` and
``inference.restrict_to_split=null`` (the directory IS the test set).
"""

import os as ops

# Keep BLAS single-threaded: every Ray task below asks for one CPU, and the
# linear-algebra backends would otherwise oversubscribe the allocated cores.
for _thread_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    ops.environ.setdefault(_thread_var, "1")

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_FNO_ROOT = (
    "/fs/scratch/rb_bd_dlp_rng_dl01_cr_MSO_employees/students/nuj7rng"
    "/msgfem_data/FNO_data"
)


def _torch():
    """torch, or a message naming the env that has it."""
    try:
        import torch  # noqa: PLC0415 - optional, absent in the dolfinx env
        return torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torch is required for this step and is absent from the dolfinx "
            "environment; run it in the training env (msgfemno), or pass "
            "--paras <file>.npy to keep this step torch-free"
        ) from exc


def split_indices(n, split, num_train, num_val, num_test, seed):
    """The partition ``msgfem.datasets.split_index_triple`` would produce.

    Deliberately duplicated rather than imported: this runs in the dolfinx
    environment, which does not have the training package. Both depend only on
    (split, seed, sizes, n). 'random' needs torch's Generator specifically,
    since that is what the training side draws its permutation from.
    """
    if num_train + num_val + num_test > n:
        raise ValueError(
            f"requested {num_train}+{num_val}+{num_test} samples > available {n}"
        )
    if split == "sequential":
        order = list(range(n))
    elif split == "random":
        torch = _torch()
        g = torch.Generator().manual_seed(int(seed))
        order = torch.randperm(n, generator=g).tolist()
    else:
        raise ValueError(f"unknown split {split!r}; use 'sequential' or 'random'")
    lo = num_train + num_val
    return {
        "train": order[:num_train],
        "val": order[num_train:lo],
        "test": order[lo:lo + num_test],
    }


def _find_one(data_dir: Path, pattern: str) -> Path:
    matches = sorted(data_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No {pattern!r} under {data_dir}")
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple {pattern!r} under {data_dir}: {[p.name for p in matches]}; "
            f"exactly one must describe this dataset"
        )
    return matches[0]


def load_parameters(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.float64)
    torch = _torch()
    return np.asarray(
        torch.load(path.as_posix(), map_location="cpu", weights_only=False),
        dtype=np.float64,
    )


def subdomain_omega_os(i_subdom, xL, yL, xR, yR, Nx, Ny, nx, ny, ol, os_):
    """``omega^*`` of one subdomain, matching computeSubdomainFNO's geometry.

    Clamping to the global box is what makes boundary subdomains smaller than
    interior ones: a side lying on dOmega gets no oversampling layer, which is
    exactly the asymmetry the padding in the training loader undoes.
    """
    from msgfem_parallel import Omega  # noqa: PLC0415 - dolfinx-side

    d_x, d_y = xR - xL, yR - yL
    i = np.rint(np.mod(i_subdom, Nx))
    j = np.rint((i_subdom - i) / Nx)
    hx, hy = d_x / nx, d_y / ny
    pad_x, pad_y = (ol + os_) * hx, (ol + os_) * hy
    x0 = [
        np.max([xL + d_x * i / Nx - pad_x, xL]),
        np.max([yL + d_y * j / Ny - pad_y, yL]),
    ]
    x1 = [
        np.min([xL + d_x * (i + 1) / Nx + pad_x, xR]),
        np.min([yL + d_y * (j + 1) / Ny + pad_y, yR]),
    ]
    return Omega(x0, x1, xR, xL, yR, yL, nx, ny)


def coeff_on_subdomain(i_subdom, parameters, store_tag, deg=1, Ny=4, ny=2 ** 8,
                       ol=2, os_=2):
    """``(n_samples, 1, nx+1, ny+1)`` float64 coefficient block for one subdomain.

    Mesh, submesh and the dof->grid index map are built once and reused for
    every sample: they depend on the decomposition, not on the parameters, and
    rebuilding them per sample is what would make this expensive.
    """
    from mpi4py import MPI  # noqa: PLC0415 - dolfinx-side
    from dolfinx.fem import Function, functionspace  # noqa: PLC0415
    from dolfinx.mesh import create_rectangle, CellType  # noqa: PLC0415
    import setup  # noqa: PLC0415

    xL, yL, xR, yR = 0, 0, 1, 1
    xy_scaling = np.rint((xR - xL) / (yR - yL))
    Nx = int(np.rint(xy_scaling * Ny))
    nx = int(ny * xy_scaling)

    if np.mod(nx, Nx) != 0 or np.mod(ny, Ny) != 0:
        raise Exception("Finemesh has to be a submesh of coarse mesh")

    msh = create_rectangle(
        comm=MPI.COMM_SELF,
        points=((xL, yL), (xR, yR)),
        n=(nx, ny),
        cell_type=CellType.quadrilateral,
    )
    V = functionspace(msh, ("Lagrange", deg))

    omega_os = subdomain_omega_os(i_subdom, xL, yL, xR, yR, Nx, Ny, nx, ny, ol, os_)
    submesh = create_rectangle(
        comm=MPI.COMM_SELF,
        points=(omega_os.x0, omega_os.x1),
        n=(omega_os.nx, omega_os.ny),
        cell_type=CellType.quadrilateral,
    )
    V_coeff = functionspace(submesh, ("Lagrange", deg))
    dof_coord = V_coeff.tabulate_dof_coordinates()
    # Same dof -> FNO grid mapping computeSubdomainFNO uses.
    input_indices = np.round(
        (dof_coord[:, :-1] - omega_os.x0) * np.array([nx, ny])
    ).astype(np.int32)
    ix, iy = input_indices[:, 0], input_indices[:, 1]

    parameters = np.atleast_2d(np.asarray(parameters, dtype=np.float64))
    out = np.zeros(
        (parameters.shape[0], 1, omega_os.nx + 1, omega_os.ny + 1), dtype=np.float64
    )
    coeff_A = Function(V_coeff)
    for s, row in enumerate(parameters):
        _, _, _, _, coeff_A_function = setup.FNO_coeffs(
            xL, yL, xR, yR, V, msh, row, store_tag
        )
        coeff_A.interpolate(coeff_A_function)
        out[s, 0, ix, iy] = coeff_A.vector.array
    return out, input_indices


def _src_dir(args) -> Path:
    d = Path(args.fno_data_root) / args.case / f"samples_{args.total_samples}"
    if not d.is_dir():
        raise FileNotFoundError(f"Source dataset not found: {d}")
    return d


def _out_dir(args, n_test) -> Path:
    return Path(args.fno_data_root) / args.out_subdir / args.case / f"samples_{n_test}"


def _effective_n(args) -> int:
    """Rows actually written, i.e. the test size after any ``--max_samples``."""
    if args.max_samples:
        return min(int(args.max_samples), int(args.num_test))
    return int(args.num_test)


def _resolve_test_rows(args):
    """``(parameters_test, indices, source_name)`` from the source dataset."""
    paras_path = Path(args.paras) if args.paras else _find_one(_src_dir(args), "input_paras*.pt")
    paras = load_parameters(paras_path)
    if paras.shape[0] != args.total_samples:
        raise ValueError(
            f"{paras_path.name} holds {paras.shape[0]} rows but --total_samples="
            f"{args.total_samples}; the split would not match the training run"
        )
    idx = split_indices(
        paras.shape[0], args.split, args.num_train, args.num_val, args.num_test, args.seed
    )["test"]
    if args.max_samples:
        m = int(args.max_samples)
        if m <= 0:
            raise ValueError(f"--max_samples must be > 0, got {args.max_samples}")
        idx = idx[:m]
    return paras[idx], idx, paras_path.name


def cmd_export_paras(args):
    """Write the test parameter rows as .npy so the dolfinx env can read them."""
    paras_test, idx, src_name = _resolve_test_rows(args)
    out_dir = _out_dir(args, paras_test.shape[0])
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "input_paras_test.npy", paras_test)
    np.save(out_dir / "source_indices.npy", np.asarray(idx, dtype=np.int64))
    print(f"[export] {src_name} {paras_test.shape} (idx {min(idx)}..{max(idx)}) "
          f"-> {out_dir/'input_paras_test.npy'}")


def cmd_pack(args):
    """Stack the generated .npy blocks into the .pt files the loader globs for."""
    torch = _torch()
    n_test = _effective_n(args)
    out_dir = _out_dir(args, n_test)
    suffix = f"num_samples_{n_test}_nloc_{args.nloc}"

    paras_npy = out_dir / "input_paras_test.npy"
    if paras_npy.is_file():
        torch.save(torch.from_numpy(np.load(paras_npy)),
                   (out_dir / f"input_paras_{suffix}.pt").as_posix())
        print(f"  input_paras -> input_paras_{suffix}.pt")

    packed = 0
    for npy in sorted(out_dir.glob("coeff_A_sub_dom_*.npy")):
        sd = npy.stem.split("_")[-1]
        arr = np.load(npy)
        if arr.shape[0] != n_test:
            raise ValueError(
                f"{npy.name} has {arr.shape[0]} samples but --num_test={n_test}"
            )
        torch.save(torch.from_numpy(arr),
                   (out_dir / f"coeff_A_sub_dom_{sd}_{suffix}.pt").as_posix())
        packed += 1
        print(f"  sub_dom {sd:>2} {tuple(arr.shape)} -> coeff_A_sub_dom_{sd}_{suffix}.pt")
    if not packed:
        raise FileNotFoundError(f"No 'coeff_A_sub_dom_*.npy' to pack under {out_dir}")
    print(f"[pack] {packed} subdomains in {out_dir}")


def cmd_generate(args):
    store_tag = args.store_tag or args.case

    # Prefer the exported .npy so this step never needs torch.
    exported = _out_dir(args, _effective_n(args)) / "input_paras_test.npy"
    if args.paras is None and exported.is_file():
        paras_test = np.load(exported)
        idx = [int(v) for v in np.load(exported.with_name("source_indices.npy"))]
        src_name = exported.name
    else:
        paras_test, idx, src_name = _resolve_test_rows(args)

    n_test = paras_test.shape[0]
    out_dir = _out_dir(args, n_test)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[src ] {_src_dir(args)}", flush=True)
    print(f"[para] {src_name} -> test {paras_test.shape} "
          f"(idx {min(idx)}..{max(idx)})", flush=True)
    print(f"[out ] {out_dir}", flush=True)
    print(f"[gen ] store_tag={store_tag} subdomains={args.subdom_idx_list}", flush=True)

    def _write(i_subdom):
        block, input_indices = coeff_on_subdomain(
            i_subdom, paras_test, store_tag, args.deg, args.Ny, args.ny, args.ol, args.os_
        )
        np.save(out_dir / f"coeff_A_sub_dom_{i_subdom}.npy", block)
        np.save(out_dir / f"indices_for_reshape_2d_sub_dom_{i_subdom}.npy", input_indices)
        return i_subdom, tuple(block.shape)

    results = []
    if args.no_ray:
        for i_subdom in args.subdom_idx_list:
            res = _write(i_subdom)
            results.append(res)
            print(f"  sub_dom {res[0]:>2} shape={res[1]}", flush=True)
    else:
        import ray  # noqa: PLC0415

        num_cpus_env = ops.environ.get("LSB_DJOB_NUMPROC")
        if not ray.is_initialized():
            ray.init(num_cpus=int(num_cpus_env) if num_cpus_env else None,
                     ignore_reinit_error=True)
        print(f"Ray resources: {ray.cluster_resources()}", flush=True)

        @ray.remote(num_cpus=1)
        def _task(i_subdom, paras, tag, deg, Ny, ny, ol, os_, out):
            block, input_indices = coeff_on_subdomain(
                i_subdom, paras, tag, deg, Ny, ny, ol, os_
            )
            np.save(Path(out) / f"coeff_A_sub_dom_{i_subdom}.npy", block)
            np.save(Path(out) / f"indices_for_reshape_2d_sub_dom_{i_subdom}.npy",
                    input_indices)
            return i_subdom, tuple(block.shape)

        futures = [
            _task.remote(i, paras_test, store_tag, args.deg, args.Ny, args.ny,
                         args.ol, args.os_, str(out_dir))
            for i in args.subdom_idx_list
        ]
        remaining = futures
        while remaining:
            done, remaining = ray.wait(remaining, num_returns=1)
            res = ray.get(done[0])
            results.append(res)
            print(f"  sub_dom {res[0]:>2} shape={res[1]}", flush=True)
        ray.shutdown()

    manifest = {
        "case": args.case,
        "store_tag": store_tag,
        "source_dir": str(_src_dir(args)),
        "source_input_paras": src_name,
        "split": {
            "name": "test", "mode": args.split, "seed": args.seed,
            "total_samples": int(args.total_samples),
            "num_train": args.num_train, "num_val": args.num_val,
            "num_test": args.num_test,
            "index_min": int(min(idx)), "index_max": int(max(idx)), "n_selected": n_test,
        },
        "decomposition": {
            "deg": args.deg, "Ny": args.Ny, "ny": args.ny,
            "ol": args.ol, "os": args.os_, "nloc": args.nloc,
        },
        "subdomains": {str(i): list(shape) for i, shape in sorted(results)},
        "contains": "coefficient only (no phi / eig_vals)",
    }
    (out_dir / "generation_manifest.json").write_text(json.dumps(manifest, indent=2))
    np.save(out_dir / "source_indices.npy", np.asarray(idx, dtype=np.int64))
    print(f"[done] {len(results)} subdomains -> {out_dir}", flush=True)
    print("[next] pack to .pt in the torch env:\n"
          f"       python generate_coeff_on_subdomains.py --pack_only --case {args.case} "
          f"--total_samples {args.total_samples} --num_test {n_test} --nloc {args.nloc}",
          flush=True)


def main():
    p = argparse.ArgumentParser(
        description="Per-subdomain coefficients for the test split of an FNO dataset"
    )
    p.add_argument("--case", required=True,
                   help="dataset directory name under the FNO data root")
    p.add_argument("--store_tag", default=None,
                   help="coefficient family for setup.FNO_coeffs (default: --case)")
    p.add_argument("--fno_data_root", default=DEFAULT_FNO_ROOT)
    p.add_argument("--out_subdir", default="test_allsub",
                   help="matches inference.data_subdir on the evaluation side")
    p.add_argument("--total_samples", type=int, required=True,
                   help="sample count of the SOURCE dataset, i.e. its samples_<N> dir")
    p.add_argument("--paras", default=None,
                   help="explicit parameter file (.npy or .pt); default: the source dataset's")
    p.add_argument("--nloc", type=int, default=5,
                   help="only names the output files; no eigenproblem is solved")
    p.add_argument("--split", default="sequential", choices=["sequential", "random"])
    p.add_argument("--num_train", type=int, default=8000)
    p.add_argument("--num_val", type=int, default=1000)
    p.add_argument("--num_test", type=int, default=1000)
    p.add_argument("--max_samples", type=int, default=None,
                   help="keep only the first N test rows; match "
                        "inference.max_samples on the evaluation side")
    p.add_argument("--seed", type=int, default=23,
                   help="must equal reproducibility.seed of the training run")
    p.add_argument("--subdom_idx_list", type=int, nargs="+", default=list(range(16)))
    p.add_argument("--deg", type=int, default=1)
    p.add_argument("--Ny", type=int, default=4)
    p.add_argument("--ny", type=int, default=2 ** 8)
    p.add_argument("--ol", type=int, default=2)
    p.add_argument("--os", type=int, dest="os_", default=2)
    p.add_argument("--no_ray", action="store_true", help="run serially")
    p.add_argument("--export_paras", action="store_true",
                   help="torch env: write the test parameter rows as .npy and exit")
    p.add_argument("--pack_only", action="store_true",
                   help="torch env: stack the generated .npy into .pt and exit")
    args = p.parse_args()

    if args.export_paras:
        cmd_export_paras(args)
    elif args.pack_only:
        cmd_pack(args)
    else:
        cmd_generate(args)


if __name__ == "__main__":
    main()

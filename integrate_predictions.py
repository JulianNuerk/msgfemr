"""
integrate_predictions.py
========================

Utility for evaluating the impact of learned (FNO) local eigenfunctions on the
overall MS-GFEM solution.  For every sample described by a parameter row (as
generated in ex_FNO.py) this module

* solves the fine-scale reference FEM problem to obtain ``uh``,
* computes the classical non-iterative MS-GFEM solution ``uG`` (eigenvalue
  problems solved on every subdomain),
* computes a "hybrid" MS-GFEM solution ``uG_pred`` in which the local
  eigenfunctions of the subdomains for which a prediction tensor is available
  are replaced by the (reshaped) network predictions while the remaining
  subdomains still use the eigsolve,
* returns the three relative energy errors
  ``err(uG, uh)``, ``err(uG_pred, uh)`` and ``err(uG_pred, uG)``,
* stores XDMF visualisations of ``uh``, ``uG`` and ``uG_pred`` under
  ``plots/overall_predictions/<store_tag>/``.

Predictions are passed as a ``dict[int, np.ndarray]`` (``predictions_data``)
mapping a subdomain index to a tensor of shape
``(num_samples, k, nx_sub+1, ny_sub+1)``.  The set of covered subdomains is
thus simply ``predictions_data.keys()``.  On the command line the predictions
are read from a *directory* holding one ``sub_dom_<idx>.npz`` file per
subdomain (key ``pred``); subdomains without such a file fall back to the
eigsolve.

The mesh / MS-GFEM parameters (``deg``, ``Ny``, ``ny``, ``ol``, ``os``,
``nloc``, ``rho``, ``bool_ring``) default to the values used in
``ex_FNO.py`` so that the predictions produced there can be plugged in
without further tuning.
"""

import os

# Every Ray task below requests a single CPU, so the linear-algebra backends
# must not spawn several threads per task (this has to happen before numpy /
# dolfinx are imported).
for _thread_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_thread_var, "1")

from mpi4py import MPI
from petsc4py import PETSc
import re
import time
from pathlib import Path

import numpy as np
import ray
from dolfinx.fem import Function, functionspace, locate_dofs_geometrical, dirichletbc
from dolfinx.io import XDMFFile
from dolfinx.mesh import create_rectangle, CellType

import msgfem_parallel as msgfem
import helper
import preconditioner as pre
import setup


@ray.remote(num_cpus=1)
def _compute_subdomain_remote(params, coord_global):
    """Ray wrapper around :func:`msgfem_parallel.computeSubdomain`.

    ``coord_global`` is passed separately (as a shared object-store reference)
    so that the large coordinate array is serialised only once instead of once
    per task; it is re-inserted at its position in the parameter list.
    """
    params = list(params)
    params[15] = coord_global
    return msgfem.computeSubdomain(params)


_coord_global_ref = None


def _get_coord_global_ref(coord_global):
    global _coord_global_ref
    if _coord_global_ref is None:
        _coord_global_ref = ray.put(coord_global)
    return _coord_global_ref


def _init_ray():
    """Initialise Ray with the cores allocated by LSF (batch_cpu queue)."""
    if ray.is_initialized():
        return
    num_cpus_env = os.environ.get("LSB_DJOB_NUMPROC")
    ray.init(num_cpus=int(num_cpus_env) if num_cpus_env else None,
             ignore_reinit_error=True)
    print(f"Ray initialised with resources: {ray.cluster_resources()}", flush=True)


# ---------------------------------------------------------------------------
# Preconditioner that can substitute predictions for the local eigsolve
# ---------------------------------------------------------------------------
class _GfemPreconditionerWithPredictions(pre.GfemPreconditioner):
    """MS-GFEM preconditioner that optionally uses a per-subdomain prediction
    as the local basis.

    ``predictions_sample`` is a ``dict[int, np.ndarray]`` mapping a subdomain
    index to a tensor of shape ``(k, nx_sub+1, ny_sub+1)`` (the current sample
    of the corresponding entry in ``predictions_data``).  For every subdomain
    whose index is a key of that dict the eigenvalue problem is skipped and
    the tensor is passed unchanged to :func:`msgfem_parallel.computeSubdomain`
    (which handles the reshape from a 2-D grid to the local dof ordering).
    All other subdomains fall back to the standard eigsolve.
    """

    def __init__(self, pc, data, predictions_sample=None, base_local_data=None,
                 perturbation_parameter=0.0):
        super().__init__(pc, data, perturbation_parameter)
        self.predictions_sample = predictions_sample if predictions_sample is not None else {}
        # When given, only the subdomains carrying a prediction are recomputed
        # and all remaining entries are taken from this (eigsolve) result.
        self.base_local_data = base_local_data
        self.local_data = None

    def _subdomain_params(self, i_subdom):
        params = [
            self.xR,
            self.xL,
            self.yR,
            self.yL,
            self.ol,
            self.os,
            self.Nx,
            self.Ny,
            self.nx,
            self.ny,
            self.nDom,
            self.coeff,
            self.deg,
            self.nloc,
            i_subdom,
            None,  # coord_global, filled in inside the Ray task
            self.dirichlet_boundary,
            self.robin_boundary,
            self.perturbation_parameter,
            self.rho,
            self.bool_ring,
        ]
        if i_subdom in self.predictions_sample:
            params.append(self.predictions_sample[i_subdom])
        return params

    def setUp(self, pc):
        start = time.time()

        if self.base_local_data is None:
            local_data = [None] * self.nDom
            todo = list(range(self.nDom))
        else:
            local_data = list(self.base_local_data)
            todo = sorted(self.predictions_sample)

        # One Ray task per subdomain -> all subdomains are computed in parallel
        # on the cores allocated to the batch_cpu job.
        coord_ref = _get_coord_global_ref(self.coord_global)
        futures = [_compute_subdomain_remote.remote(self._subdomain_params(i), coord_ref)
                   for i in todo]
        for i_subdom, result in zip(todo, ray.get(futures)):
            local_data[i_subdom] = result
        self.local_data = local_data

        nloc_cutoff = np.zeros(self.nDom)
        for i in range(self.nDom):
            nloc_cutoff[i] = local_data[i][0].shape[1]
        nloc_cutoff = nloc_cutoff.astype(int)
        end = time.time()
        print("Time of local computations: ", end - start)

        # Sanity-check partition of unity
        R = [local_data[i][4] for i in range(self.nDom)]
        Xi = [local_data[i][3] for i in range(self.nDom)]

        vec_tmp = np.ones(self.coord_global.shape[0])
        vec_tmp2 = np.zeros(self.coord_global.shape[0])
        for i in range(self.nDom):
            vec_tmp2 = vec_tmp2 + R[i].T.dot(Xi[i].dot(R[i].dot(vec_tmp)))
        if np.max(np.abs(vec_tmp2 - vec_tmp)) > 1e-12:
            raise Exception(
                "Partition of unity is not a partition of unity. Error is: "
                + str(np.max(np.abs(vec_tmp2 - vec_tmp)))
            )

        # Assemble the coarse space and local / coarse solvers
        start = time.time()
        self.AH, self.M_basis = msgfem.assembleCoarseSpace(
            self.A_scipy, local_data, nloc_cutoff
        )
        end = time.time()
        print("Assemble Coarse space:", end - start)
        print("Size of coarse space: ", self.AH.size[0])

        start = time.time()
        self.prepare_local_solver(local_data)
        end = time.time()
        print("Prepare local solvers:", end - start)

        start = time.time()
        self.prepare_coarse_solver()
        end = time.time()
        print("Prepare coarse solvers:", end - start)


# ---------------------------------------------------------------------------
# Small utility helpers
# ---------------------------------------------------------------------------
def _plot_to_path(u, msh, full_path_no_ext):
    """XDMF plot to an arbitrary path (creates parent directory)."""
    parent = os.path.dirname(full_path_no_ext)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with XDMFFile(msh.comm, full_path_no_ext + ".xdmf", "w") as f:
        f.write_mesh(msh)
        f.write_function(u)


def _build_coeff_A_function(store_tag, parameter_row, xL, yL, xR, yR, V, msh):
    """Return the ``coeff_A_function`` for a single parameter sample.

    We reuse :func:`setup.FNO_coeffs` to guarantee identical semantics to the
    data generation pipeline in ``ex_FNO.py``.
    """
    dirichlet_boundary, robin_boundary, u_D, f, coeff_A_function = setup.FNO_coeffs(
        xL, yL, xR, yR, V, msh, parameter_row, store_tag
    )
    return dirichlet_boundary, robin_boundary, u_D, f, coeff_A_function


def _compute_uG(V, msh, coeff_A_function, coeff_A, u_D, f,
                dirichlet_boundary, robin_boundary,
                deg, Nx, Ny, nx, ny, nDom, ol, os_, nloc, rho, bool_ring,
                xL, yL, xR, yR, coord_global,
                predictions_sample=None, base_local_data=None):
    """Assemble the MS-GFEM problem and evaluate the non-iterative multiscale
    solution using ``pc.apply``.  ``predictions_sample`` is a
    ``dict[int, np.ndarray]`` mapping subdomain index to a
    ``(k, nx_sub+1, ny_sub+1)`` tensor replacing the local eigsolve; passing
    ``base_local_data`` (the per-subdomain result of a previous eigsolve run)
    restricts the recomputation to exactly those predicted subdomains.

    Returns ``(uG, local_data)``.
    """
    # Boundary conditions and forms
    dirichlet_dofs_global = locate_dofs_geometrical(V, dirichlet_boundary)
    bc = dirichletbc(u_D, dirichlet_dofs_global)
    a, L = helper.getEllipticProblem(coeff_A, f, V)

    # System matrices
    A, b = helper.assemble_matrix_and_vector(a, L, bc)
    A_tmp, b_tmp, A_scipy = helper.assemble_matrix_and_vector_noniterative(a, L)

    # PETSc KSP / PC scaffolding
    ksp = PETSc.KSP().create()
    ksp.setOperators(A)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.PYTHON)

    data = (
        xR, xL, yR, yL, ol, os_, Nx, Ny, nx, ny, nDom,
        coeff_A_function, deg, nloc, coord_global,
        dirichlet_boundary, robin_boundary, A_scipy, rho, bool_ring,
    )

    gfem_pre = _GfemPreconditionerWithPredictions(
        pc, data, predictions_sample=predictions_sample,
        base_local_data=base_local_data,
    )
    pc.setPythonContext(gfem_pre)

    # Non-iterative multiscale solution (single application of the PC to the RHS)
    uG = Function(V)
    uG_vec = PETSc.Vec().createSeq(b.array.size)
    pc.apply(b_tmp, uG_vec)
    uG.vector.array[:] = uG_vec.array

    return uG, gfem_pre.local_data


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compute_overall_errors(store_tag, parameters, predictions_data,
                           deg=1, Ny=4, ny=2 ** 8, ol=2, os_=2, nloc=5,
                           rho=0.0, bool_ring=False,
                           plot_root="plots/overall_predictions"):
    """Evaluate the effect of network-predicted local bases on the overall
    MS-GFEM solution for every sample in ``parameters``.

    Parameters
    ----------
    store_tag : str
        Identifier of the coefficient family (e.g. ``"channel_coeff"``).  It
        selects the branch in :func:`setup.FNO_coeffs`.
    parameters : np.ndarray, shape ``(num_samples, sample_dim)``
        Rows contain the parameter vector consumed by the coefficient function
        associated with ``store_tag``.
    predictions_data : dict[int, np.ndarray]
        Mapping ``{subdomain_index: tensor}`` where each tensor has shape
        ``(num_samples, k, nx_sub+1, ny_sub+1)`` and contains the network
        predictions of the local eigenfunctions on that subdomain's
        oversampling grid.  The set of covered subdomains is inferred from
        the keys.  Every tensor must expose the same ``num_samples`` as
        ``parameters``; per-subdomain ``(k, nx_sub+1, ny_sub+1)`` shapes may
        differ.
    deg, Ny, ny, ol, os_, nloc, rho, bool_ring
        MS-GFEM configuration; the defaults match ``ex_FNO.py``.
    plot_root : str, optional
        Base directory for the XDMF visualisations of ``uh``, ``uG`` and
        ``uG_pred``, which are always written to ``plot_root/<store_tag>/``.

    Returns
    -------
    dict
        ``{
            'store_tag': ...,
            'sub_dom_idx': sorted list of subdomains with predictions,
            'error_uG_vs_uh': np.ndarray of shape (num_samples,),
            'error_uG_pred_vs_uh': np.ndarray of shape (num_samples,),
            'error_uG_pred_vs_uG': np.ndarray of shape (num_samples,),
        }``
    """
    parameters = np.asarray(parameters)
    if parameters.ndim == 1:
        parameters = parameters[None, :]
    num_samples = parameters.shape[0]

    _init_ray()

    if not isinstance(predictions_data, dict):
        raise TypeError(
            "predictions_data must be a dict[int, np.ndarray] mapping "
            "subdomain index to a (num_samples, k, nx+1, ny+1) tensor."
        )
    # Normalise keys to int and validate per-tensor sample count
    predictions_data = {int(k): np.asarray(v) for k, v in predictions_data.items()}
    for i_sub, tensor in predictions_data.items():
        if tensor.shape[0] != num_samples:
            raise ValueError(
                f"predictions_data[{i_sub}] has {tensor.shape[0]} samples but "
                f"parameters has {num_samples}."
            )
        if tensor.ndim != 4:
            raise ValueError(
                f"predictions_data[{i_sub}] must have shape "
                f"(num_samples, k, nx+1, ny+1); got {tensor.shape}."
            )

    # ---- mesh, function space and geometry (identical to run_msgfem) -------
    xL, yL, xR, yR = 0, 0, 1, 1
    xy_scaling = np.rint((xR - xL) / (yR - yL))
    Nx = np.rint(xy_scaling * Ny).astype(int)
    nDom = Nx * Ny
    nx = (ny * xy_scaling).astype(int)

    if np.mod(nx, Nx) != 0 or np.mod(ny, Ny) != 0:
        raise Exception("Finemesh has to be a submesh of coarse mesh")

    msh = create_rectangle(
        comm=MPI.COMM_WORLD,
        points=((xL, yL), (xR, yR)),
        n=(nx, ny),
        cell_type=CellType.quadrilateral,
    )
    V = functionspace(msh, ("Lagrange", deg))
    coord_global = V.tabulate_dof_coordinates()

    # Output buffers
    err_uG_vs_uh = np.zeros(num_samples)
    err_uG_pred_vs_uh = np.zeros(num_samples)
    err_uG_pred_vs_uG = np.zeros(num_samples)
    err_uG_pred_vs_uh_rel_L2 = np.zeros(num_samples)
    err_uG_pred_vs_uh_rel_H1 = np.zeros(num_samples)

    plot_dir = os.path.join(plot_root, store_tag)

    for sample_idx in range(num_samples):
        print(f"\n=== integrate_predictions: sample {sample_idx + 1}/{num_samples} ===",
              flush=True)

        parameter_row = parameters[sample_idx]

        # Extract per-subdomain prediction slice for this sample:
        # {i_sub: tensor[sample_idx]} with each value of shape (k, nx+1, ny+1)
        predictions_sample = {
            i_sub: tensor[sample_idx]
            for i_sub, tensor in predictions_data.items()
        }

        # Coefficient / bc / rhs for this sample
        (dirichlet_boundary, robin_boundary, u_D, f, coeff_A_function
         ) = _build_coeff_A_function(store_tag, parameter_row,
                                     xL, yL, xR, yR, V, msh)

        coeff_A = Function(functionspace(msh, ("Lagrange", deg)))
        coeff_A.interpolate(coeff_A_function)

        # Fine reference solution uh
        dirichlet_dofs_global = locate_dofs_geometrical(V, dirichlet_boundary)
        bc = dirichletbc(u_D, dirichlet_dofs_global)
        a, L = helper.getEllipticProblem(coeff_A, f, V)
        print('Compute reference uh via fine-scale FEM solve...')
        uh = msgfem.fem_solve(V, a, L, bc)
        print('Done computing uh!')
        # Classical MS-GFEM solution (all subdomains via eigsolve)
        print('Compute uG')
        uG, local_data_eig = _compute_uG(
            V, msh, coeff_A_function, coeff_A, u_D, f,
            dirichlet_boundary, robin_boundary,
            deg, Nx, Ny, nx, ny, nDom, ol, os_, nloc, rho, bool_ring,
            xL, yL, xR, yR, coord_global,
            predictions_sample=None,
        )
        print('Done computing uG!')

        # Hybrid MS-GFEM solution: only the predicted subdomains are rebuilt,
        # every other subdomain reuses the eigsolve result of uG.
        print('Compute uG_pred')
        uG_pred, _ = _compute_uG(
            V, msh, coeff_A_function, coeff_A, u_D, f,
            dirichlet_boundary, robin_boundary,
            deg, Nx, Ny, nx, ny, nDom, ol, os_, nloc, rho, bool_ring,
            xL, yL, xR, yR, coord_global,
            predictions_sample=predictions_sample,
            base_local_data=local_data_eig,
        )
        print('Done computing uG_pred!')

        # Three relative energy errors and classic L2, H1 errors 
        err_uG_vs_uh[sample_idx] = helper.compute_errors(uG, uh, msh, coeff_A)
        err_uG_pred_vs_uh[sample_idx] = helper.compute_errors(uG_pred, uh, msh, coeff_A)
        err_uG_pred_vs_uG[sample_idx] = helper.compute_errors(uG_pred, uG, msh, coeff_A)
        err_uG_pred_vs_uh_rel_L2[sample_idx] = helper.compute_rel_L2(uG_pred, uh, msh)
        err_uG_pred_vs_uh_rel_H1[sample_idx] = helper.compute_rel_H1(uG_pred, uh, msh)


        print(
            f"sample {sample_idx}: "
            f"err(uG,uh)={err_uG_vs_uh[sample_idx]:.4e}, "
            f"err(uG_pred,uh)={err_uG_pred_vs_uh[sample_idx]:.4e}, "
            f"err(uG_pred,uG)={err_uG_pred_vs_uG[sample_idx]:.4e}",
            f"err(uG_pred,uh)_rel_L2={err_uG_pred_vs_uh_rel_L2[sample_idx]:.4e}, "
            f"err(uG_pred,uh)_rel_H1={err_uG_pred_vs_uh_rel_H1[sample_idx]:.4e}",
            flush=True,
        )

        _plot_to_path(uh,      msh, os.path.join(plot_dir, f"uh_sample_{sample_idx}"))
        _plot_to_path(uG,      msh, os.path.join(plot_dir, f"uG_sample_{sample_idx}"))
        _plot_to_path(uG_pred, msh, os.path.join(plot_dir, f"uG_pred_sample_{sample_idx}"))

        # Absolute prediction error field |uG - uG_pred|
        u_abs_err = Function(V)
        u_abs_err.vector.array[:] = np.abs(uG.vector.array - uG_pred.vector.array)
        _plot_to_path(u_abs_err, msh, os.path.join(plot_dir, f"abs_err_uG_vs_uG_pred_sample_{sample_idx}"))

    return {
        "store_tag": store_tag,
        "sub_dom_idx": sorted(predictions_data.keys()),
        "error_uG_vs_uh": err_uG_vs_uh,
        "error_uG_pred_vs_uh": err_uG_pred_vs_uh,
        "error_uG_pred_vs_uG": err_uG_pred_vs_uG,
        "error_uG_pred_vs_uh_rel_L2": err_uG_pred_vs_uh_rel_L2,
        "error_uG_pred_vs_uh_rel_H1": err_uG_pred_vs_uh_rel_H1,
    }


def load_predictions_dir(predictions_dir):
    """Load ``sub_dom_<idx>.npz`` files (key ``pred``) from a directory.

    Returns ``{subdomain_index: array of shape (num_samples, k, nx+1, ny+1)}``.
    Subdomains without a file are simply absent and hence fall back to the
    local eigsolve.
    """
    predictions_dir = Path(predictions_dir)
    predictions_data = {}
    for path in sorted(predictions_dir.glob("sub_dom_*.npz")):
        match = re.fullmatch(r"sub_dom_(\d+)", path.stem)
        if match is None:
            continue
        with np.load(path) as npz:
            predictions_data[int(match.group(1))] = npz["pred"]
    if not predictions_data:
        raise FileNotFoundError(
            f"No 'sub_dom_<idx>.npz' prediction files found in {predictions_dir}."
        )
    return predictions_data


if __name__ == "__main__":
    # Load parameters and predictions and run compute_overall_errors.
    # ``--predictions`` is the directory holding one ``sub_dom_<idx>.npz`` file
    # per subdomain (key ``pred``, shape ``(num_samples, k, nx+1, ny+1)``) and
    # ``--parameters`` the directory holding ``input_parameters.npy``.
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate FNO predictions on the overall MS-GFEM solution.",
    )
    parser.add_argument("--store_tag", type=str, required=True,
                        help="Coefficient family (see setup.FNO_coeffs).")
    parser.add_argument("--parameters", type=str, required=True,
                        help="Directory containing 'input_parameters.npy' of "
                             "shape (num_samples, sample_dim).")
    parser.add_argument("--predictions", type=str, required=True,
                        help="Directory containing 'sub_dom_<idx>.npz' files "
                             "with key 'pred' of shape "
                             "(num_samples, k, nx+1, ny+1).")
    parser.add_argument("--nloc", type=int, default=5)
    parser.add_argument("--out", type=str, default=None,
                        help="Where to store the resulting error arrays. "
                             "Defaults to 'errors/<store_tag>/errors.npz'.")
    args = parser.parse_args()

    if args.out is None:
        args.out = os.path.join("errors", args.store_tag, "errors.npz")

    parameters = np.load(Path(args.parameters) / "input_parameters.npy")
    predictions_data = load_predictions_dir(args.predictions)
    print(f"Loaded predictions for subdomains {sorted(predictions_data)} "
          f"({parameters.shape[0]} parameter samples).", flush=True)

    result = compute_overall_errors(
        store_tag=args.store_tag,
        parameters=parameters,
        predictions_data=predictions_data,
        nloc=args.nloc,
    )

    out_parent = os.path.dirname(args.out)
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)
    np.savez(args.out, **result)
    print(f"Wrote errors to {args.out}")
    ray.shutdown()

"""
gmsfem_driver.py
================

Computes GMsFEM solutions and their energy errors for the PDE and coefficients
of msgfemr, using the same mesh, fine reference solve and error functionals as
``experiments_driver.run_msgfem`` and ``integrate_predictions.compute_overall_errors``.
Optionally the MS-GFEM solution is computed for the same sample so that both
methods can be compared directly.

For every sample:

* ``uh``    fine-scale FEM reference solution (``msgfem_parallel.fem_solve``),
* ``uGMs``  GMsFEM solution for every requested number of basis functions N_bf,
* ``uG``    (optional) non-iterative MS-GFEM solution
            (``integrate_predictions._compute_uG``),

and the relative energy errors ``err(uGMs, uh)``, ``err(uG, uh)`` and
``err(uGMs, uG)`` (``helper.compute_errors``) together with relative L2 / H1
errors of ``uGMs``.
"""

import os
import time

import numpy as np
from mpi4py import MPI
from dolfinx.fem import Function, dirichletbc, functionspace, locate_dofs_geometrical
from dolfinx.mesh import CellType, create_rectangle

import helper
import integrate_predictions as ip
import msgfem_parallel as msgfem
import setup
from GMsFEM.gmsfem import DEFAULT_SIGMA, GMsFEM

COEFF_ELEMENTS = {"lagrange": lambda deg: ("Lagrange", deg), "dg0": lambda deg: ("DG", 0)}


def compute_errors_gmsfem(problem, n_bf_list, parameters,
                          deg=1, Ny=4, ny=2 ** 8, coeff_space="lagrange", sigma=DEFAULT_SIGMA,
                          compare_msgfem=False, nloc=5, ol=2, os_=2, rho=0.0, bool_ring=False,
                          plot_root=None, num_plot_samples=3):
    """GMsFEM (and optionally MS-GFEM) errors for every sample.

    Parameters
    ----------
    problem : str
        store_tag of :func:`setup.FNO_coeffs`.
    n_bf_list : list of int
        Numbers N_bf of basis functions per local domain. The local spectral
        problems are solved once for ``max(n_bf_list)``.
    parameters : np.ndarray, shape (num_samples, sample_dim)
        Coefficient parameters, e.g. from :func:`setup.sample_FNO_parameters`.
    deg, Ny, ny
        Polynomial degree, coarse cells (GMsFEM coarse grid and MS-GFEM
        subdomains per direction) and fine cells per direction.
    coeff_space : {"lagrange", "dg0"}
        Space the coefficient is interpolated into: ``"lagrange"`` as in
        ``integrate_predictions.py``, ``"dg0"`` as in ``run_msgfem``.
    sigma : float
        Shift of the local shift-invert eigensolver.
    compare_msgfem : bool
        Also compute the MS-GFEM solution with ``nloc, ol, os_, rho, bool_ring``.
    plot_root : str, optional
        If given, XDMF files of the first ``num_plot_samples`` samples are
        written to ``plot_root/<problem>/``.

    Returns
    -------
    dict
        Errors of shape ``(num_samples, len(n_bf_list))`` for GMsFEM and
        ``(num_samples,)`` for MS-GFEM, coarse space sizes and timings
        (``time_eig_*``: wall-clock time of the parallel local eigenproblem
        stage; ``time_overall_*``: global assembly + local stage + coarse
        space + solve, per N_bf for GMsFEM).
    """
    n_bf_list = sorted(int(n) for n in n_bf_list)
    samples = np.atleast_2d(parameters)
    num_samples = len(samples)

    # Ray parallelises the local problems (GMsFEM offline stage and MS-GFEM).
    ip._init_ray()

    # ---- mesh and function space (identical to run_msgfem) -----------------
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
    coeff_element = COEFF_ELEMENTS[coeff_space](deg)

    n_k = len(n_bf_list)
    err_uGMs_vs_uh = np.zeros((num_samples, n_k))
    err_uGMs_vs_uh_rel_L2 = np.zeros((num_samples, n_k))
    err_uGMs_vs_uh_rel_H1 = np.zeros((num_samples, n_k))
    err_uGMs_vs_uG = np.full((num_samples, n_k), np.nan)
    err_uG_vs_uh = np.full(num_samples, np.nan)
    time_offline = np.zeros(num_samples)
    time_online = np.zeros((num_samples, n_k))
    time_eig_gmsfem = np.zeros(num_samples)
    time_overall_gmsfem = np.zeros((num_samples, n_k))
    time_eig_msgfem = np.full(num_samples, np.nan)
    time_overall_msgfem = np.full(num_samples, np.nan)
    coarse_size_gmsfem = np.zeros(n_k, dtype=int)
    local_eigenvalues = []

    plot_dir = os.path.join(plot_root, problem) if plot_root else None

    for sample_idx, parameter_row in enumerate(samples):
        print(f"\n=== GMsFEM: sample {sample_idx + 1}/{num_samples} ===", flush=True)

        (dirichlet_boundary, robin_boundary, u_D, f, coeff_A_function) = setup.FNO_coeffs(
            xL, yL, xR, yR, V, msh, parameter_row, problem)

        coeff_A = Function(functionspace(msh, coeff_element))
        coeff_A.interpolate(coeff_A_function)

        # Fine reference solution uh
        dirichlet_dofs = locate_dofs_geometrical(V, dirichlet_boundary)
        bc = dirichletbc(u_D, dirichlet_dofs)
        a, L = helper.getEllipticProblem(coeff_A, f, V)
        uh = msgfem.fem_solve(V, a, L, bc)

        # Global system without boundary conditions (as for MS-GFEM)
        start = time.time()
        _, b_tmp, A_scipy = helper.assemble_matrix_and_vector_noniterative(a, L)
        time_assembly = time.time() - start

        # GMsFEM offline stage
        gmsfem = GMsFEM((xR, xL, yR, yL, Nx, Ny, nx, ny, coeff_A_function, coeff_element,
                         deg, n_bf_list[-1], coord_global, dirichlet_dofs, A_scipy, sigma))
        start = time.time()
        gmsfem.offline()
        time_offline[sample_idx] = time.time() - start
        time_eig_gmsfem[sample_idx] = gmsfem.time_eig
        local_eigenvalues.append(gmsfem.eigenvalues)

        # MS-GFEM solution of the same sample
        uG = None
        if compare_msgfem:
            msgfem_timings = {}
            start = time.time()
            uG, _ = ip._compute_uG(
                V, msh, coeff_A_function, coeff_A, u_D, f,
                dirichlet_boundary, robin_boundary,
                deg, Nx, Ny, nx, ny, nDom, ol, os_, nloc, rho, bool_ring,
                xL, yL, xR, yR, coord_global,
                timings=msgfem_timings,
            )
            time_overall_msgfem[sample_idx] = time.time() - start
            time_eig_msgfem[sample_idx] = msgfem_timings["local"]
            err_uG_vs_uh[sample_idx] = helper.compute_errors(uG, uh, msh, coeff_A)

        # GMsFEM online stage for every N_bf
        uGMs = Function(V)
        for k, n_bf in enumerate(n_bf_list):
            start = time.time()
            gmsfem.assemble_coarse_space(n_bf)
            time_coarse = time.time() - start
            start = time.time()
            uGMs.x.array[:], _ = gmsfem.online(b_tmp.array)
            time_online[sample_idx, k] = time.time() - start
            # Same stages as the MS-GFEM overall time: assembly, local eigenproblems, coarse space, solve.
            time_overall_gmsfem[sample_idx, k] = (time_assembly + time_eig_gmsfem[sample_idx]
                                                  + time_coarse + time_online[sample_idx, k])
            coarse_size_gmsfem[k] = gmsfem.A0.shape[0]

            err_uGMs_vs_uh[sample_idx, k] = helper.compute_errors(uGMs, uh, msh, coeff_A)
            err_uGMs_vs_uh_rel_L2[sample_idx, k] = helper.compute_rel_L2(uGMs, uh, msh)
            err_uGMs_vs_uh_rel_H1[sample_idx, k] = helper.compute_rel_H1(uGMs, uh, msh)
            if uG is not None:
                err_uGMs_vs_uG[sample_idx, k] = helper.compute_errors(uGMs, uG, msh, coeff_A)

            print(f"sample {sample_idx}, N_bf={n_bf}: "
                  f"err(uGMs,uh)={err_uGMs_vs_uh[sample_idx, k]:.4e}, "
                  f"err(uGMs,uh)_rel_L2={err_uGMs_vs_uh_rel_L2[sample_idx, k]:.4e}, "
                  f"err(uGMs,uh)_rel_H1={err_uGMs_vs_uh_rel_H1[sample_idx, k]:.4e}, "
                  f"time_overall={time_overall_gmsfem[sample_idx, k]:.4f} s", flush=True)
        print(f"sample {sample_idx}: GMsFEM time_eig={time_eig_gmsfem[sample_idx]:.4f} s", flush=True)
        if uG is not None:
            print(f"sample {sample_idx}: err(uG,uh)={err_uG_vs_uh[sample_idx]:.4e} (MS-GFEM, nloc={nloc}), "
                  f"time_eig={time_eig_msgfem[sample_idx]:.4f} s, "
                  f"time_overall={time_overall_msgfem[sample_idx]:.4f} s", flush=True)

        if plot_dir is not None and sample_idx < num_plot_samples:
            # uGMs holds the solution for the largest N_bf
            ip._plot_to_path(uh, msh, os.path.join(plot_dir, f"uh_sample_{sample_idx}"))
            ip._plot_to_path(uGMs, msh, os.path.join(plot_dir, f"uGMs_nbf_{n_bf_list[-1]}_sample_{sample_idx}"))
            if uG is not None:
                ip._plot_to_path(uG, msh, os.path.join(plot_dir, f"uG_sample_{sample_idx}"))

    return {
        "problem": problem,
        "n_bf": np.array(n_bf_list),
        "Ny": Ny,
        "ny": ny,
        "deg": deg,
        "coeff_space": coeff_space,
        "coarse_space_size_gmsfem": coarse_size_gmsfem,
        "coarse_space_size_msgfem": nDom * nloc if compare_msgfem else -1,
        "nloc_msgfem": nloc,
        "error_uGMs_vs_uh": err_uGMs_vs_uh,
        "error_uGMs_vs_uh_rel_L2": err_uGMs_vs_uh_rel_L2,
        "error_uGMs_vs_uh_rel_H1": err_uGMs_vs_uh_rel_H1,
        "error_uG_vs_uh": err_uG_vs_uh,
        "error_uGMs_vs_uG": err_uGMs_vs_uG,
        "local_eigenvalues": np.array(local_eigenvalues),
        "time_offline": time_offline,
        "time_online": time_online,
        "time_eig_gmsfem": time_eig_gmsfem,
        "time_overall_gmsfem": time_overall_gmsfem,
        "time_eig_msgfem": time_eig_msgfem,
        "time_overall_msgfem": time_overall_msgfem,
    }

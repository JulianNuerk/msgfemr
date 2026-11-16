"""
gmsfem.py
=========

Generalized Multiscale Finite Element Method (GMsFEM) for

    -div(kappa grad u) = f  in  Omega = (xL, xR) x (yL, yR),   u = 0 on dOmega,

discretised with the same fine Q_deg finite element space V^h as the MS-GFEM
code in msgfemr. The method follows GMsFEM_NO/main_parts/{GMsFEM, spectral_problem,
online_GMsFEM}.tex.

Offline stage
-------------
1. Coarse grid T_H: ``Nx x Ny`` rectangular cells K whose edges are fine-grid
   lines. The coarse nodes x_i are numbered ``i = ix + iy * (Nx + 1)``,
   ``N_v = (Nx + 1)(Ny + 1)``. The local domain omega_i is the union of the
   coarse cells containing x_i (4 cells: full, 2 cells: half, 1 cell: corner).
2. Spectral problem on omega_i, posed on V^h(omega_i) with homogeneous
   Dirichlet conditions on dOmega \cap d omega_i only (natural conditions on
   the interior boundary of omega_i):

       A_h^{omega_i} phi = lambda S_h^{omega_i} phi,
       v^T A_h^{omega_i} w = int_{omega_i} kappa grad v . grad w dx   (Neumann matrix),
       v^T S_h^{omega_i} w = int_{omega_i} kappa v w dx               (mass matrix).

   The eigenpairs belonging to the N_bf smallest eigenvalues are kept and
   normalised to be S_h^{omega_i}-orthonormal (for interior omega_i:
   lambda_1 = 0, phi_1 = const).
3. Multiscale basis psi_j^{omega_i} = chi_i phi_j^{omega_i}, where chi_i is the
   bilinear coarse-grid hat function of x_i, i.e. sum_i chi_i = 1 on Omega.
4. R^T = [psi_1^{omega_1}, ..., psi_{N_bf}^{omega_1}, ..., psi_{N_bf}^{omega_{N_v}}]
   (columns are fine-grid coefficient vectors).

Online stage
------------
    A_0 = R A R^T,   f_0 = R b,   A_0 u_0 = f_0,   u = R^T u_0.

Dirichlet boundary
------------------
The fine solution lives in V_0^h (zero on dOmega). Since the local
eigenfunctions vanish on dOmega \cap d omega_i, every chi_i phi_j lies in
V_0^h, as in MS-GFEM where the local spectral problems also carry the global
Dirichlet condition. u is the Galerkin (energy-orthogonal) projection of the
fine FEM solution onto the multiscale space, so energy errors are directly
comparable to the MS-GFEM ones.
"""

import time

import numpy as np
import ray
from mpi4py import MPI
from dolfinx.fem import Function, assemble_matrix, form, functionspace
from dolfinx.mesh import CellType, create_rectangle
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from ufl import TestFunction, TrialFunction, dx, grad, inner

import msgfem_parallel as msgfem

# Shift for the shift-invert eigensolver: A_h^{omega_i} is singular (Neumann,
# constants in the kernel), A - sigma S with sigma < 0 is SPD. The returned
# eigenpairs are those closest to sigma, i.e. the smallest lambda >= 0.
DEFAULT_SIGMA = -1e-3


def coarse_node(i_node, Nx, Ny, xL, xR, yL, yR):
    """Coordinates of the coarse node x_i and its local domain omega_i.

    Returns
    -------
    node : np.ndarray, shape (2,)
        Coordinates of x_i.
    x0, x1 : np.ndarray, shape (2,)
        Lower-left and upper-right corner of omega_i (cut at dOmega).
    H : np.ndarray, shape (2,)
        Coarse mesh sizes (Hx, Hy).
    """
    H = np.array([(xR - xL) / Nx, (yR - yL) / Ny])
    ix, iy = i_node % (Nx + 1), i_node // (Nx + 1)
    node = np.array([xL + ix * H[0], yL + iy * H[1]])
    x0 = np.maximum(node - H, [xL, yL])
    x1 = np.minimum(node + H, [xR, yR])
    return node, x0, x1, H


def compute_local_domain(parameters):
    """Offline computations on one local domain omega_i.

    Parameters
    ----------
    parameters : tuple
        ``(xR, xL, yR, yL, Nx, Ny, nx, ny, coeff_A_function, coeff_element,
        deg, n_bf, i_node, coord_global, sigma, dirichlet_dofs)`` with
        - Nx, Ny : number of coarse cells per direction,
        - nx, ny : number of fine cells per direction,
        - coeff_A_function : kappa(x), callable on arrays of shape (3, n),
        - coeff_element : element of the coefficient, e.g. ("Lagrange", 1) or
          ("DG", 0); must match the one used for the global matrix so that
          A_h^{omega_i} is exactly the restriction of the global stiffness,
        - deg : polynomial degree of V^h,
        - n_bf : number of eigenpairs computed,
        - i_node : index of the coarse node x_i,
        - coord_global : dof coordinates of the global space V^h,
        - sigma : shift of the shift-invert eigensolver.
        - dirichlet_dofs : global Dirichlet dofs; the local eigenfunctions
          vanish at those inside omega_i.

    Returns
    -------
    psi : np.ndarray, shape (n_loc, n_bf)
        Multiscale basis functions chi_i * phi_j on omega_i (local dofs).
    eigvals : np.ndarray, shape (n_bf,)
        Ascending eigenvalues lambda_j^{omega_i}.
    loc2glob : np.ndarray, shape (n_loc,)
        Local to global dof map of omega_i.
    chi : np.ndarray, shape (n_loc,)
        Partition of unity function chi_i at the local dofs.
    """
    (xR, xL, yR, yL, Nx, Ny, nx, ny, coeff_A_function, coeff_element,
     deg, n_bf, i_node, coord_global, sigma, dirichlet_dofs) = parameters

    node, x0, x1, H = coarse_node(i_node, Nx, Ny, xL, xR, yL, yR)
    h = np.array([(xR - xL) / nx, (yR - yL) / ny])
    n_cells = np.rint((x1 - x0) / h).astype(int)

    # Fine grid of omega_i; it coincides with T_h restricted to omega_i.
    submesh = create_rectangle(
        comm=MPI.COMM_SELF,
        points=(x0, x1),
        n=n_cells,
        cell_type=CellType.quadrilateral,
    )
    Vs = functionspace(submesh, ("Lagrange", deg))
    loc2glob = msgfem.make_submesh_local_to_global_map(
        coord_global, Vs.tabulate_dof_coordinates(), nx, ny, deg)

    kappa = Function(functionspace(submesh, coeff_element))
    kappa.interpolate(coeff_A_function)

    # Neumann matrix and kappa-weighted mass matrix of the spectral problem.
    u, v = TrialFunction(Vs), TestFunction(Vs)
    A_loc = assemble_matrix(form(inner(kappa * grad(u), grad(v)) * dx)).to_scipy().tocsc()
    S_loc = assemble_matrix(form(inner(kappa * u, v) * dx)).to_scipy().tocsc()

    free = np.flatnonzero(~np.isin(loc2glob, dirichlet_dofs))
    eigvals, phi_free = eigsh(A_loc[free][:, free], k=n_bf, M=S_loc[free][:, free],
                              sigma=sigma, which="LM")
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    phi = np.zeros((loc2glob.shape[0], n_bf))
    phi[free] = phi_free[:, order]
    phi = phi / np.sqrt(np.einsum("ij,ij->j", phi, S_loc @ phi))

    # Bilinear coarse hat function of x_i (exactly representable in V^h).
    chi_fn = Function(Vs)
    chi_fn.interpolate(
        lambda x: np.maximum(0.0, 1.0 - np.abs(x[0] - node[0]) / H[0])
        * np.maximum(0.0, 1.0 - np.abs(x[1] - node[1]) / H[1])
    )
    chi = chi_fn.x.array.copy()

    psi = chi[:, None] * phi
    return psi, eigvals, loc2glob, chi


@ray.remote(num_cpus=1)
def _compute_local_domain_remote(params, coord_global):
    """Ray wrapper; ``coord_global`` is shared through the object store."""
    params = list(params)
    params[13] = coord_global
    return compute_local_domain(params)


class GMsFEM:
    """GMsFEM solver on a uniform coarse grid of ``Nx x Ny`` cells.

    Parameters
    ----------
    data : tuple
        ``(xR, xL, yR, yL, Nx, Ny, nx, ny, coeff_A_function, coeff_element,
        deg, n_bf, coord_global, dirichlet_dofs, A_scipy, sigma)``, where
        ``A_scipy`` is the global stiffness matrix *without* boundary
        conditions (as for MS-GFEM, see
        ``helper.assemble_matrix_and_vector_noniterative``), ``dirichlet_dofs``
        are the global Dirichlet dofs and ``n_bf`` the number of eigenpairs per
        local domain computed in the offline stage.
    """

    def __init__(self, data):
        (
            self.xR,
            self.xL,
            self.yR,
            self.yL,
            self.Nx,
            self.Ny,
            self.nx,
            self.ny,
            self.coeff,
            self.coeff_element,
            self.deg,
            self.n_bf,
            self.coord_global,
            self.dirichlet_dofs,
            self.A_scipy,
            self.sigma,
        ) = data
        self.n_nodes = (self.Nx + 1) * (self.Ny + 1)
        self.n_fine = self.coord_global.shape[0]
        self.local_data = None
        self.RT = None
        self.A0 = None
        self._A0_lu = None
        self.time_eig = None

    def _local_params(self, i_node, coord_global):
        return (self.xR, self.xL, self.yR, self.yL, self.Nx, self.Ny, self.nx, self.ny,
                self.coeff, self.coeff_element, self.deg, self.n_bf, i_node,
                coord_global, self.sigma, self.dirichlet_dofs)

    def offline(self):
        """Local spectral problems on all omega_i (in parallel if Ray is
        initialised) and assembly of the coarse space with ``n_bf`` functions
        per local domain."""
        start = time.time()
        if ray.is_initialized():
            coord_ref = ray.put(self.coord_global)
            futures = [_compute_local_domain_remote.remote(self._local_params(i, None), coord_ref)
                       for i in range(self.n_nodes)]
            self.local_data = ray.get(futures)
        else:
            self.local_data = [compute_local_domain(self._local_params(i, self.coord_global))
                               for i in range(self.n_nodes)]
        self.time_eig = time.time() - start
        print("GMsFEM time of local spectral problems: ", self.time_eig)

        self._check_partition_of_unity()
        self.assemble_coarse_space(self.n_bf)

    def _check_partition_of_unity(self):
        pou = np.zeros(self.n_fine)
        for _, _, loc2glob, chi in self.local_data:
            pou[loc2glob] += chi
        deviation = np.max(np.abs(pou - 1.0))
        if deviation > 1e-12:
            raise Exception("Partition of unity is not a partition of unity. Error is: " + str(deviation))

    def assemble_coarse_space(self, n_bf):
        """Build R^T from the first ``n_bf`` basis functions of every omega_i,
        the coarse matrix A_0 = R A R^T and its factorisation."""
        if self.local_data is None:
            raise RuntimeError("Run offline() before assembling the coarse space.")
        if n_bf > self.n_bf:
            raise ValueError(f"n_bf={n_bf} exceeds the {self.n_bf} eigenpairs computed offline.")

        start = time.time()
        rows, cols, vals = [], [], []
        for i, (psi, _, loc2glob, _) in enumerate(self.local_data):
            rows.append(np.repeat(loc2glob, n_bf))
            cols.append(np.tile(i * n_bf + np.arange(n_bf), loc2glob.shape[0]))
            vals.append(psi[:, :n_bf].ravel())
        RT = csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.n_fine, self.n_nodes * n_bf),
        )
        RT.eliminate_zeros()

        self.RT = RT
        self.A0 = (RT.T @ (self.A_scipy @ RT)).tocsc()
        self._A0_lu = msgfem.factorizeCoarseMatrix(self.A0)
        print("GMsFEM assemble coarse space:", time.time() - start)
        print("GMsFEM size of coarse space: ", self.A0.shape[0])

    def online(self, b):
        """Coarse solve A_0 u_0 = R b and reconstruction u = R^T u_0.

        Parameters
        ----------
        b : np.ndarray
            Global load vector (no boundary conditions applied).

        Returns
        -------
        u : np.ndarray
            Fine-scale GMsFEM solution (coefficients in V^h).
        u0 : np.ndarray
            Coarse-scale solution.
        """
        if self._A0_lu is None:
            raise RuntimeError("Run offline() before online().")
        u0 = self._A0_lu.solve(self.RT.T @ b)
        return self.RT @ u0, u0

    @property
    def eigenvalues(self):
        """Local eigenvalues, shape (N_v, n_bf)."""
        return np.array([vals for _, vals, _, _ in self.local_data])

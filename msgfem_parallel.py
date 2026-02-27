from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.mesh import create_rectangle, CellType
from dolfinx.fem import functionspace, Function, locate_dofs_geometrical, assemble_matrix, form
from dolfinx.fem.petsc import LinearProblem
from ufl import dx, grad, inner, TrialFunction, TestFunction
from ray.util.multiprocessing import Pool

import numpy as np
from scipy.sparse import csr_array, diags, csr_matrix, bmat, hstack
from scipy.sparse.linalg import eigsh, splu, spsolve
from scipy.linalg import orth
import helper

import time

class Ring():
    """
    Represents a ring-shaped domain defined to be the outer regian minus the inner region
    Parameters
    ----------
    omega_inner : object
        The inner region, expected to provide geometric methods such as inside, boundary, interior, on_corner, on_global_boundary, and on_global_corner.
    omega_outer : object
        The outer region, expected to provide geometric methods such as inside, boundary, and on_global_boundary.
    Methods
    -------
    inside(x)
        Determines if point(s) x are inside the ring domain, excluding the inner region except for specific corner and boundary cases.
    boundary(x)
        Determines if point(s) x are on the boundary of the ring domain, including boundaries of both inner and outer regions.
    interior(x)
        Determines if point(s) x are strictly inside the ring domain, excluding the boundary.
    on_global_boundary(x)
        Checks if point(s) x are on the global boundary of the global domain.
    on_interior_boundary(x)
        Checks if point(s) x are on the interior boundary of the ring, i.e., on the boundary but not on the global boundary.
    """

    def __init__(self, omega_inner, omega_outer):
        self.omega_inner = omega_inner
        self.omega_outer = omega_outer

    def inside(self, x):
        return np.logical_or(np.logical_and(np.logical_and(self.omega_outer.inside(x), np.logical_not(self.omega_inner.interior(x))), np.logical_not(np.logical_and(self.omega_inner.boundary(x), self.omega_inner.on_global_boundary(x)))),
                              np.logical_and(np.logical_and(self.omega_inner.on_corner(x), np.logical_not(self.omega_inner.on_global_corner(x))), self.omega_inner.on_global_boundary(x)))

    def boundary(self, x):
        return np.logical_and(self.inside(x), np.logical_or(self.omega_outer.boundary(x), self.omega_inner.boundary(x)))
    
    def interior(self, x):
        return np.logical_and(self.inside(x), np.logical_not(self.boundary(x)))
    
    def on_global_boundary(self, x):
        return self.omega_outer.on_global_boundary(x)

    def on_interior_boundary(self, x):
        return np.logical_and(self.boundary(x), np.logical_not(self.on_global_boundary(x)))

class Omega():
    """
    Represents a rectangular domain (possibly a subdomain) in 2D space, with methods to check point locations relative to the domain and its boundaries.
    Parameters
    ----------
    x0 : array-like
        Lower-left corner coordinates of the domain.
    x1 : array-like
        Upper-right corner coordinates of the domain.
    xR : float
        Global right boundary coordinate.
    xL : float
        Global left boundary coordinate.
    yR : float
        Global top boundary coordinate.
    yL : float
        Global bottom boundary coordinate.
    nx : int
        Number of grid points in the x-direction.
    ny : int
        Number of grid points in the y-direction.
    Methods
    -------
    inside(x)
        Returns True if point x is inside the domain (including boundaries).
    interior(x)
        Returns True if point x is strictly inside the domain (excluding boundaries).
    on_global_boundary(x)
        Returns True if point x is on the global boundary.
    not_on_global_boundary(x)
        Returns True if point x is not on the global boundary.
    boundary(x)
        Returns True if point x is on the boundary of the domain (but inside the domain).
    on_interior_boundary(x)
        Returns True if point x is on the interior boundary (domain boundary but not global boundary).
    on_band(x)
        Returns True if point x is within the x or y bounds of the domain.
    on_global_corner(x)
        Returns True if point x is at a global corner (intersection of global boundaries).
    on_corner(x)
        Returns True if point x is at a corner of the domain (intersection of domain boundaries).
    """

    def __init__(self, x0, x1, xR, xL, yR, yL, nx, ny):
        self.x0 = x0
        self.x1 = x1
        self.xR = xR
        self.xL = xL
        self.yR = yR
        self.yL = yL
        hx = (xR - xL) / nx 
        hy = (yR - yL) / ny
        self.nx = np.rint((x1[0] - x0[0]) / hx).astype(int)                     
        self.ny = np.rint((x1[1] - x0[1]) / hy).astype(int) 


    def inside(self, x):
        return np.logical_and(
                              np.logical_and(x[0] >= self.x0[0] - 1e-10, x[1] >= self.x0[1] - 1e-10), 
                              np.logical_and(x[0] <= self.x1[0] + 1e-10, x[1] <= self.x1[1] + 1e-10)
        )
    def interior(self, x):
        return np.logical_and(
                              np.logical_and(x[0] > self.x0[0] + 1e-10, x[1] > self.x0[1] + 1e-10), 
                              np.logical_and(x[0] < self.x1[0] - 1e-10, x[1] < self.x1[1] - 1e-10)
        )

    def on_global_boundary(self, x):
        return np.logical_or(
            np.logical_or(x[0] < self.xL + 1e-10, x[0] > self.xR - 1e-10),
            np.logical_or(x[1] < self.yL + 1e-10, x[1] > self.yR - 1e-10)
        )
    def not_on_global_boundary(self, x):
        return np.logical_not(self.on_global_boundary(x))


    def boundary(self, x):
        return np.logical_and(self.inside(x), np.logical_not(self.interior(x)))


    def on_interior_boundary(self, x):
        return np.logical_and(self.boundary(x), np.logical_not(self.on_global_boundary(x)))
    
    def on_band(self,x):
        # return True if x coordinate is between x boundaries or y coordinate is beween y boundaries
        return np.logical_or(np.logical_and(x[0]>self.x0[0]-1e-10, x[0]<self.x1[0]+1e-10),
                             np.logical_and(x[1]>self.x0[1]-1e-10, x[1]<self.x1[1]+1e-10))
    
    def on_global_corner(self,x): 
        return np.logical_or(
            np.logical_or(np.logical_and(x[0] < self.xL + 1e-10, x[1] < self.yL + 1e-10), np.logical_and(x[0] < self.xL + 1e-10, x[1] > self.yR - 1e-10)),
            np.logical_or(np.logical_and(x[0] > self.xR - 1e-10, x[1] > self.yR - 1e-10), np.logical_and(x[0] > self.xR - 1e-10, x[1] < self.yL + 1e-10))
        )
    
    def on_corner(self,x): 
        return np.logical_and(
                np.logical_or(
                np.logical_or(np.logical_and(x[0] < self.x0[0] + 1e-10, x[1] < self.x0[1] + 1e-10), np.logical_and(x[0] < self.x0[0] + 1e-10, x[1] > self.x1[1] - 1e-10)),
                np.logical_or(np.logical_and(x[0] > self.x1[0] - 1e-10, x[1] > self.x1[1] - 1e-10), np.logical_and(x[0] > self.x1[0] - 1e-10, x[1] < self.x0[1] + 1e-10))
            ),
            self.inside(x)
        )
        
    
        

def make_submesh_local_to_global_map(coord, coord_sub, nx, ny, deg): 
    """
    Constructs a mapping from local degrees of freedom (DOFs) in a submesh to their corresponding global DOF indices.
    This function identifies the correspondence between the coordinates of DOFs in a submesh (`coord_sub`) and the global mesh (`coord`),
    and returns an array that maps each local DOF index in the submesh to its global DOF index.
    Parameters
    ----------
    coord : np.ndarray
        Array of shape (N, 2) containing the coordinates of the global mesh DOFs.
    coord_sub : np.ndarray
        Array of shape (M, 2) containing the coordinates of the submesh DOFs.
    nx : int
        Number of elements in the x-direction of the mesh.
    ny : int
        Number of elements in the y-direction of the mesh.
    deg : int
        Polynomial degree of the finite element basis.
    Returns
    -------
    loc2glob : np.ndarray
        Array of shape (M,) containing the global DOF indices corresponding to each local DOF in the submesh.
    Raises
    ------
    Exception
        If the mapping between local and global DOFs cannot be established due to coordinate mismatch.
    """

    dx_inv = nx * deg      # inverse of x distance between two dofs
    dy_inv = ny * deg      # inverse of y distance between two dofs

    id_sub = np.rint(coord_sub[:,0] * dy_inv) * (dy_inv+1)  + np.rint(coord_sub[:,1]* dy_inv)
    id = np.rint(coord[:,0] * dy_inv) * (dy_inv+1)  + np.rint(coord[:,1]* dy_inv)


    x, index, index_sub = np.intersect1d(id, id_sub, assume_unique=True, return_indices=True)

    if np.linalg.norm(coord_sub[index_sub] - coord[index]) > 1e-10:
        raise Exception("Something went wront for the local to global dofmap construction")

    loc2glob = np.zeros(index.shape[0])
    loc2glob[index_sub] = index
    return loc2glob.astype(int)

def assembleCoarseSpace(A_gfem, local_data, nloc):
    """
    Assembles the global coarse space matrix and basis for the GFEM method.

    Parameters
    ----------
    A_gfem : scipy.sparse.csr_matrix
        Global stiffness matrix.
    local_data : list
        List of tuples containing local coarse space data for each subdomain.
    nloc : list
        Number of local basis functions per subdomain.

    Returns
    -------
    AH : petsc4py.PETSc.Mat
        Coarse space matrix in PETSc format.
    M_basis : scipy.sparse.csr_matrix
        Matrix whose columns are the global coarse basis functions.
    """

    start = time.time()
    basis = []
    for i in range(len(local_data)):
        basis.append(assembleLocalCoarseSpace([local_data[i], nloc[i]]))
    end = time.time()
    print("time local assembly: ", end-start)

    start = time.time()
    M_basis = hstack(basis)
    # AH = M_basis.T.dot(A.dot(M_basis))
    AH = M_basis.T.dot(A_gfem.dot(M_basis))
    end = time.time()
    print("time global assembly: ", end-start)
    return scipy_to_petsc_matrix(AH), M_basis
    

def assembleLocalCoarseSpace(data):
    """
    Assembles the local coarse space basis for a subdomain.

    Parameters
    ----------
    data : tuple
        A tuple containing:
        - local_data: tuple
            Contains local eigenvectors, eigenvalues, local matrices, partition of unity, restriction matrix, non-Dirichlet dofs, and interior dofs.
        - nloc: int
            Number of local basis functions to use.

    Returns
    -------
    csr_matrix
        The local coarse basis functions mapped to the global space.
    """
    (local_data, nloc) = data
    (vecs, vals, As, Xi, R, non_dirichlet_dofs, _) = local_data

    vecs_tmp = np.zeros((Xi.shape[0], nloc))
    vecs_tmp = vecs[:, :nloc] 

    loc_vecs = Xi.dot(vecs_tmp)
    
    loc_basis = loc_vecs[:, :nloc]
    loc_basis = orth(loc_basis)
    loc_basis = csr_matrix(loc_basis)
    
    return R.T.dot(loc_basis)



# def assembleCoarseSpace_old(nDom, Vs, omega_os, eigenpairs, Xi, nloc, A_gfem, R):
#     vecs = []

#     for i in range(nDom):
#         # Locate dofs that are not on global boundary
#         # non_global_bdry_dofs = locate_dofs_geometrical(Vs[i], omega_os[i].not_on_global_boundary)

#         (vs, vals) = eigenpairs[i]

#         # vs = vs[: len(non_global_bdry_dofs), :]

#         non_neg = vals >= 0
#         vals = vals[non_neg]
#         vs = vs[:, non_neg]
#         ordered = np.argsort(vals)
#         vals = vals[ordered]
#         vs = vs[:, ordered]

#         loc_vecs = np.zeros((Vs[i].dofmap.index_map.size_global, vs.shape[1]))
#         # loc_vecs[non_global_bdry_dofs, :] = vs

#         loc_vecs = Xi[i].dot(vs[:Vs[i].dofmap.index_map.size_global,:])

#         vecs.append(loc_vecs)

#     basis = []
#     ndof = 0
#     for i in range(nDom):
#         loc_basis = vecs[i][:, :nloc]
#         loc_basis = orth(loc_basis)
#         loc_basis = csr_matrix(loc_basis)
#         ndof = ndof + loc_basis.shape[1]
#         basis.append(R[i].T.dot(loc_basis))

#     M_basis = hstack(basis)
#     # AH = M_basis.T.dot(A.dot(M_basis))
#     AH = M_basis.T.dot(A_gfem.dot(M_basis))
#     return AH, M_basis

def factorizeCoarseMatrix(AH):
    """
    Factorizes the coarse matrix AH using the SuperLU solver.
    Parameters
    ----------
    AH : scipy.sparse.csr_matrix
        The coarse matrix to be factorized.
    Returns
    -------
    splu(AH)
        The factorized matrix in SuperLU format.
    """
    return splu(AH)

# def factorizeMatrices(As, nDom):
#     """
#     Factorizes a list of sparse matrices using LU decomposition.

#     Parameters
#     ----------
#     As : list of scipy.sparse.csc_matrix
#         List containing the sparse matrices to be factorized.
#     nDom : int
#         Number of domains (i.e., the number of matrices in As).

#     Returns
#     -------
#     As_inv : list of scipy.sparse.linalg.SuperLU
#         List containing the LU factorization objects for each matrix in As.
#     """

#     As_inv = []

#     for i in range(nDom):
#         As_inv.append(splu(As[i]))
#     return As_inv

# def factorizeMatricesParallel(M):
#     (A, interior_dofs) = M 
#     return splu(A)

def computeSubdomain(parameters):
    """
    Constructs and solves a local eigenproblem for a subdomain.
    This function partitions the global domain into overlapping subdomains, builds local meshes and function spaces, 
    computes partition of unity and cut-off functions, assembles local stiffness matrices, and solves a generalized 
    eigenvalue problem to obtain local basis functions. It supports oversampling and ring-based localization strategies, 
    and handles Dirichlet boundary conditions.
    Parameters
    ----------
    parameters : tuple
        A tuple containing all necessary parameters for subdomain construction and computation, including:
        - Domain bounds (xR, xL, yR, yL)
        - Overlap and oversampling sizes (ol, os)
        - Number of subdomains and mesh resolution (Nx, Ny, nx, ny, nDom)
        - Coefficient function and polynomial degree (coeff_A_function, deg)
        - Number of local basis functions (nloc)
        - Subdomain index (k, i_subdom)
        - Global coordinate array (coord_global)
        - Boundary locator functions (dirichlet_boundary, robin_boundary)
        - Perturbation parameter (perturbation_parameter)
        - Eigenvalue cutoff (rho)
        - Ring localization flag (bool_ring)
    Returns
    -------
    tuple
        A tuple containing:
        - vecs_tmp : ndarray
            Local basis functions (extended eigenvectors) for the subdomain.
        - vals : ndarray
            Corresponding eigenvalues.
        - As : scipy.sparse.csr_matrix
            Local stiffness matrix.
        - Xi : scipy.sparse.dia_matrix
            Partition of unity diagonal matrix.
        - R : scipy.sparse.csr_matrix
            Local-to-global mapping matrix.
        - non_dirichlet_dofs : ndarray
            Indices of degrees of freedom not on Dirichlet boundary.
        - interior_dofs : ndarray
            Indices of interior degrees of freedom for the subdomain.
    """


    (xR,xL,yR,yL,ol,os,Nx,Ny,nx,ny,nDom,coeff_A_function,deg,nloc,i_subdom,coord_global, dirichlet_boundary, robin_boundary, perturbation_parameter, rho, bool_ring) = parameters

    d_x = xR - xL
    d_y = yR - yL

    i = np.rint(np.mod(i_subdom, Nx))
    j = np.rint((i_subdom - i)/ Nx) 

    hx = d_x / nx
    hy = d_y / ny

    # Coordinates of the nonoverlapping region of the subdomain, i.e. into which no other subdomain (omega) overlaps
    x0_no_overlap = np.array([xL + d_x * i / Nx + ol * hx, yL + d_y * j / Ny + ol * hy]) 
    x1_no_overlap = np.array([xL + d_x * (i + 1) / Nx - ol * hx, yL + d_y * (j + 1) / Ny - ol * hy])

    x0_ommin = x0_no_overlap
    x1_ommin = x1_no_overlap

    # Coordinates of the inner rectangle used to create R
    x0_ometa= np.array([xL + d_x * i / Nx + (ol +1)* hx, yL + d_y * j / Ny + (ol + 1) * hy]) 
    x1_ometa = np.array([xL + d_x * (i + 1) / Nx - (ol + 1) * hx, yL + d_y * (j + 1) / Ny - (ol + 1) * hy])


    # Coordinates of the inner rectangle used to create R^*
    x0_osmin = np.array([xL + d_x * i / Nx + (ol + os) * hx , yL + d_y * j / Ny + (ol + os) * hy ]) 
    x1_osmin = np.array([xL + d_x * (i + 1) / Nx - (ol + os) * hx , yL + d_y * (j + 1) / Ny - (ol + os) * hy])

    # Need to take care of the case, when the subdomain is on the boundary of the global domain,
    # then we do not want to extend the subdomain beyond the global domain boundaries
    if i == 0:
        x0_no_overlap[0] = xL
        x0_ommin[0] = xL
        x0_osmin[0] = xL
        x0_ometa[0] = xL
    if i == Nx-1:
        x1_no_overlap[0] = xR
        x1_ommin[0] = xR
        x1_osmin[0] = xR
        x1_ometa[0] = xR
    if j == 0:
        x0_no_overlap[1] = yL
        x0_ommin[1] = yL
        x0_osmin[1] = yL
        x0_ometa[1] = yL
    if j == Ny-1:
        x1_no_overlap[1] = yR
        x1_ommin[1] = yR
        x1_osmin[1] = yR
        x1_ometa[1] = yR


    # Create the subdomain omega_no_overlap, which is the nonoverlapping part of the subdomain
    # Subset of omega, into which no other subdomain overlaps
    omega_no_overlap = Omega(
            x0_no_overlap,
            x1_no_overlap,
            xR,
            xL,
            yR,
            yL,
            nx,
            ny
    ) 
    # Same as omoega_no_overlap
    omega_min = Omega(
            x0_ommin,
            x1_ommin,
            xR,
            xL,
            yR,
            yL,
            nx,
            ny
    )  

    # To create the oversampling ring R^*, this is the part that will be subtracted from the oversampling region omega_os
    omega_os_min = Omega(
            x0_osmin,
            x1_osmin,
            xR,
            xL,
            yR,
            yL,
            nx,
            ny
    )  

    # The overlapping subdomain omega, as in the paper.
    omega = Omega(
            [
                np.max([xL + d_x * i / Nx - ol * hx, xL]),
                np.max([yL + d_y * j / Ny - ol * hy, yL]),
            ],
            [
                np.min([xL + d_x * (i + 1) / Nx + ol * hx, xR]),
                np.min([yL + d_y * (j + 1) / Ny + ol * hy, yR]),
            ],
            xR,
            xL,
            yR,
            yL,
            nx,
            ny
    )
    # The oversampling region omega^* 
    omega_os = Omega(
            [
                np.max([xL + d_x * i / Nx - ol * hx - os * hx, xL]),
                np.max([yL + d_y * j / Ny - ol * hy - os * hy, yL]),
            ],
            [
                np.min([xL + d_x * (i + 1) / Nx + ol * hx + os * hx, xR]),
                np.min([yL + d_y * (j + 1) / Ny + ol * hy + os * hy, yR]),
            ],
            xR,
            xL,
            yR,
            yL,
            nx,
            ny
    )
    # Used to construct the additional layer of the ring, where the cut-off function eta is defined.
    omega_eta= Omega(
            x0_ometa,
            x1_ometa,
            xR,
            xL,
            yR,
            yL,
            nx,
            ny
    )  
    # Create rings 
    ring = Ring(omega_min, omega)             #  This is the overlap region of the current subdomain with other subdomains
    ring_os = Ring(omega_os_min, omega_os)    # The oversampling ring R^*
    ring_eta = Ring(omega_eta, omega_min)     # The additional layer of the ring, where the cut- off function eta is defined.
                                              # With the notation in the paper, we have R = ring \cup \ring_eta

    # Build a submesh for the oversampling domain
    submesh = create_rectangle(
        comm=MPI.COMM_WORLD,
        points=(omega_os.x0, omega_os.x1),
        n=(omega_os.nx, omega_os.ny),
        cell_type=CellType.quadrilateral,
    )
    # Build local function space
    Vs = functionspace(submesh, ("Lagrange", deg))
    
    # Create local to global dof mapping for this subdomain
    loc2glob_os = make_submesh_local_to_global_map(coord_global, Vs.tabulate_dof_coordinates(), nx, ny, deg)

    # Create restriction matrix R
    data = np.ones(loc2glob_os.shape[0])
    row_ind = np.arange(loc2glob_os.shape[0])
    col_ind = loc2glob_os.astype(int)
    R = csr_array(
            (data, (row_ind, col_ind)), shape=(loc2glob_os.shape[0], coord_global.shape[0])
    )

    # Create partition of unity matrix
    pu_dofs_1 = locate_dofs_geometrical(Vs, omega_no_overlap.inside)     # locate dofs on nonoverlapping part
    pu_dofs_0 = locate_dofs_geometrical(Vs, lambda x : np.logical_not(omega.inside(x)))     # locate dofs not on omega

    pu_dofs_x_limited = locate_dofs_geometrical(Vs, lambda x : np.logical_and(x[0] >= x0_no_overlap[0] -1e-10, x[0] <= x1_no_overlap[0] + 1e-10)) # dofs whose x coordinates are limited by omega_no_overlap x boundaries
    pu_dofs_y_limited = locate_dofs_geometrical(Vs, lambda x : np.logical_and(x[1] >= x0_no_overlap[1] -1e-10, x[1] <= x1_no_overlap[1] + 1e-10)) # dofs whose x coordinates are limited by omega_no_overlap x boundaries

    midpoint_no_overlap = omega_no_overlap.x0 + (omega_no_overlap.x1 - omega_no_overlap.x0)/2 # midpoint of omega_no_overlap 
    L_no_overlap = (omega_no_overlap.x1 - omega_no_overlap.x0)/2 # half side lenght of omega_no_overlap
    l_band_x = np.max([(omega.x1[0] - omega_no_overlap.x1[0]), np.abs(omega.x0[0] - omega_no_overlap.x0[0])]) # distance between boundary of omega and boundary of omega_no_overlap
    l_band_y = np.max([(omega.x1[1] - omega_no_overlap.x1[1]), np.abs(omega.x0[1] - omega_no_overlap.x0[1])]) # distance between boundary of omega and boundary of omega_no_overlap
    l_band = np.array([l_band_x, l_band_y])
    b_pu = L_no_overlap / l_band + 1

    xi_local = Function(Vs)
    xi_local.interpolate(lambda x : ((- b_pu[0] / (L_no_overlap[0] + l_band[0])) * np.abs(x[0]-midpoint_no_overlap[0]) + b_pu[0]) * (- (b_pu[1] / (L_no_overlap[1] + l_band[1])) * np.abs(x[1]-midpoint_no_overlap[1]) + b_pu[1]))

    xi_local_x = Function(Vs)
    xi_local_x.interpolate(lambda x : (- b_pu[0] / (L_no_overlap[0] + l_band[0])) * np.abs(x[0]-midpoint_no_overlap[0]) + b_pu[0])

    xi_local_y = Function(Vs)
    xi_local_y.interpolate(lambda x : (- b_pu[1] / (L_no_overlap[1] + l_band[1])) * np.abs(x[1]-midpoint_no_overlap[1]) + b_pu[1])

    xi_local.vector.array[pu_dofs_x_limited] = xi_local_y.vector.array[pu_dofs_x_limited]
    xi_local.vector.array[pu_dofs_y_limited] = xi_local_x.vector.array[pu_dofs_y_limited]
 
    xi_local.vector.array[pu_dofs_1] = 1.0
    xi_local.vector.array[pu_dofs_0] = 0.0

    Xi = diags(xi_local.vector.array)

    # Cut-off function eta at interior of ring:
    eta = Function(Vs)
    eta.vector.array[:] = 1
    eta.vector.array[locate_dofs_geometrical(Vs, omega_eta.inside)] = 0
    Eta = diags(eta.vector.array)
    
    # Create local coefficient function for the subdomain
    element = ('Lagrange', deg)
    V_coeff = functionspace(submesh, element)
    coeff_A = Function(V_coeff)
    coeff_A.interpolate(coeff_A_function)
        
    # coeff_A.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    # Define indicator function for oversampling ring, ring, ring_eta
    chi_ring_os = Function(functionspace(submesh, ("DG", 0)))
    chi_ring = Function(functionspace(submesh, ("DG", 0)))
    chi_ring_eta = Function(functionspace(submesh, ("DG", 0)))

    if bool_ring:  
        chi_ring_os.interpolate(lambda x: ring_os.inside(x))

        chi_ring.interpolate(lambda x: ring.inside(x))

        chi_ring_eta.interpolate(lambda x: ring_eta.inside(x))

    else:
        chi_ring_os.interpolate(lambda x: 1 + 0 * x[0])

    # Assemble local stiffness matrix for eigenvalue problem 
    # If the ring method is used, we only assemble over the 
    # oversampling ring, which is why the indicator function chi_ring_os is used.
    # If the ring method is not used, we assemble over the whole oversampling region omega_os.
    u, v = TrialFunction(Vs), TestFunction(Vs)
    a = inner(chi_ring_os * coeff_A * grad(u), grad(v)) * dx
    A_constraint = assemble_matrix(form(a)).to_scipy()
    As = assemble_matrix(form(a)).to_scipy() 
    
    # Prepare matrix for local eigenproblem, corresponding to 
    # chosen scalar product. 
    u, v = TrialFunction(Vs), TestFunction(Vs)
    a_plus = inner(chi_ring_os * coeff_A * grad(u), grad(v)) * dx
    A_plus = assemble_matrix(form(a_plus)).to_scipy()

    # Dofs on oversampling ring or oversampling domain
    if bool_ring:
        all_dofs = locate_dofs_geometrical(Vs, ring_os.inside)
    else:
        all_dofs = np.arange(Vs.dofmap.index_map.size_global)

    dirichlet_dofs = np.intersect1d(all_dofs, locate_dofs_geometrical(Vs, dirichlet_boundary))
    robin_dofs = np.intersect1d(all_dofs, locate_dofs_geometrical(Vs, robin_boundary))         # Artefact of tests with Helmholtz, will be empty
    non_dirichlet_dofs = np.setdiff1d(all_dofs, dirichlet_dofs)

    
    if bool_ring:
        # Locate interior dofs
        interior_dofs = locate_dofs_geometrical(Vs, ring_os.interior)     
        # Locate interior bdry dofs
        interior_bdry_dofs = locate_dofs_geometrical(Vs, ring_os.on_interior_boundary)
    else:
        # Locate interior dofs
        interior_dofs = locate_dofs_geometrical(Vs, omega_os.interior)     
        # Locate interior bdry dofs
        interior_bdry_dofs = locate_dofs_geometrical(Vs, omega_os.on_interior_boundary)
    
    interior_and_robin_bdry_dofs = np.union1d(interior_dofs, robin_dofs)
    interior_bdry_and_robin_bdry_dofs = np.union1d(interior_bdry_dofs, robin_dofs)

    # Create lhs matrix for the local eigenproblem

    MA = bmat(
            [
                [
                    A_plus[non_dirichlet_dofs,:][:, non_dirichlet_dofs],
                    A_constraint[:, interior_and_robin_bdry_dofs][non_dirichlet_dofs,:].conj(),
                ],
                [
                    A_constraint[interior_and_robin_bdry_dofs, :][:,non_dirichlet_dofs],
                    csr_matrix((len(interior_and_robin_bdry_dofs), len(interior_and_robin_bdry_dofs))),
                ],
            ]
        )
    # Create rhs matrix for the local eigenproblem
    if bool_ring:
        a_ring = inner((chi_ring + chi_ring_eta) * coeff_A * grad(u), grad(v)) * dx
        A_ring = assemble_matrix(form(a_ring)).to_scipy()
        A_omega_xi = (Eta.dot(Xi)).T.dot(A_ring.dot(Eta.dot(Xi)))
    
    else:
        A_omega_xi = Xi.T.dot(A_plus.dot(Xi))

    MB = bmat(
            [
                [
                    A_omega_xi[non_dirichlet_dofs,:][:, non_dirichlet_dofs],
                    csr_matrix(
                        (len(non_dirichlet_dofs), len(interior_and_robin_bdry_dofs))
                    ),
                ],
                [
                    csr_matrix(
                        (len(interior_and_robin_bdry_dofs), len(non_dirichlet_dofs))
                    ),
                    csr_matrix((len(interior_and_robin_bdry_dofs), len(interior_and_robin_bdry_dofs))),
                ],
            ]
    )

    if rho > 0.0:
        # This if statement checks whether the eigenvalue cutoff parameter `rho` is greater than zero.
        # If so, it starts with an initial guess for the number of eigenvectors (`nloc_tmp`) and increases it in steps (`nloc_step`)
        # until all eigenvalues above the cutoff (`rho**2`) are found.
        # The eigenvalues and eigenvectors are sorted, and only those with eigenvalues above the cutoff are kept.
        # If `rho` is not greater than zero, a fixed number of eigenvectors (`nloc`) is computed.
        nloc_tmp = np.rint(len(interior_bdry_and_robin_bdry_dofs) / 10 + 1).astype(int)
        nloc_step = np.rint(len(interior_bdry_and_robin_bdry_dofs) / 10 + 1).astype(int)
        while True:  
            # Solve eigenproblem 
            vals, vecs = eigsh(MA, k=nloc_tmp, M=MB, sigma=0)
            vals = 1/vals
            vals = np.abs(vals)          # Sometimes get negative eigenvalue corresponding to the zero eigenvalue, so take absolute value for proper ordering
            vals, vecs = helper.sort_eigenpairs(vals, vecs)

            idx_values = np.where(vals >= rho**2)[0]
            _, nloc_cutoff = idx_values[[0, -1]] + 1
            # print("nloc_cutoff", nloc_cutoff[i])
            if nloc_cutoff == nloc_tmp:
                nloc_tmp = nloc_tmp + nloc_step
            else:
                vals = vals[:nloc_cutoff]
                vecs = vecs[:, :nloc_cutoff]
                break
    else:
        # Solve eigenproblem 
        
        vals, vecs = eigsh(MA, k=nloc, M=MB, sigma=0)
        vals = 1/vals
        vals = np.abs(vals)          # Sometimes get negative eigenvalue corresponding to the zero eigenvalue, so take absolute value for proper ordering
        vals, vecs = helper.sort_eigenpairs(vals, vecs)
        
    vecs_tmp = np.zeros((Xi.shape[0], vecs.shape[1]))
    vecs_tmp[non_dirichlet_dofs,:] = vecs[:len(non_dirichlet_dofs), :]
    vecs_tmp, _ = np.linalg.qr(vecs_tmp)

    if bool_ring:
        # Extend a-harmonically to the interior
        
        # Assemble matrix over ommin
        chi_ommin = Function(functionspace(submesh, ("DG", 0)))
        chi_ommin.interpolate(lambda x: omega_min.inside(x))
        u, v = TrialFunction(Vs), TestFunction(Vs)
        a = inner(chi_ommin * coeff_A * grad(u), grad(v)) * dx
        A_ommin = assemble_matrix(form(a)).to_scipy() 

        # Locate interior dofs of ommin
        ommin_interior_dofs = locate_dofs_geometrical(Vs, omega_min.interior)

        # Harmonically extend
        u_g = vecs_tmp
        u_0 = spsolve(A_ommin[ommin_interior_dofs,:][:,ommin_interior_dofs], - A_ommin.dot(u_g)[ommin_interior_dofs])
        if nloc > 1:
            vecs_tmp[ommin_interior_dofs, :] = u_g[ommin_interior_dofs, :] + u_0
        else:
            vecs_tmp[ommin_interior_dofs,0] = u_g[ommin_interior_dofs,0] + u_0

        # assemble matrix for local solves, which will be return of this function
        a = inner(coeff_A * grad(u), grad(v)) * dx
        As = assemble_matrix(form(a)).to_scipy() 

        # interior dofs is used for the local solves
        interior_dofs = locate_dofs_geometrical(Vs, omega_os.interior)

    return (vecs_tmp, vals, As, Xi, R, non_dirichlet_dofs, interior_dofs)

def computeSubdomainFNO(parameters):
    """
    Constructs and solves a local eigenproblem for a subdomain.
    This function partitions the global domain into overlapping subdomains, builds local meshes and function spaces, 
    computes partition of unity and cut-off functions, assembles local stiffness matrices, and solves a generalized 
    eigenvalue problem to obtain local basis functions. It supports oversampling and ring-based localization strategies, 
    and handles Dirichlet boundary conditions.
    Parameters
    ----------
    parameters : tuple
        A tuple containing all necessary parameters for subdomain construction and computation, including:
        - Domain bounds (xR, xL, yR, yL)
        - Overlap and oversampling sizes (ol, os)
        - Number of subdomains and mesh resolution (Nx, Ny, nx, ny, nDom)
        - Coefficient function and polynomial degree (coeff_A_function, deg)
        - Number of local basis functions (nloc)
        - Subdomain index (k, i_subdom)
        - Global coordinate array (coord_global)
        - Boundary locator functions (dirichlet_boundary, robin_boundary)
        - Perturbation parameter (perturbation_parameter)
        - Eigenvalue cutoff (rho)
        - Ring localization flag (bool_ring)
    Returns
    -------
    tuple
        A tuple containing:
        - coeff_A_FNO: ndarray
            PDE coefficient on local subdomain.
        - eig_func_2d: ndarray
            Eigenfunctions on local subdomain of shape [nx, ny, nloc]. 
        - vecs_tmp: ndarray
            Eigenfunctions on local subdomain of shape [nx*ny, nloc].
        - vals: ndarray
            Eigenvalues off the nloc eigenfunctions.
    """


    (xR,xL,yR,yL,ol,os,Nx,Ny,nx,ny,nDom,coeff_A_function,deg,nloc,i_subdom,coord_global, dirichlet_boundary, robin_boundary, perturbation_parameter, rho, bool_ring) = parameters

    d_x = xR - xL
    d_y = yR - yL

    i = np.rint(np.mod(i_subdom, Nx))
    j = np.rint((i_subdom - i)/ Nx) 

    hx = d_x / nx
    hy = d_y / ny

    # Coordinates of the nonoverlapping region of the subdomain, i.e. into which no other subdomain (omega) overlaps
    x0_no_overlap = np.array([xL + d_x * i / Nx + ol * hx, yL + d_y * j / Ny + ol * hy]) 
    x1_no_overlap = np.array([xL + d_x * (i + 1) / Nx - ol * hx, yL + d_y * (j + 1) / Ny - ol * hy])

    x0_ommin = x0_no_overlap
    x1_ommin = x1_no_overlap

    # Coordinates of the inner rectangle used to create R
    x0_ometa= np.array([xL + d_x * i / Nx + (ol +1)* hx, yL + d_y * j / Ny + (ol + 1) * hy]) 
    x1_ometa = np.array([xL + d_x * (i + 1) / Nx - (ol + 1) * hx, yL + d_y * (j + 1) / Ny - (ol + 1) * hy])


    # Coordinates of the inner rectangle used to create R^*
    x0_osmin = np.array([xL + d_x * i / Nx + (ol + os) * hx , yL + d_y * j / Ny + (ol + os) * hy ]) 
    x1_osmin = np.array([xL + d_x * (i + 1) / Nx - (ol + os) * hx , yL + d_y * (j + 1) / Ny - (ol + os) * hy])

    # Need to take care of the case, when the subdomain is on the boundary of the global domain,
    # then we do not want to extend the subdomain beyond the global domain boundaries
    if i == 0:
        x0_no_overlap[0] = xL
        x0_ommin[0] = xL
        x0_osmin[0] = xL
        x0_ometa[0] = xL
    if i == Nx-1:
        x1_no_overlap[0] = xR
        x1_ommin[0] = xR
        x1_osmin[0] = xR
        x1_ometa[0] = xR
    if j == 0:
        x0_no_overlap[1] = yL
        x0_ommin[1] = yL
        x0_osmin[1] = yL
        x0_ometa[1] = yL
    if j == Ny-1:
        x1_no_overlap[1] = yR
        x1_ommin[1] = yR
        x1_osmin[1] = yR
        x1_ometa[1] = yR


    # Create the subdomain omega_no_overlap, which is the nonoverlapping part of the subdomain
    # Subset of omega, into which no other subdomain overlaps
    omega_no_overlap = Omega(
            x0_no_overlap,
            x1_no_overlap,
            xR,
            xL,
            yR,
            yL,
            nx,
            ny
    ) 
    # Same as omoega_no_overlap
    omega_min = Omega(
            x0_ommin,
            x1_ommin,
            xR,
            xL,
            yR,
            yL,
            nx,
            ny
    )  

    # To create the oversampling ring R^*, this is the part that will be subtracted from the oversampling region omega_os
    omega_os_min = Omega(
            x0_osmin,
            x1_osmin,
            xR,
            xL,
            yR,
            yL,
            nx,
            ny
    )  

    # The overlapping subdomain omega, as in the paper.
    omega = Omega(
            [
                np.max([xL + d_x * i / Nx - ol * hx, xL]),
                np.max([yL + d_y * j / Ny - ol * hy, yL]),
            ],
            [
                np.min([xL + d_x * (i + 1) / Nx + ol * hx, xR]),
                np.min([yL + d_y * (j + 1) / Ny + ol * hy, yR]),
            ],
            xR,
            xL,
            yR,
            yL,
            nx,
            ny
    )
    # The oversampling region omega^* 
    omega_os = Omega(
            [
                np.max([xL + d_x * i / Nx - ol * hx - os * hx, xL]),
                np.max([yL + d_y * j / Ny - ol * hy - os * hy, yL]),
            ],
            [
                np.min([xL + d_x * (i + 1) / Nx + ol * hx + os * hx, xR]),
                np.min([yL + d_y * (j + 1) / Ny + ol * hy + os * hy, yR]),
            ],
            xR,
            xL,
            yR,
            yL,
            nx,
            ny
    )
    # Used to construct the additional layer of the ring, where the cut-off function eta is defined.
    omega_eta= Omega(
            x0_ometa,
            x1_ometa,
            xR,
            xL,
            yR,
            yL,
            nx,
            ny
    )  
    # Create rings 
    ring = Ring(omega_min, omega)             #  This is the overlap region of the current subdomain with other subdomains
    ring_os = Ring(omega_os_min, omega_os)    # The oversampling ring R^*
    ring_eta = Ring(omega_eta, omega_min)     # The additional layer of the ring, where the cut- off function eta is defined.
                                              # With the notation in the paper, we have R = ring \cup \ring_eta

    # Build a submesh for the oversampling domain
    submesh = create_rectangle(
        comm=MPI.COMM_WORLD,
        points=(omega_os.x0, omega_os.x1),
        n=(omega_os.nx, omega_os.ny),
        cell_type=CellType.quadrilateral,
    )
    # Build local function space
    Vs = functionspace(submesh, ("Lagrange", deg))
    
    # Create local to global dof mapping for this subdomain
    loc2glob_os = make_submesh_local_to_global_map(coord_global, Vs.tabulate_dof_coordinates(), nx, ny, deg)

    # Create restriction matrix R
    data = np.ones(loc2glob_os.shape[0])
    row_ind = np.arange(loc2glob_os.shape[0])
    col_ind = loc2glob_os.astype(int)
    R = csr_array(
            (data, (row_ind, col_ind)), shape=(loc2glob_os.shape[0], coord_global.shape[0])
    )

    # Create partition of unity matrix
    pu_dofs_1 = locate_dofs_geometrical(Vs, omega_no_overlap.inside)     # locate dofs on nonoverlapping part
    pu_dofs_0 = locate_dofs_geometrical(Vs, lambda x : np.logical_not(omega.inside(x)))     # locate dofs not on omega

    pu_dofs_x_limited = locate_dofs_geometrical(Vs, lambda x : np.logical_and(x[0] >= x0_no_overlap[0] -1e-10, x[0] <= x1_no_overlap[0] + 1e-10)) # dofs whose x coordinates are limited by omega_no_overlap x boundaries
    pu_dofs_y_limited = locate_dofs_geometrical(Vs, lambda x : np.logical_and(x[1] >= x0_no_overlap[1] -1e-10, x[1] <= x1_no_overlap[1] + 1e-10)) # dofs whose x coordinates are limited by omega_no_overlap x boundaries

    midpoint_no_overlap = omega_no_overlap.x0 + (omega_no_overlap.x1 - omega_no_overlap.x0)/2 # midpoint of omega_no_overlap 
    L_no_overlap = (omega_no_overlap.x1 - omega_no_overlap.x0)/2 # half side lenght of omega_no_overlap
    l_band_x = np.max([(omega.x1[0] - omega_no_overlap.x1[0]), np.abs(omega.x0[0] - omega_no_overlap.x0[0])]) # distance between boundary of omega and boundary of omega_no_overlap
    l_band_y = np.max([(omega.x1[1] - omega_no_overlap.x1[1]), np.abs(omega.x0[1] - omega_no_overlap.x0[1])]) # distance between boundary of omega and boundary of omega_no_overlap
    l_band = np.array([l_band_x, l_band_y])
    b_pu = L_no_overlap / l_band + 1

    xi_local = Function(Vs)
    xi_local.interpolate(lambda x : ((- b_pu[0] / (L_no_overlap[0] + l_band[0])) * np.abs(x[0]-midpoint_no_overlap[0]) + b_pu[0]) * (- (b_pu[1] / (L_no_overlap[1] + l_band[1])) * np.abs(x[1]-midpoint_no_overlap[1]) + b_pu[1]))

    xi_local_x = Function(Vs)
    xi_local_x.interpolate(lambda x : (- b_pu[0] / (L_no_overlap[0] + l_band[0])) * np.abs(x[0]-midpoint_no_overlap[0]) + b_pu[0])

    xi_local_y = Function(Vs)
    xi_local_y.interpolate(lambda x : (- b_pu[1] / (L_no_overlap[1] + l_band[1])) * np.abs(x[1]-midpoint_no_overlap[1]) + b_pu[1])

    xi_local.vector.array[pu_dofs_x_limited] = xi_local_y.vector.array[pu_dofs_x_limited]
    xi_local.vector.array[pu_dofs_y_limited] = xi_local_x.vector.array[pu_dofs_y_limited]
 
    xi_local.vector.array[pu_dofs_1] = 1.0
    xi_local.vector.array[pu_dofs_0] = 0.0

    Xi = diags(xi_local.vector.array)

    # Cut-off function eta at interior of ring:
    eta = Function(Vs)
    eta.vector.array[:] = 1
    eta.vector.array[locate_dofs_geometrical(Vs, omega_eta.inside)] = 0
    Eta = diags(eta.vector.array)
    
    # Create local coefficient function for the subdomain
    element = ('Lagrange', deg)
    V_coeff = functionspace(submesh, element)
    coeff_A = Function(V_coeff)
    coeff_A.interpolate(coeff_A_function)

    A_ = coeff_A.vector.array
    dof_coord = V_coeff.tabulate_dof_coordinates()
    # this formula only works when the coefficient is a DG0 function on a rectangular mesh
    #indices = (dof_coord[:,:-1] - omega_os.x0 - ((np.array(omega_os.x1) - np.array(omega_os.x0))/np.array([2*omega_os.nx, 2*omega_os.ny])) )*(np.array([omega_os.nx, omega_os.ny])/(np.array(omega_os.x1)-np.array(omega_os.x0))) # fenics always thinks in 3D
    #input_indices = np.round(indices).astype(np.int32)
    indices = (dof_coord[:,:-1] - omega_os.x0)*np.array([nx, ny]) # fenics always thinks in 3D
    input_indices = np.round(indices).astype(np.int32)
    coeff_A_FNO = np.zeros((omega_os.nx+1, omega_os.ny+1))

    for i, idxs in enumerate(input_indices):
        ix, iy = idxs
        coeff_A_FNO[ix, iy] = A_[i]
        
    # Define indicator function for oversampling ring, ring, ring_eta
    chi_ring_os = Function(functionspace(submesh, ("DG", 0)))
    chi_ring = Function(functionspace(submesh, ("DG", 0)))
    chi_ring_eta = Function(functionspace(submesh, ("DG", 0)))

    if bool_ring:  
        chi_ring_os.interpolate(lambda x: ring_os.inside(x))

        chi_ring.interpolate(lambda x: ring.inside(x))

        chi_ring_eta.interpolate(lambda x: ring_eta.inside(x))

    else:
        chi_ring_os.interpolate(lambda x: 1 + 0 * x[0])

    # Assemble local stiffness matrix for eigenvalue problem 
    # If the ring method is used, we only assemble over the 
    # oversampling ring, which is why the indicator function chi_ring_os is used.
    # If the ring method is not used, we assemble over the whole oversampling region omega_os.
    u, v = TrialFunction(Vs), TestFunction(Vs)
    a = inner(chi_ring_os * coeff_A * grad(u), grad(v)) * dx
    A_constraint = assemble_matrix(form(a)).to_scipy()
    As = assemble_matrix(form(a)).to_scipy() 
    
    # Prepare matrix for local eigenproblem, corresponding to 
    # chosen scalar product. 
    u, v = TrialFunction(Vs), TestFunction(Vs)
    a_plus = inner(chi_ring_os * coeff_A * grad(u), grad(v)) * dx
    A_plus = assemble_matrix(form(a_plus)).to_scipy()

    # Dofs on oversampling ring or oversampling domain
    if bool_ring:
        all_dofs = locate_dofs_geometrical(Vs, ring_os.inside)
    else:
        all_dofs = np.arange(Vs.dofmap.index_map.size_global)

    dirichlet_dofs = np.intersect1d(all_dofs, locate_dofs_geometrical(Vs, dirichlet_boundary))
    robin_dofs = np.intersect1d(all_dofs, locate_dofs_geometrical(Vs, robin_boundary))         # Artefact of tests with Helmholtz, will be empty
    non_dirichlet_dofs = np.setdiff1d(all_dofs, dirichlet_dofs)

    
    if bool_ring:
        # Locate interior dofs
        interior_dofs = locate_dofs_geometrical(Vs, ring_os.interior)     
        # Locate interior bdry dofs
        interior_bdry_dofs = locate_dofs_geometrical(Vs, ring_os.on_interior_boundary)
    else:
        # Locate interior dofs
        interior_dofs = locate_dofs_geometrical(Vs, omega_os.interior)     
        # Locate interior bdry dofs
        interior_bdry_dofs = locate_dofs_geometrical(Vs, omega_os.on_interior_boundary)
    
    interior_and_robin_bdry_dofs = np.union1d(interior_dofs, robin_dofs)
    interior_bdry_and_robin_bdry_dofs = np.union1d(interior_bdry_dofs, robin_dofs)

    # Create lhs matrix for the local eigenproblem

    MA = bmat(
            [
                [
                    A_plus[non_dirichlet_dofs,:][:, non_dirichlet_dofs],
                    A_constraint[:, interior_and_robin_bdry_dofs][non_dirichlet_dofs,:].conj(),
                ],
                [
                    A_constraint[interior_and_robin_bdry_dofs, :][:,non_dirichlet_dofs],
                    csr_matrix((len(interior_and_robin_bdry_dofs), len(interior_and_robin_bdry_dofs))),
                ],
            ]
        )
    # Create rhs matrix for the local eigenproblem
    if bool_ring:
        a_ring = inner((chi_ring + chi_ring_eta) * coeff_A * grad(u), grad(v)) * dx
        A_ring = assemble_matrix(form(a_ring)).to_scipy()
        A_omega_xi = (Eta.dot(Xi)).T.dot(A_ring.dot(Eta.dot(Xi)))
    
    else:
        A_omega_xi = Xi.T.dot(A_plus.dot(Xi))

    MB = bmat(
            [
                [
                    A_omega_xi[non_dirichlet_dofs,:][:, non_dirichlet_dofs],
                    csr_matrix(
                        (len(non_dirichlet_dofs), len(interior_and_robin_bdry_dofs))
                    ),
                ],
                [
                    csr_matrix(
                        (len(interior_and_robin_bdry_dofs), len(non_dirichlet_dofs))
                    ),
                    csr_matrix((len(interior_and_robin_bdry_dofs), len(interior_and_robin_bdry_dofs))),
                ],
            ]
    )

    if rho > 0.0:
        # This if statement checks whether the eigenvalue cutoff parameter `rho` is greater than zero.
        # If so, it starts with an initial guess for the number of eigenvectors (`nloc_tmp`) and increases it in steps (`nloc_step`)
        # until all eigenvalues above the cutoff (`rho**2`) are found.
        # The eigenvalues and eigenvectors are sorted, and only those with eigenvalues above the cutoff are kept.
        # If `rho` is not greater than zero, a fixed number of eigenvectors (`nloc`) is computed.
        nloc_tmp = np.rint(len(interior_bdry_and_robin_bdry_dofs) / 10 + 1).astype(int)
        nloc_step = np.rint(len(interior_bdry_and_robin_bdry_dofs) / 10 + 1).astype(int)
        while True:  
            # Solve eigenproblem 
            vals, vecs = eigsh(MA, k=nloc_tmp, M=MB, sigma=0)
            vals = 1/vals
            vals = np.abs(vals)          # Sometimes get negative eigenvalue corresponding to the zero eigenvalue, so take absolute value for proper ordering
            vals, vecs = helper.sort_eigenpairs(vals, vecs)

            idx_values = np.where(vals >= rho**2)[0]
            _, nloc_cutoff = idx_values[[0, -1]] + 1
            # print("nloc_cutoff", nloc_cutoff[i])
            if nloc_cutoff == nloc_tmp:
                nloc_tmp = nloc_tmp + nloc_step
            else:
                vals = vals[:nloc_cutoff]
                vecs = vecs[:, :nloc_cutoff]
                break
    else:
        # Solve eigenproblem 
        
        vals, vecs = eigsh(MA, k=nloc, M=MB, sigma=0)
        vals = 1/vals
        vals = np.abs(vals)          # Sometimes get negative eigenvalue corresponding to the zero eigenvalue, so take absolute value for proper ordering
        vals, vecs = helper.sort_eigenpairs(vals, vecs)
        
    vecs_tmp = np.zeros((Xi.shape[0], vecs.shape[1]))
    vecs_tmp[non_dirichlet_dofs,:] = vecs[:len(non_dirichlet_dofs), :]
    vecs_tmp, _ = np.linalg.qr(vecs_tmp) # orthogonalize in standard scalar product (this does not brake msgfem)

    if bool_ring:
        # Extend a-harmonically to the interior
        
        # Assemble matrix over ommin
        chi_ommin = Function(functionspace(submesh, ("DG", 0)))
        chi_ommin.interpolate(lambda x: omega_min.inside(x))
        u, v = TrialFunction(Vs), TestFunction(Vs)
        a = inner(chi_ommin * coeff_A * grad(u), grad(v)) * dx
        A_ommin = assemble_matrix(form(a)).to_scipy() 

        # Locate interior dofs of ommin
        ommin_interior_dofs = locate_dofs_geometrical(Vs, omega_min.interior)

        # Harmonically extend
        u_g = vecs_tmp
        u_0 = spsolve(A_ommin[ommin_interior_dofs,:][:,ommin_interior_dofs], - A_ommin.dot(u_g)[ommin_interior_dofs])
        if nloc > 1:
            vecs_tmp[ommin_interior_dofs, :] = u_g[ommin_interior_dofs, :] + u_0
        else:
            vecs_tmp[ommin_interior_dofs,0] = u_g[ommin_interior_dofs,0] + u_0

        # assemble matrix for local solves, which will be return of this function
        a = inner(coeff_A * grad(u), grad(v)) * dx
        As = assemble_matrix(form(a)).to_scipy() 

        # interior dofs is used for the local solves
        interior_dofs = locate_dofs_geometrical(Vs, omega_os.interior)

    # save local basis functions and eigenvalues and reshape to fit the 2D domain
    dof_coord = Vs.tabulate_dof_coordinates()
    indices = (dof_coord[:,:-1] - omega_os.x0)*np.array([nx, ny]) # fenics always thinks in 3D
    input_indices = np.round(indices).astype(np.int32)
    eigenfunctions_2d = np.zeros((omega_os.nx+1, omega_os.ny+1, nloc))

    for i, idxs in enumerate(input_indices):
        ix, iy = idxs
        eigenfunctions_2d[ix, iy, :] = vecs_tmp[i]
    
    return (coeff_A_FNO, eigenfunctions_2d, vecs_tmp, vals, input_indices)


# def gfem_solve(rhs, A_gfem, local_data, nDom, nloc_cut): 
#     start = time.time()
#     AH, M_basis = assembleCoarseSpace(A_gfem, local_data, nloc_cut)
#     end = time.time()
#     print("Assemble Coarse space:", end - start)

#     u_par_glob = np.zeros(rhs.shape[0])
#     for i in range(nDom):
#         u_par_glob = u_par_glob + local_data[i][2]

#     rhsH = M_basis.T.dot((rhs - A_gfem.dot(u_par_glob)))

#     opts = PETSc.Options()
#     opts.setValue("-pc_type", "lu")
#     opts.setValue("-pc_factor_mat_solver_type", "mumps")

#     AH_petsc = scipy_to_petsc_matrix(AH)

#     ksp = PETSc.KSP().create()
#     ksp.setOperators(AH_petsc)
#     ksp.setFromOptions()
#     start = time.time()

#     solH = petsc_solve(ksp, AH_petsc, rhsH)
#     end = time.time()
#     print("time of coarse solve: ", end-start)

#     return u_par_glob + M_basis.dot(solH)
#     # return u_par_glob


def fem_solve(V, a, L, bc):
    """
    Solves a finite element problem using the provided variational form, linear form, and boundary conditions.
    Parameters
    ----------
    V : FunctionSpace
        The finite element function space in which to solve the problem.
    a : ufl.Form
        The bilinear form representing the left-hand side of the variational problem.
    L : ufl.Form
        The linear form representing the right-hand side of the variational problem.
    bc : DirichletBC or list of DirichletBC
        The boundary condition(s) to apply to the problem.
    Returns
    -------
    uh : Function
        The computed solution function in the given function space.
    """

    uh = Function(V)
    uh.name = "u"
    problem = LinearProblem(a, L, u=uh, petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}, bcs = [bc])
    problem.solve()
    return uh


def petsc_solve(ksp, A_petsc, p):
    """
    Solves a linear system using PETSc KSP solver.

    Parameters
    ----------
    ksp : petsc4py.PETSc.KSP
        The PETSc Krylov Subspace solver object.
    A_petsc : petsc4py.PETSc.Mat
        The PETSc matrix representing the system.
    p : np.ndarray
        The right-hand side vector.

    Returns
    -------
    np.ndarray
        The solution vector as a NumPy array.
    """
    b = PETSc.Vec().createWithArray(p)
    x = A_petsc.createVecRight()
    # print(p)
    ksp.solve(b, x)
    # print(x.getArray())
    return np.array(x.getArray())

def scipy_to_petsc_matrix(A_scipy):
    """
    Converts a SciPy sparse matrix (CSR format) to a PETSc AIJ matrix.
    Parameters
    ----------
    A_scipy : scipy.sparse.csr_matrix
        The input matrix in SciPy CSR (Compressed Sparse Row) format.
    Returns
    -------
    petsc_mat : petsc4py.PETSc.Mat
        The corresponding PETSc AIJ matrix.
    Description
    -----------
    """
    return PETSc.Mat().createAIJ(size=A_scipy.shape, csr=(A_scipy.indptr.astype(np.int32), A_scipy.indices.astype(np.int32), A_scipy.data))
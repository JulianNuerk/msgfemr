from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.mesh import create_rectangle, CellType, locate_entities_boundary, meshtags, locate_entities, create_box
from dolfinx.fem import functionspace, Function, locate_dofs_geometrical, assemble_matrix, form, Constant
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
from ufl import dx, grad, inner, TrialFunction, TestFunction, ds, FacetNormal, Measure
from ray.util.multiprocessing import Pool

import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import csr_array, diags, csr_matrix, bmat, hstack
from scipy.sparse.linalg import eigsh, splu, spsolve
from scipy.linalg import orth
import helper

import time

class Ring():
    def __init__(self, omega_inner, omega_outer):
        self.omega_inner = omega_inner
        self.omega_outer = omega_outer

    def inside(self, x):
        return np.logical_and(self.omega_outer.inside(x), np.logical_not(self.omega_inner.interior(x)))

    def boundary(self, x):
        return np.logical_and(self.inside(x), np.logical_or(self.omega_outer.boundary(x), self.omega_inner.boundary(x)))
    
    def interior(self, x):
        return np.logical_and(self.inside(x), np.logical_not(self.boundary(x)))
    
    def on_global_boundary(self, x):
        return self.omega_outer.on_global_boundary(x)

    def on_interior_boundary(self, x):
        return np.logical_and(self.boundary(x), np.logical_not(self.on_global_boundary(x)))

class Omega():
    def __init__(self, x0, x1, xR, xL, yR, yL, zR, zL, nx, ny, nz):
        self.x0 = x0
        self.x1 = x1
        self.xR = xR
        self.xL = xL
        self.yR = yR
        self.yL = yL
        self.zR = zR
        self.zL = zL
        hx = (xR - xL) / nx 
        hy = (yR - yL) / ny
        hz = (zR - zL) / nz
        self.nx = np.rint((x1[0] - x0[0]) / hx).astype(int)                     
        self.ny = np.rint((x1[1] - x0[1]) / hy).astype(int) 
        self.nz = np.rint((x1[2] - x0[2]) / hz).astype(int)


    def inside(self, x):
        return np.logical_and(np.logical_and(
                              np.logical_and(x[0] >= self.x0[0] - 1e-10, x[1] >= self.x0[1] - 1e-10), 
                              np.logical_and(x[0] <= self.x1[0] + 1e-10, x[1] <= self.x1[1] + 1e-10)),
                            np.logical_and(x[2] >= self.x0[2] - 1e-10, x[2] <= self.x1[2] + 1e-10))
                                
        
    def interior(self, x):
        return np.logical_and(np.logical_and(
                              np.logical_and(x[0] > self.x0[0] + 1e-10, x[1] > self.x0[1] + 1e-10), 
                              np.logical_and(x[0] < self.x1[0] - 1e-10, x[1] < self.x1[1] - 1e-10)),
                            np.logical_and(x[2] > self.x0[2] + 1e-10, x[2] < self.x1[2] - 1e-10))
                    

    def on_global_boundary(self, x):
        return np.logical_or(
            np.logical_or(x[0] < self.xL + 1e-10, x[0] > self.xR - 1e-10),
            np.logical_or(x[1] < self.yL + 1e-10, x[1] > self.yR - 1e-10),
            np.logical_or(x[2] < self.zL + 1e-10, x[2] > self.zR - 1e-10)
        )
    def not_on_global_boundary(self, x):
        return np.logical_not(self.on_global_boundary(x))


    def boundary(self, x):
        return np.logical_and(self.inside(x), np.logical_not(self.interior(x)))


    def on_interior_boundary(self, x):
        return np.logical_and(self.boundary(x), np.logical_not(self.on_global_boundary(x)))
    

def computeSubdomain(n_part, nloc, bool_ring, ol, os, nnz_computation = False):
    xR = 1
    xL = 0
    yR = xR
    yL = xL
    zR = xR
    zL = xL
    Nx = 3
    Ny = Nx
    Nz = Nx
    nx = 3 * n_part
    ny = nx 
    nz = nx
    nDom = Nx * Ny * Nz
    coeff_A_function = lambda x: np.ones(x.shape[1])
    deg = 1
    def robin_boundary(x):
        bool_tmp = np.isclose(x[1], -1)
        return bool_tmp

    def dirichlet_boundary(x):
        bool_tmp = np.logical_or(np.isclose(x[0], xL), np.isclose(x[0], xR))
        bool_tmp = np.logical_or(bool_tmp, np.isclose(x[1], yR))
        bool_tmp = np.logical_or(bool_tmp, np.isclose(x[1], yL))
        
        bool_tmp = np.logical_and(bool_tmp, np.logical_not(robin_boundary(x)))   
        return bool_tmp
    
    rho = 0.0

    d_x = xR - xL
    d_y = yR - yL
    d_z = zR - zL

    # Select the interior subdomain
    i = 1
    j = 1
    k = 1

    Nx = Ny = Nz = 3

    hx = d_x / nx
    hy = d_y / ny
    hz = d_z / nz

    x0_no_overlap = np.array([xL + d_x * i / Nx + ol * hx, yL + d_y * j / Ny + ol * hy, zL + d_z * k / Nz + ol * hz]) 
    x1_no_overlap = np.array([xL + d_x * (i + 1) / Nx - ol * hx, yL + d_y * (j + 1) / Ny - ol * hy, zL + d_z * (k + 1) / Nz - ol * hz])

    x0_ometa= np.array([xL + d_x * i / Nx + (ol +1)* hx, yL + d_y * j / Ny + (ol + 1) * hy, zL + d_z * k / Nz + (ol + 1) * hz]) 
    x1_ometa = np.array([xL + d_x * (i + 1) / Nx - (ol + 1) * hx, yL + d_y * (j + 1) / Ny - (ol + 1) * hy, zL + d_z * (k + 1) / Nz - (ol + 1) * hz])

    x0_ommin = x0_no_overlap
    x1_ommin = x1_no_overlap

    x0_osmin = np.array([xL + d_x * i / Nx + (ol + os) * hx , yL + d_y * j / Ny + (ol + os) * hy, zL + d_z * k / Nz + (ol + os) * hz])
    x1_osmin = np.array([xL + d_x * (i + 1) / Nx - (ol + os) * hx , yL + d_y * (j + 1) / Ny - (ol + os) * hy, zL + d_z * (k + 1) / Nz - (ol + os) * hz])


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
    if k == 0:
        x0_no_overlap[2] = zL
        x0_ommin[2] = zL
        x0_osmin[2] = zL
        x0_ometa[2] = zL
    if k == Nz-1:
        x1_no_overlap[2] = zR
        x1_ommin[2] = zR
        x1_osmin[2] = zR
        x1_ometa[2] = zR

    omega_no_overlap = Omega(
            x0_no_overlap,
            x1_no_overlap,
            xR,
            xL,
            yR,
            yL,
            zR,
            zL,
            nx,
            ny,
            nz
    )   # part of omega, into which no other omega overlaps

    omega_min = Omega(
            x0_ommin,
            x1_ommin,
            xR,
            xL,
            yR,
            yL,
            zR,
            zL,
            nx,
            ny,
            nz
    )

    omega_os_min = Omega(
            x0_osmin,
            x1_osmin,
            xR,
            xL,
            yR,
            yL,
            zR,
            zL,
            nx,
            ny,
            nz
    )


    omega = Omega(
            [
                np.max([xL + d_x * i / Nx - ol * hx, xL]),
                np.max([yL + d_y * j / Ny - ol * hy, yL]),
                np.max([zL + d_z * k / Nz - ol * hz, zL]),
            ],
            [
                np.min([xL + d_x * (i + 1) / Nx + ol * hx, xR]),
                np.min([yL + d_y * (j + 1) / Ny + ol * hy, yR]),
                np.min([zL + d_z * (k + 1) / Nz + ol * hz, zR]),
            ],
            xR,
            xL,
            yR,
            yL,
            zR,
            zL,
            nx,
            ny,
            nz
    )
    omega_os = Omega(
            [
                np.max([xL + d_x * i / Nx - ol * hx - os * hx, xL]),
                np.max([yL + d_y * j / Ny - ol * hy - os * hy, yL]),
                np.max([zL + d_z * k / Nz - ol * hz - os * hz, zL]),
            ],
            [
                np.min([xL + d_x * (i + 1) / Nx + ol * hx + os * hx, xR]),
                np.min([yL + d_y * (j + 1) / Ny + ol * hy + os * hy, yR]),
                np.min([zL + d_z * (k + 1) / Nz + ol * hz + os * hz, zR]),
            ],
            xR,
            xL,
            yR,
            yL,
            zR,
            zL,
            nx,
            ny,
            nz
    )

    omega_eta= Omega(
            x0_ometa,
            x1_ometa,
            xR,
            xL,
            yR,
            yL,
            zR,
            zL,
            nx,
            ny,
            nz
    )  

    ring = Ring(omega_min, omega)
    ring_os = Ring(omega_os_min, omega_os)
    ring_eta = Ring(omega_eta, omega_min)

    # Build a submesh for each domain
    submesh = create_box(
        comm=MPI.COMM_WORLD,
        points=(omega_os.x0, omega_os.x1),
        n=(omega_os.nx, omega_os.ny, omega_os.nz),
        cell_type=CellType.hexahedron,
    )
    # Build local function space
    Vs = functionspace(submesh, ("Lagrange", deg))
    
    midpoint_no_overlap = omega_no_overlap.x0 + (omega_no_overlap.x1 - omega_no_overlap.x0)/2 # midpoint of omega_no_overlap 
    L_no_overlap = (omega_no_overlap.x1 - omega_no_overlap.x0)/2 # half side lenght of omega_no_overlap
    l_band_x = np.max([(omega.x1[0] - omega_no_overlap.x1[0]), np.abs(omega.x0[0] - omega_no_overlap.x0[0])]) # distance between boundary of omega and boundary of omega_no_overlap
    l_band_y = np.max([(omega.x1[1] - omega_no_overlap.x1[1]), np.abs(omega.x0[1] - omega_no_overlap.x0[1])]) # distance between boundary of omega and boundary of omega_no_overlap
    l_band_z = np.max([(omega.x1[2] - omega_no_overlap.x1[2]), np.abs(omega.x0[2] - omega_no_overlap.x0[2])]) # distance between boundary of omega and boundary of omega_no_overlap
    l_band = np.array([l_band_x, l_band_y, l_band_z])
    b_pu = L_no_overlap / l_band + 1
    b_pu = b_pu
    xi_local = Function(Vs)
    xi_local.interpolate(lambda x : (- b_pu[0] / (L_no_overlap[0] + l_band[0])) * np.max(np.abs(x.T - midpoint_no_overlap), axis = 1) + b_pu[0])
    pu_dofs_1 = locate_dofs_geometrical(Vs, omega_no_overlap.inside)     # locate dofs on nonoverlapping part
    pu_dofs_0 = locate_dofs_geometrical(Vs, lambda x : np.logical_not(omega.inside(x)))     # locate dofs not on omega
    xi_local.vector.array[pu_dofs_1] = 1.0
    xi_local.vector.array[pu_dofs_0] = 0.0

    # helper.plotFunction(xi_local, submesh, "3d/xi_local_3d")

    Xi = diags(xi_local.vector.array)

    # Cut-off function eta at interior of ring:
    eta = Function(Vs)
    eta.vector.array[:] = 1
    eta.vector.array[locate_dofs_geometrical(Vs, omega_eta.inside)] = 0
    Eta = diags(eta.vector.array)

    coeff_A = Function(functionspace(submesh, ("DG", 0)))
    coeff_A.interpolate(coeff_A_function)

    coeff_A.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    # Define indicator function for oversampling ring
    chi_ring_os = Function(functionspace(submesh, ("DG", 0)))
    chi_ring = Function(functionspace(submesh, ("DG", 0)))
    chi_ring_eta = Function(functionspace(submesh, ("DG", 0)))
    if bool_ring:  
        chi_ring_os.interpolate(lambda x: ring_os.inside(x))
        chi_ring.interpolate(lambda x: ring.inside(x))
        chi_ring_eta.interpolate(lambda x: ring_eta.inside(x))
    else:
        chi_ring_os.interpolate(lambda x: 1 + 0 * x[0])

    # print("TODO: change local coeff")
    u, v = TrialFunction(Vs), TestFunction(Vs)
    a = inner(chi_ring_os * coeff_A * grad(u), grad(v)) * dx
    A_constraint = assemble_matrix(form(a)).to_scipy()
    As = assemble_matrix(form(a)).to_scipy() 
    
    # Prepare local eigenproblem 
    u, v = TrialFunction(Vs), TestFunction(Vs)
    a_plus = inner(chi_ring_os * coeff_A * grad(u), grad(v)) * dx
    A_plus = assemble_matrix(form(a_plus)).to_scipy()

    # Dofs on oversampling ring
    if bool_ring:
        all_dofs = locate_dofs_geometrical(Vs, ring_os.inside)
    else:
        all_dofs = np.arange(Vs.dofmap.index_map.size_global)

    dirichlet_dofs = np.intersect1d(all_dofs, locate_dofs_geometrical(Vs, dirichlet_boundary))
    robin_dofs = np.intersect1d(all_dofs, locate_dofs_geometrical(Vs, robin_boundary))
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
    print("Solve eigenproblem")

    if rho > 0.0:
        nloc_tmp = np.rint(len(interior_bdry_and_robin_bdry_dofs) / 10 + 1).astype(int)
        nloc_step = np.rint(len(interior_bdry_and_robin_bdry_dofs) / 10 + 1).astype(int)
        while True:  
            # Solve eigenproblem 
            vals, vecs = eigsh(MA, k=nloc_tmp, M=MB, sigma=0)
            vals = 1/vals
            vals, vecs = helper.sort_eigenpairs(vals, vecs)

            if np.any(vals < -1e-3):
                raise Exception("detected negative eigenvalues")
            idx_values = np.where(vals >= rho**2)[0]
            _, nloc_cutoff = idx_values[[0, -1]] + 1
            # print("nloc_cutoff", nloc_cutoff[i])
            if nloc_cutoff == nloc_tmp:
                nloc_tmp = nloc_tmp + nloc_step
            else:
                vals = vals[:nloc_cutoff]
                vecs = vecs[:, :nloc_cutoff]
                break
        # print("subdomain ", i, "done of total number of subdomains ", self.nDom)
    else:
        timing = 0
        timing_aharmonic_extension = 0
        if nnz_computation:
            timing_iterations = 1 # how often to solve the eigenproblem and average times
        else: 
            timing_iterations = 10
        # Solve eigenproblem 
        for i in range(timing_iterations):
            start = time.time()
            vals, vecs = eigsh(MA, k=nloc, M=MB, sigma=0)
            end = time.time()
            timing = timing + end - start
            
            # A - harmonic extension in case of ring 
            if bool_ring:
                # Extend a-harmonically to the interior
                
                # Assemble matrix over ommin
                chi_ommin = Function(functionspace(submesh, ("DG", 0)))
                chi_ommin.interpolate(lambda x: omega_min.inside(x))
                u, v = TrialFunction(Vs), TestFunction(Vs)
                a = inner(chi_ommin * coeff_A * grad(u), grad(v)) * dx
                A_ommin = assemble_matrix(form(a)).to_scipy() 
                ommin_interior_dofs = locate_dofs_geometrical(Vs, omega_min.interior)

                vecs_tmp = np.zeros((Xi.shape[0], vecs.shape[1]),dtype=np.complex128)
                vecs_tmp[non_dirichlet_dofs,:] = vecs[:len(non_dirichlet_dofs), :]

                start_harmonic_extension = time.time()
                u_g = vecs_tmp
                u_0 = spsolve(A_ommin[ommin_interior_dofs,:][:,ommin_interior_dofs], - A_ommin.dot(u_g)[ommin_interior_dofs])
                if nloc > 1:
                    vecs_tmp[ommin_interior_dofs, :] = u_g[ommin_interior_dofs, :] + u_0
                else:
                    vecs_tmp[ommin_interior_dofs,0] = u_g[ommin_interior_dofs,0] + u_0
                end_harmonic_extension = time.time()
                timing_aharmonic_extension = timing_aharmonic_extension + (end_harmonic_extension - start_harmonic_extension)
                

        print("on rings: ", bool_ring)
        timing = timing / timing_iterations
        print("Time for eigenproblem: ", timing)
        size_eigenproblem = MA.shape[0]
        timing_aharmonic_extension = timing_aharmonic_extension / timing_iterations
        print("Time for harmonic extension: ", timing_aharmonic_extension)
        print("Size of eigenproblem: ", MA.shape[0])
        vals = 1/vals
        print("vals: ", vals)

    if nnz_computation:
        # Computation of nonzero entries of lu factorization
        size_eigenproblem = MA.shape[0]
        lu = splu(MA)    # Compute LU factorization
        nnz_superlu = lu.L.count_nonzero() + lu.U.count_nonzero()    # Number of nonzeros in LU factorization
        nnz_MA = MA.count_nonzero()    # Number of nonzeros in matrix MA
        print(nnz_superlu, nnz_MA, size_eigenproblem)

        return nnz_superlu, nnz_MA, size_eigenproblem
    else:
        return timing, size_eigenproblem, timing_aharmonic_extension
import petsc4py
from petsc4py import PETSc

import msgfem_parallel as msgfem
import numpy as np
from scipy.sparse import vstack, block_diag

from ray.util.multiprocessing import Pool

import time

# Initialize PETSc
petsc4py.init()

class GfemPreconditioner:
    """
    GfemPreconditioner is a class implementing a domain decomposition preconditioner for solving large sparse linear systems,
    arising from finite element discretizations of partial differential equations. The preconditioner is based on
    the Generalized Finite Element Method (GFEM).
    Attributes:
        xR, xL, yR, yL: Domain boundary coordinates.
        ol, os: Overlap and oversampling parameters for subdomains.
        Nx, Ny: Number of subdomains in x and y directions.
        nx, ny: Number of fine elements in each direction per subdomain.
        nDom: Total number of subdomains.
        coeff: Coefficient field for the PDE.
        deg: Polynomial degree of basis functions.
        nloc: Number of local basis functions per subdomain.
        coord_global: Global coordinates of the mesh nodes.
        dirichlet_boundary: Indices of Dirichlet boundary nodes.
        robin_boundary: Indices of Robin boundary nodes.
        A_scipy: Global system matrix (scipy sparse format).
        rho: Eigenvalue cutoff parameter for coarse space construction.
        bool_ring: Boolean flag for ring method.
        perturbation_parameter: Optional perturbation parameter for eigenvalue problems.
    Methods:
        setUp(pc):
            Sets up the preconditioner by computing local eigenproblems, assembling the coarse space,
            and preparing local and coarse solvers.
        apply(pc, x, y):
            Applies the preconditioner to a given vector x, storing the result in y.
        prepare_local_solver(local_data):
            Prepares the block local solver using the computed local data.
        prepare_local_solver_serial(local_data):
            Prepares the serial local solver for each subdomain.
        prepare_coarse_solver():
            Prepares the coarse solver for the coarse space correction.
    """

    def __init__(self, pc, data, perturbation_parameter = 0.0):
        (
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
        self.coord_global,
        self.dirichlet_boundary,
        self.robin_boundary, 
        self.A_scipy,
        self.rho,
        self.bool_ring
        )       = data
        self.perturbation_parameter = perturbation_parameter

        
        # debug part 
       # db_domain = 0
        #db = msgfem.computeSubdomain((self.xR, self.xL, self.yR, self.yL, self.ol, self.os, self.Nx, 
         #                               self.Ny, self.nx, self.ny, self.nDom, self.coeff, self.deg, self.nloc,
          #                              db_domain, self.coord_global, self.dirichlet_boundary, self.robin_boundary, 
           #                             self.perturbation_parameter, self.rho, self.bool_ring))
        # debug end
    
    def setUp(self, pc):
        start = time.time()

        
        local_data = []
        for i_subdom in range(self.nDom):
            subdom_result = msgfem.computeSubdomain([
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
                                                        self.coeff,     #need to put coeff here
                                                        self.deg,
                                                        self.nloc,
                                                        i_subdom,
                                                        self.coord_global,
                                                        self.dirichlet_boundary,
                                                        self.robin_boundary, 
                                                        self.perturbation_parameter,
                                                        self.rho,
                                                        self.bool_ring])
            local_data.append(subdom_result)
                                                    
                           

        nloc_cutoff = np.zeros(self.nDom)
        for i in range(self.nDom):
            nloc_cutoff[i] = local_data[i][0].shape[1]
        nloc_cutoff = nloc_cutoff.astype(int)
        end = time.time()
        print("Time of local computations: ", end-start)

        # Build restriction and partition of unity operators
        R = [local_data[i][4] for i in range(self.nDom)]
        Xi = [local_data[i][3] for i in range(self.nDom)]

        # Check if the partition of unity property holds
        vec_tmp = np.ones(self.coord_global.shape[0])
        vec_tmp2 = np.zeros(self.coord_global.shape[0])
        for i in range(self.nDom):
            vec_tmp2 = vec_tmp2 + R[i].T.dot(Xi[i].dot(R[i].dot(vec_tmp)))
        # print("Max deviation from pou property: ", np.max(np.abs(vec_tmp2-vec_tmp)))

        if np.max(np.abs(vec_tmp2-vec_tmp)) > 1e-12:
            raise Exception("Partition of unity is not a partition of unity. Error is: " + str(np.max(np.abs(vec_tmp2-vec_tmp))))

        # Assemble coarse space
        start = time.time()
        self.AH, self.M_basis = msgfem.assembleCoarseSpace(self.A_scipy, local_data, nloc_cutoff)
        end = time.time()
        print("Assemble Coarse space:", end - start)
        print("Size of coarse space: ", self.AH.size[0])
        # print("ratio of coarse space to maximal size of local problems: ", self.AH.size[0] / local_data[self.Nx+3][3].shape[0])

        # Prepare local solver
        start = time.time()
        self.prepare_local_solver(local_data)
        end = time.time()
        print("Prepare local solvers:", end - start)

        # Prepare coarse solver
        start = time.time()
        self.prepare_coarse_solver()
        end = time.time()
        print("Prepare coarse solvers:", end - start)


    def apply(self, pc, x, y):
        """
        Applies the MS-GFEM two-level preconditioner to the input vector `x` and stores the result in `y`.
        Args:
            pc: The PETSc preconditioner context (unused in this implementation).
            x: The PETSc vector representing the right-hand side.
            y: The PETSc vector where the result is stored.
        """
        # Get the right-hand side vector
        rhs = x.array
        # Instead of solving each local problem separately, we construct a block 
        # diagonal system and solve it in one go.
        rhs_block = self.R_block.dot(rhs)
        # Solve the local problems
        u_tmp = msgfem.petsc_solve(self.ksp_block, self.As_block_petsc, rhs_block[self.interior_dofs_block])

        # From the block solutions, construct the global particular function
        u_block = np.zeros(self.Xi_block.shape[0], dtype=np.float64)
        u_block[self.interior_dofs_block] = u_tmp
        u_block = self.Xi_block.dot(u_block)
        u_par_glob = self.R_block.T.dot(u_block)

        # Construct coarse right hand side and solve coarse problem
        rhsH = self.M_basis.T.dot((rhs - self.A_scipy.dot(u_par_glob)))
        solH = msgfem.petsc_solve(self.ksp_coarse, self.AH, rhsH)

        # Write the result into the output vector
        y.array = u_par_glob + self.M_basis.dot(solH)

    def prepare_local_solver(self, local_data):
        """
        Prepares the block local solver by assembling block diagonal matrices for all subdomains,
        setting up the restriction and partition of unity operators, and configuring the PETSc KSP solver
        for the block system. Computes the indices of interior degrees of freedom for the block system.
        """

        # Create block version of restriction and partition of unity operators
        self.R_block = vstack([local_data[i][4] for i in range(self.nDom)])
        self.Xi_block = block_diag([local_data[i][3] for i in range(self.nDom)])
        offsets = np.array([local_data[i][3].shape[0] for i in range(self.nDom)])
        offsets = np.cumsum(offsets)
        offsets[-1] = 0
        offsets = np.roll(offsets, 1)
        id_dofs = 6     # index of interior dofs in local data

        self.interior_dofs_block = np.concatenate([local_data[i][id_dofs] + offsets[i] for i in range(self.nDom)])
        
        opts = PETSc.Options()
        opts.setValue("-pc_type", "lu")
        opts.setValue("-pc_factor_mat_solver_type", "mumps")

        As_block = block_diag([local_data[i][2][local_data[i][id_dofs],:][:,local_data[i][id_dofs]] for i in range(self.nDom)]).tocsr()
        self.As_block_petsc = msgfem.scipy_to_petsc_matrix(As_block)

        self.ksp_block = PETSc.KSP().create()
        self.ksp_block.setOperators(self.As_block_petsc)
        self.ksp_block.setFromOptions()

        p = np.ones(As_block.shape[0])
        u = msgfem.petsc_solve(self.ksp_block, self.As_block_petsc, p)

    def prepare_local_solver_serial(self, local_data):
        self.As_inv = msgfem.factorizeMatrices([local_data[i][2] for i in range(self.nDom)], self.nDom)
        self.Rs = [local_data[i][4] for i in range(self.nDom)]
        self.Xis = [local_data[i][3] for i in range(self.nDom)]

    def prepare_coarse_solver(self):
        """
        Prepares and configures the coarse-level linear solver for the preconditioner.
        This method sets up a PETSc KSP (Krylov Subspace Solver) object for solving the coarse system,
        using LU factorization with the MUMPS solver.
        Returns:
            None
        """

        start = time.time()
        opts = PETSc.Options()
        opts.setValue("-pc_type", "lu")
        opts.setValue("-pc_factor_mat_solver_type", "mumps")

        self.ksp_coarse= PETSc.KSP().create()
        self.ksp_coarse.setOperators(self.AH)
        self.ksp_coarse.setFromOptions()

        p = np.ones(self.AH.size[0])
        u = msgfem.petsc_solve(self.ksp_coarse, self.AH, p)
        end = time.time()
        print("block factorizing time: ", end-start)

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
import ufl
from dolfinx.fem import Function, functionspace, locate_dofs_geometrical, dirichletbc
from dolfinx.mesh import create_rectangle, CellType

import msgfem_parallel as msgfem
import helper
import preconditioner as pre
import setup


def run_msgfem(deg, Ny, ny, ol, os, nloc, rho, problem_label, bool_ring, parameters=None, contrast = 1):
    """
    Runs the Multiscale Generalized Finite Element Method (MS-GFEM) for solving elliptic PDEs on a rectangular mesh.
    This function sets up the finite element mesh, defines the problem parameters, assembles the system matrices,
    applies a custom MS-GFEM preconditioner, and solves the linear system using PETSc's iterative solvers.
    It computes both the reference fine FEM solution and the MS-GFEM solution, and evaluates the relative error
    between them. Solver statistics and coarse space size are also reported.
    Parameters
    ----------
    deg : int
        Polynomial degree of the finite element basis functions.
    Ny : int
        Number of coarse mesh elements in the y-direction.
    ny : int
        Number of fine mesh elements in the y-direction.
    ol : int
        Overlap parameter for local subdomains in the MS-GFEM.
    os : int
        Oversampling parameter for local subdomains in the MS-GFEM.
    nloc : int
        Number of local eigenvectors used in MS-GFEM per subdomain. Use same number of eigenvector for all subdomains.
        If rho is not zero, the tolerance rho is used to determine the number of local eigenvectors.
    rho : float
        Tolerance for the local eigenvalue problem in MS-GFEM. Not implemented yet!
    problem_label : str
        Label specifying the problem setup (e.g., "source_dirichlet", "iid", "channel", "skyscraper").
    bool_ring : bool
        Flag indicating whether to use the ring method or the original MS-GFEM.
    parameters: List
        Free parameters that are passed to the coefficient function.  
    contrast : float, optional
        Contrast parameter for the coefficient field (default is 1).
    Returns
    -------
    iterations : int
        Number of iterations performed by the iterative solver.
    gfem_error : float
        Relative energy error of the MS-GFEM solution compared to the fine FEM solution.
    coarse_space_size : int
        Size of the coarse space used in the MS-GFEM preconditioner.
    Raises
    ------
    Exception
        If the fine mesh is not a submesh of the coarse mesh.
    Notes
    -----
    - Problem setups are loaded from the `setup` module.
    - The function can optionally plot coefficient fields and solutions if plotting lines are uncommented.
    """

    if rho != 0.0:
        raise NotImplementedError("The tolerance rho is not implemented yet!")

    # Endpoints of mesh
    xL, yL, xR, yR = 0, 0, 1, 1
    xy_scaling = np.rint((xR - xL) / (yR - yL))

    # Scaling of the mesh (Would only be needed if we want to change the aspect ratio of the mesh)
    Nx = np.rint(xy_scaling * Ny).astype(int)
    nDom = Nx * Ny

    # Number of elements in each direction of the mesh
    nx = ny * xy_scaling
    nx = nx.astype(int)

    print("Nx", Nx, "Ny:", Ny, "nx:", nx, "ny:", ny)

    if np.mod(nx, Nx) != 0 or np.mod(ny, Ny) != 0:
        raise Exception("Finemesh has to be a submesh of coarse mesh")

    # Create mesh consisting of quadrilaterals
    msh = create_rectangle(
        comm=MPI.COMM_WORLD,
        points=((xL, yL), (xR, yR)),
        n=(nx, ny),
        cell_type=CellType.quadrilateral,
    )

    # Test and trial function space
    V = functionspace(msh, ("Lagrange", deg))

    # Load different problem setups

    if problem_label == "source_dirichlet":
        dirichlet_boundary, robin_boundary, u_D, f, coeff_A_function = setup.getSetupSourceDirichlet(xL, yL, xR, yR, V, msh)
    elif problem_label == "iid":
        dirichlet_boundary, robin_boundary, u_D, f, coeff_A_function = setup.getSetupIID(xL, yL, xR, yR, V, contrast)
    elif problem_label == "channel":
        dirichlet_boundary, robin_boundary, u_D, f, coeff_A_function = setup.getSetupChannel(xL, yL, xR, yR, V, ny, Ny, contrast)
    elif problem_label == "square_middle":
        dirichlet_boundary, robin_boundary, u_D, f, coeff_A_function = setup.getSetupSquareMiddle(xL, yL, xR, yR, V, contrast)
    elif problem_label == "skyscraper":
        dirichlet_boundary, robin_boundary, u_D, f, coeff_A_function = setup.getSetupSkyscraper(xL, yL, xR, yR, V, contrast)
    elif problem_label == "Dirichlet_FNO":
        dirichlet_boundary, robin_boundary, u_D, f, coeff_A_function = setup.Dirichlet_FNO(xL, yL, xR, yR, V, msh, parameters)

    
    # Create FE function for coefficient A, which is a DG0 function
    coeff_A = Function(functionspace(msh, ("DG", 0))) 
    coeff_A.interpolate(coeff_A_function)

    

    # helper.plotFunction(coeff_A, msh, "coeff_A_" + problem_label)    # Uncomment to plot the coefficient A
    
    # Locate global dirichlet dofs and global robin dofs
    dirichlet_dofs_global = locate_dofs_geometrical(V, dirichlet_boundary)

    # Create Dirichlet boundary condition
    bc = dirichletbc(u_D, dirichlet_dofs_global)

    # Define the bilinear form a and linear form L
    a, L = helper.getEllipticProblem(coeff_A, f, V)

    # Compute reference fine FEM solution with sparse direct solver
    uh = msgfem.fem_solve(V, a, L , bc)

    # helper.plotFunction(uh, msh, "uh")    # uncomment to plot the fine FEM solution

    # Obtain coordinates of the dofs on the mesh
    coord_global = V.tabulate_dof_coordinates()

    # Create a KSP solver context
    ksp = PETSc.KSP().create()

    # Assemble system from bilinear form a and linear form L for the preconditioner
    A, b = helper.assemble_matrix_and_vector(a, L, bc)

    # Assemble system from bilinear form a and linear form L for the multiscale method
    A_tmp, b_tmp, A_scipy = helper.assemble_matrix_and_vector_noniterative(a, L)

    # Set operator
    ksp.setOperators(A)

    # Get the preconditioner context
    pc = ksp.getPC()

    # Set the preconditioner type to PYTHON
    pc.setType(PETSc.PC.Type.PYTHON)

    # Data needed for MS-GFEM preconditioner
    data =(
            xR,
            xL,
            yR,
            yL,
            ol,
            os,
            Nx,
            Ny,
            nx,
            ny,
            nDom,
            coeff_A_function,     
            deg,
            nloc,
            coord_global,
            dirichlet_boundary,
            robin_boundary, 
            A_scipy,
            rho,
            bool_ring
        )

    # Attach the custom preconditioner
    gfem_pre = pre.GfemPreconditioner(pc, data)
    pc.setPythonContext(gfem_pre)

    # Set up the KSP solver
    ksp.setType(PETSc.KSP.Type.RICHARDSON)  # Can also change to GMRES      
    ksp.setTolerances(rtol=1e-8, max_it = 1000)

    # ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED) # uncomment to switch to unpreconditioned norm (then right-preconditioning is used!)

    # Set the monitor function to print the residual norm at each iteration
    ksp.setMonitor(helper.monitor)

    # Compute MS-GFEM multiscale solution error
    uG = Function(V)
    uG_vec = PETSc.Vec().createSeq(b.array.size)
    pc.apply(b_tmp, uG_vec)
    uG.vector.array[:] = uG_vec.array

    # Difference between the fine FEM solution and the MS-GFEM solution
    udiff = Function(V)
    udiff.vector.array[:] = uG.vector.array - uh.vector.array

    # Compute energy error of the MS-GFEM solution to the fine FEM solution
    gfem_error = helper.compute_errors(uG, uh, msh, coeff_A)
    print("Relative energy error of GFEM solution:", gfem_error)

    # Solve the linear system Ax = b with the MS-GFEM preconditioner
    x = PETSc.Vec().createSeq(b.array.size)
    ksp.solve(b, x)

    # Write solution to a FE function
    u_iterative = Function(V)
    u_iterative.vector.array[:] = x.array

    print("Relative energy error of iterative solution:", helper.compute_errors(u_iterative, uh, msh, coeff_A))

    # Get KSP solver information
    converged_reason = ksp.getConvergedReason()
    iterations = ksp.getIterationNumber()
    final_residual_norm = ksp.getResidualNorm()
    coarse_space_size = pc.getPythonContext().AH.size[0]

    # Print the solver information
    print("Converged Reason:", converged_reason)
    print("Number of Iterations:", iterations)
    print("Final Residual Norm:", final_residual_norm)

    return iterations, gfem_error, coarse_space_size

def run_fno_data_generation(deg, Ny, ny, ol, os, nloc, rho, bool_ring, parameters=None, domain_idx=5,):
    """
    Runs the Multiscale Generalized Finite Element Method (MS-GFEM) for solving elliptic PDEs on a rectangular mesh.
    This function sets up the finite element mesh, defines the problem parameters, and solves the local problems on one specified subdomain
    Parameters
    ----------
    deg : int
        Polynomial degree of the finite element basis functions.
    Ny : int
        Number of coarse mesh elements in the y-direction.
    ny : int
        Number of fine mesh elements in the y-direction.
    ol : int
        Overlap parameter for local subdomains in the MS-GFEM.
    os : int
        Oversampling parameter for local subdomains in the MS-GFEM.
    nloc : int
        Number of local eigenvectors used in MS-GFEM per subdomain. Use same number of eigenvector for all subdomains.
        If rho is not zero, the tolerance rho is used to determine the number of local eigenvectors.
    rho : float
        Tolerance for the local eigenvalue problem in MS-GFEM. Not implemented yet!
    bool_ring : bool
        Flag indicating whether to use the ring method or the original MS-GFEM.
    domain_idx : int
        Index for the subdomain where FNO data is computed, default 5, where an inner domain is taken by assuming 16 square subdomains.
    parameters: List
        Free parameters that are passed to the coefficient function.  

    Returns
    -------
    coeff_A_FNO : ndarray
        Coefficient function of PDE discretized on the FEM grid of shape (nx, ny, out_dim).
    eig_func_2d : ndarray
        Eigenfunctions of the local eigenvalue problem on the specified subdomain, shape (Nx_sub, Ny_sub, nloc).
    vals : ndarray
        Eigenvalues of the local eigenvalue problem on the specified subdomain, shape (nloc,).
    """

    # Endpoints of mesh
    xL, yL, xR, yR = 0, 0, 1, 1
    xy_scaling = np.rint((xR - xL) / (yR - yL))

    # Scaling of the mesh (Would only be needed if we want to change the aspect ratio of the mesh)
    Nx = np.rint(xy_scaling * Ny).astype(int)
    nDom = Nx * Ny

    # Number of elements in each direction of the mesh
    nx = ny * xy_scaling
    nx = nx.astype(int)

    print("Nx", Nx, "Ny:", Ny, "nx:", nx, "ny:", ny)

    if np.mod(nx, Nx) != 0 or np.mod(ny, Ny) != 0:
        raise Exception("Finemesh has to be a submesh of coarse mesh")

    # Create mesh consisting of quadrilaterals
    msh = create_rectangle(
        comm=MPI.COMM_WORLD,
        points=((xL, yL), (xR, yR)),
        n=(nx, ny),
        cell_type=CellType.quadrilateral,
    )

    # Test and trial function space
    V = functionspace(msh, ("Lagrange", deg))

    # Load different problem setups
    dirichlet_boundary, robin_boundary, u_D, f, coeff_A_function = setup.Dirichlet_FNO(xL, yL, xR, yR, V, msh, parameters)


    # Obtain coordinates of the dofs on the mesh
    coord_global = V.tabulate_dof_coordinates()

    perturbation_parameter = 0.0
    coeff_A_FNO, eig_func_2d, vals = msgfem.computeSubdomain((xR, xL, yR, yL, ol, os, Nx, 
                                                            Ny, nx, ny, nDom, coeff_A_function, deg, nloc,
                                                            domain_idx, coord_global, dirichlet_boundary, robin_boundary, 
                                                            perturbation_parameter, rho, bool_ring))
    
    return coeff_A_FNO, eig_func_2d, vals

    
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.fem import Function, assemble_scalar, form, functionspace, Constant, locate_dofs_geometrical, dirichletbc, apply_lifting, set_bc
from dolfinx.fem.petsc import LinearProblem, assemble_matrix, assemble_vector
from dolfinx.io import XDMFFile
from dolfinx.mesh import create_rectangle, CellType, meshtags, locate_entities
from ufl import dx, grad, inner, TrialFunction, TestFunction, ds, FacetNormal, Measure
from ray.util.multiprocessing import Pool
import ufl
import os

import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import csr_array, diags, csr_matrix, bmat, hstack
from scipy.sparse.linalg import eigsh, splu
from scipy.linalg import orth

import matplotlib.pyplot as plt
import pandas as pd

import subprocess

def get_commit_hash():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")

def assemble_matrix_and_vector(a, L, bc):
    """
    Assembles the system matrix and right-hand side vector for a finite element problem.
    Parameters
    ----------
    a : ufl.Form
        The bilinear form representing the system matrix.
    L : ufl.Form
        The linear form representing the right-hand side vector.
    bc : DirichletBC or list of DirichletBC
        The boundary condition(s) to be applied to the system.
    Returns
    -------
    A : PETScMatrix
        The assembled system matrix with boundary conditions applied.
    b : PETScVector
        The assembled right-hand side vector with boundary conditions applied.
    """

    A = assemble_matrix(form(a), bcs=[bc])
    A.assemble()
    b = assemble_vector(form(L))
    apply_lifting(b, [form(a)], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc])
    b.assemble()
    return A, b

def assemble_matrix_and_vector_noniterative(a, L):
    """
    Assembles the system matrix and right-hand side vector for a non-iterative MS-GFEM problem.
    Parameters
    ----------
    a : object
        The bilinear form or operator representing the system matrix to be assembled.
    L : object
        The linear form or operator representing the right-hand side vector to be assembled.
    Returns
    -------
    A_tmp : object
        The assembled system matrix in the original format (e.g., PETSc or similar).
    b_tmp : object
        The assembled right-hand side vector in the original format.
    A_scipy : scipy.sparse.csr_matrix
        The assembled system matrix converted to a SciPy CSR sparse matrix format.
    Notes
    -----
    This function is intended for non-iterative Multi-Scale Generalized Finite Element Method (MS-GFEM)
    computations. It assembles the matrix and vector using provided forms, and also returns a SciPy-compatible
    sparse matrix for further numerical operations.
    """

    A_tmp = assemble_matrix(form(a))     
    A_tmp.assemble()
    b_tmp = assemble_vector(form(L))
    b_tmp.assemble()
    ai, aj, av = A_tmp.getValuesCSR()
    A_scipy = csr_matrix((av, aj, ai))
    return A_tmp, b_tmp, A_scipy


# Define a monitor function to print resaidual norm at each iteration
def monitor(ksp, its, rnorm):
    print(f"Iteration {its}, Residual norm {rnorm}")

def compute_errors(u, uexact, msh, coeff_A):
    """
    Computes the relative energy error between a numerical solution and the exact solution.
    Parameters
    ----------
    u : Function or array-like
        The numerical solution to compare.
    uexact : Function or array-like
        The exact solution for reference.
    msh : Mesh object
        The mesh over which the solutions are defined. 
    coeff_A : scalar valued coefficient function
        The coefficient used calculation.
    Returns
    -------
    float
        The relative energy error, defined as sqrt(H1_diff) / sqrt(H1_exact), where H1_diff is the seminorm of the difference and H1_exact is the seminorm of the exact solution.
    """
    
    diff = u - uexact
    H1_diff = msh.comm.allreduce(assemble_scalar(form(coeff_A * inner(grad(diff), grad(diff)) * dx)), op=MPI.SUM)
    H1_exact = msh.comm.allreduce(
        assemble_scalar(form(coeff_A * inner(grad(uexact), grad(uexact)) * dx )), op=MPI.SUM
    )
    return abs(np.sqrt(H1_diff) / np.sqrt(H1_exact))


# def getHelmholtzProblem(k, coeff_V, u_R, ds, f, imag_unit, msh, V):
#     # Define variational problem
#     u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
#     n = FacetNormal(msh)
#     if k > 1e-9:
#         a = inner(grad(u), grad(v)) * dx - k**2 * inner(coeff_V * coeff_V * u, v) * dx - imag_unit * k * inner(coeff_V * u,v) * ds(2)
#         L = inner(f, v) * dx + inner(inner(grad(u_R),n) - imag_unit * k * coeff_V * u_R,v) * ds(2)
#     else:
#         a = inner(grad(u), grad(v)) * dx 
#         L = inner(f, v) * dx 
#     return a, L

def getEllipticProblem(coeff_A, f, V):
    """
    Constructs the variational forms for a general elliptic PDE problem.
    Given a coefficient `coeff_A`, a source term `f`, and a finite element function space `V`,
    this function returns the bilinear form `a` and linear form `L`..
    The bilinear form `a` corresponds to the weak formulation of the elliptic operator:
        a(u, v) = ∫ (coeff_A * ∇u) · ∇v dx
    The linear form `L` corresponds to the source term:
        L(v) = ∫ f · v dx
    Parameters
    ----------
    coeff_A : ufl.Expr or compatible type
        Coefficient for the elliptic operator.
    f : ufl.Expr or compatible type
        Source term in the PDE.
    V : ufl.FunctionSpace or compatible type
        Function space for trial and test functions.
    Returns
    -------
    a : ufl.Form
        Bilinear form representing the left-hand side of the weak problem.
    L : ufl.Form
        Linear form representing the right-hand side (source term) of the weak problem.
    """

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = inner(coeff_A * grad(u), grad(v)) * dx 
    L = inner(f, v) * dx 
    return a, L

def sort_eigenpairs(vals, vecs):
    """
    Sorts eigenvalues and corresponding eigenvectors in descending order.
    Parameters
    ----------
    vals : numpy.ndarray
        Array of eigenvalues.
    vecs : numpy.ndarray
        2D array where each column is an eigenvector corresponding to the eigenvalues in `vals`.
    Returns
    -------
    sorted_vals : numpy.ndarray
        Eigenvalues sorted in descending order.
    sorted_vecs : numpy.ndarray
        Eigenvectors reordered to match the sorted eigenvalues.
    """

    # Sort by eigenvalues
    # get the indices that would sort the array in ascending order
    ascending_indices = vals.argsort()  
    
    # reverse the ascending indices to get descending indices
    descending_indices = ascending_indices[::-1]  

    vecs = vecs[:,descending_indices]
    vals = vals[descending_indices]

    return vals, vecs

def plotFunction(u, msh, filename):
    with XDMFFile(msh.comm, "out_functions/" + filename + ".xdmf", "w") as file:
        file.write_mesh(msh)
        file.write_function(u)

def createLatexFile(df, caption, filename):
    latex_table = df.to_latex()
    # Open the file in write mode and write the text
    if not os.path.exists("tables/"):
        os.makedirs("tables/")
    with open("tables/" + filename, "w") as file:
        # file.write(r"\begin{table}")
        file.write(latex_table)
        # file.write(r"\caption{" + r"}")
        # file.write("\n")
        # file.write(r"\label{tab:"+ filename + "_" + caption + "}")
        # file.write("\n")
        # file.write(r"\end{table}")

def createTables(filename):
    print("**filename: "+ filename+ "**")
    data = np.load("data_rings/"+ filename, allow_pickle = False)
    filename = filename[:-4]

    print("ny = ", data["ny"])
    print("Ny = ", data["Ny"])
    print("ol = ", data["ol"])
    print("os = ", data["os"])

    contrasts = data["contrasts"].astype(int)
    # contrasts = np.array([1, 10, 100, 1000, 10000])
    nlocs = data["nlocs"]
    iteration_numbers = data["iteration_numbers"].astype(int).T
    gfem_errors = data["gfem_errors"]
    iteration_numbers_rings = data["iteration_numbers_ring"].astype(int).T
    gfem_errors_rings = data["gfem_errors_ring"]

    cutoff = np.min([iteration_numbers.shape[1], 12])
    nlocs = nlocs[:cutoff]
    iteration_numbers = iteration_numbers[:, :cutoff]
    iteration_numbers_rings = iteration_numbers_rings[:, :cutoff]
    gfem_errors = gfem_errors[:cutoff,:]
    gfem_errors_rings = gfem_errors_rings[:cutoff, :]

    df = pd.DataFrame(data=iteration_numbers,    # values
                index=contrasts,    # 1st column as index
                columns=nlocs)  # 1st row as the column names
    
    # col_name = r"contrast\textbackslash $n_{loc}$"
    col_name = r"$\alpha_{\mathrm{max}} \setminus n_{loc}$"

    df.columns.name = col_name

    df_rings = pd.DataFrame(data=iteration_numbers_rings,    # values
                index=contrasts,    # 1st column as index
                columns=nlocs)  # 1st row as the column names

    df_rings.columns.name = col_name

    df_gfem_errors = pd.DataFrame(data=gfem_errors.T,    # values
                index=contrasts,    # 1st column as index
                columns=nlocs)  # 1st row as the column names

    df_gfem_errors.columns.name = col_name

    df_gfem_errors_rings = pd.DataFrame(data=gfem_errors_rings.T,    # values
                index=contrasts,    # 1st column as index 
                columns=nlocs)  # 1st row as the column names

    df_gfem_errors_rings.columns.name = col_name

    # print("iteration numbers")
    # print(df)

    # print("\n")
    # print("iteration numbers ring")
    # print(df_rings)

    # print("\n")
    # print("gfem errors:")
    # print(df_gfem_errors)

    # print("\n")
    # print("gfem errors rings:")
    # print(df_gfem_errors_rings)

    # Get the colormap
    # jet = cm.get_cmap('jet')

    createLatexFile(df, "iteration_numbers", filename + "iteration_numbers.tex")
    createLatexFile(df_rings, "iteration_numbers_rings", filename + "iteration_numbers_rings.tex")
    createLatexFile(df_gfem_errors, "gfem_errors", filename + "gfem_errors.tex")
    createLatexFile(df_gfem_errors_rings, "gfem_errors_rings", filename + "gfem_errors_rings.tex")
    
    # plot_coefficient(data)

    # for contrast_id in range(len(contrasts)):
    #     plt.semilogy(nlocs, gfem_errors[:, contrast_id], label = r"$\omega^{\ast}$", marker = "o", color = jet(contrast_id/len(contrasts)) )
    #     plt.semilogy(nlocs, gfem_errors_rings[:, contrast_id], label = r"$R^{\ast}$", marker = "x", color = jet(contrast_id/len(contrasts)) )
    #     plt.xlabel("$n_{loc}$")
    #     plt.ylabel("$error$")
    # plt.legend()
    # # plt.show()
    # plt.savefig("../../manuscript/plots/"+ filename + ".png", dpi = 300)
    # plt.show()
    # Assuming nlocs, gfem_errors, gfem_errors_rings, contrasts, and filename are defined
    # jet = cm.get_cmap('jet')
    colors = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30", "#4DBEEE", "#A2142F", "#77AC30", "#4DBEEE", "#A2142F"]

    # Create a figure and axis with appropriate size for readability
    plt.figure(figsize=(8, 6))

    # Loop through contrasts and plot the data with enhancements
    for contrast_id in range(len(contrasts)):
        # color = jet(contrast_id / len(contrasts))  # Get a color from the colormap
        plt.semilogy(
            nlocs, 
            gfem_errors[:, contrast_id], 
            label=fr"$\omega^{{\ast}}$", 
            marker="o", 
            color=colors[contrast_id], 
            linestyle='-',
            linewidth=1
        )
        plt.semilogy(
            nlocs, 
            gfem_errors_rings[:, contrast_id], 
            label=fr"$R^{{\ast}}$", 
            marker="x", 
            color=colors[contrast_id], 
            linestyle='--',
            linewidth=1
        )

    # Add labels and title
    plt.xlabel(r"$n$", fontsize=16)
    plt.ylabel(r"$\text{error}$", fontsize=16)

    # Customize tick font size
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Add grid for better readability
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Add legend outside the plot for better clarity
    plt.legend(loc='lower left', fontsize=14)

    # Adjust layout for tight fit
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # Save figure with high resolution for publication
    if not os.path.exists("plots/"):
        os.makedirs("plots/")
    output_path = f"plots/{filename}.pdf"
    plt.savefig(output_path, dpi=600, bbox_inches="tight")

    # Display the plot
    plt.show()


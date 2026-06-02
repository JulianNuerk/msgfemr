"""
Karhunen-Loève Expansion for Stochastic Permeability Field Generation.

References:
    Wong (1971); Aarnes & Efendiev (2008); Vasilyeva et al. (2021)

Steps:
    1. Define exponential covariance function R(x,y)
    2. Discretize and solve the Fredholm integral eigenvalue problem
    3. Construct the random field via KL expansion
    4. Generate permeability fields kappa(x, omega) = exp(a * Y_L(x, omega))
"""

import numpy as np
from scipy.linalg import eigh
from scipy.interpolate import RegularGridInterpolator


def covariance_kernel_2d(x1, x2, lx=0.02, ly=0.6, sigma2=2.0):
    """
    Exponential covariance kernel for 2D case.

    R(x, y) = sigma^2 * exp(-sqrt(Delta^2))

    with Delta^2 = |x1_1 - x2_1|^2 / lx^2 + |x1_2 - x2_2|^2 / ly^2

    Parameters
    ----------
    x1 : array of shape (N, 2)
        First set of points.
    x2 : array of shape (M, 2)
        Second set of points.
    lx : float
        Correlation length in x-direction.
    ly : float
        Correlation length in y-direction.
    sigma2 : float
        Variance of the field.

    Returns
    -------
    R : array of shape (N, M)
        Covariance matrix between points x1 and x2.
    """
    # Compute pairwise differences
    dx = x1[:, 0:1] - x2[:, 0:1].T  # (N, M)
    dy = x1[:, 1:2] - x2[:, 1:2].T  # (N, M)

    Delta2 = (dx ** 2) / lx ** 2 + (dy ** 2) / ly ** 2
    R = sigma2 * np.exp(-np.sqrt(Delta2))
    return R


def covariance_kernel_3d(x1, x2, lx=0.02, ly=0.6, lz=0.2, sigma2=2.0):
    """
    Exponential covariance kernel for 3D case.

    R(x, y) = sigma^2 * exp(-sqrt(Delta^2))

    with Delta^2 = |x1-x2|^2/lx^2 + |y1-y2|^2/ly^2 + |z1-z2|^2/lz^2

    Parameters
    ----------
    x1 : array of shape (N, 3)
        First set of points.
    x2 : array of shape (M, 3)
        Second set of points.
    lx : float
        Correlation length in x-direction.
    ly : float
        Correlation length in y-direction.
    lz : float
        Correlation length in z-direction.
    sigma2 : float
        Variance of the field.

    Returns
    -------
    R : array of shape (N, M)
        Covariance matrix between points x1 and x2.
    """
    dx = x1[:, 0:1] - x2[:, 0:1].T
    dy = x1[:, 1:2] - x2[:, 1:2].T
    dz = x1[:, 2:3] - x2[:, 2:3].T

    Delta2 = (dx ** 2) / lx ** 2 + (dy ** 2) / ly ** 2 + (dz ** 2) / lz ** 2
    R = sigma2 * np.exp(-np.sqrt(Delta2))
    return R


def discretize_covariance_2d(Nx, Ny, domain=(0, 1, 0, 1),
                              lx=0.02, ly=0.6, sigma2=2.0):
    """
    Discretize the covariance operator on a 2D rectangular grid using
    the midpoint quadrature rule.

    The Fredholm integral equation:
        int_Omega R(x, y) phi_k(y) dy = lambda_k phi_k(x)

    is discretized as:
        C @ phi = lambda * phi

    where C[i,j] = R(x_i, x_j) * w_j, and w_j is the quadrature weight
    (area of each cell).

    Parameters
    ----------
    Nx : int
        Number of grid points in x-direction.
    Ny : int
        Number of grid points in y-direction.
    domain : tuple (x_min, x_max, y_min, y_max)
        Rectangular domain bounds.
    lx : float
        Correlation length in x.
    ly : float
        Correlation length in y.
    sigma2 : float
        Variance.

    Returns
    -------
    C_weighted : array of shape (Nx*Ny, Nx*Ny)
        Symmetrized weighted covariance matrix for the eigenvalue problem.
    quad_weights : array of shape (Nx*Ny,)
        Quadrature weights for each grid point.
    grid_points : array of shape (Nx*Ny, 2)
        Grid point coordinates.
    grid_shape : tuple (Nx, Ny)
        Shape of the 2D grid.
    """
    x_min, x_max, y_min, y_max = domain

    # Midpoint grid
    hx = (x_max - x_min) / Nx
    hy = (y_max - y_min) / Ny
    x_centers = np.linspace(x_min + hx / 2, x_max - hx / 2, Nx)
    y_centers = np.linspace(y_min + hy / 2, y_max - hy / 2, Ny)

    # Create 2D grid of points (row-major order)
    xx, yy = np.meshgrid(x_centers, y_centers, indexing='ij')
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])  # (Nx*Ny, 2)

    # Quadrature weight (area per cell)
    w = hx * hy
    quad_weights = np.full(Nx * Ny, w)

    # Assemble covariance matrix
    R = covariance_kernel_2d(grid_points, grid_points, lx=lx, ly=ly, sigma2=sigma2)

    # Weighted covariance matrix: C[i,j] = R(x_i, x_j) * w_j
    # To solve generalized eigenvalue problem as standard symmetric problem,
    # symmetrize: C_sym = W^{1/2} R W^{1/2} where W = diag(quad_weights)
    # Then eigenvalues are the same and eigenvectors of original are
    # phi = W^{-1/2} psi (where psi are eigenvectors of C_sym)
    sqrt_w = np.sqrt(quad_weights)
    C_weighted = R * np.outer(sqrt_w, sqrt_w)

    return C_weighted, quad_weights, grid_points, (Nx, Ny)


def discretize_covariance_3d(Nx, Ny, Nz, domain=(0, 1, 0, 1, 0, 1),
                              lx=0.02, ly=0.6, lz=0.2, sigma2=2.0):
    """
    Discretize the covariance operator on a 3D rectangular grid using
    the midpoint quadrature rule.

    Parameters
    ----------
    Nx, Ny, Nz : int
        Number of grid points in each direction.
    domain : tuple (x_min, x_max, y_min, y_max, z_min, z_max)
        Domain bounds.
    lx, ly, lz : float
        Correlation lengths.
    sigma2 : float
        Variance.

    Returns
    -------
    C_weighted : array of shape (N_tot, N_tot)
        Symmetrized weighted covariance matrix.
    quad_weights : array of shape (N_tot,)
        Quadrature weights.
    grid_points : array of shape (N_tot, 3)
        Grid point coordinates.
    grid_shape : tuple (Nx, Ny, Nz)
        Shape of the 3D grid.
    """
    x_min, x_max, y_min, y_max, z_min, z_max = domain

    hx = (x_max - x_min) / Nx
    hy = (y_max - y_min) / Ny
    hz = (z_max - z_min) / Nz

    x_centers = np.linspace(x_min + hx / 2, x_max - hx / 2, Nx)
    y_centers = np.linspace(y_min + hy / 2, y_max - hy / 2, Ny)
    z_centers = np.linspace(z_min + hz / 2, z_max - hz / 2, Nz)

    xx, yy, zz = np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')
    grid_points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    w = hx * hy * hz
    N_tot = Nx * Ny * Nz
    quad_weights = np.full(N_tot, w)

    R = covariance_kernel_3d(grid_points, grid_points, lx=lx, ly=ly, lz=lz, sigma2=sigma2)

    sqrt_w = np.sqrt(quad_weights)
    C_weighted = R * np.outer(sqrt_w, sqrt_w)

    return C_weighted, quad_weights, grid_points, (Nx, Ny, Nz)


def solve_eigenvalue_problem(C_weighted, quad_weights, num_modes=None):
    """
    Solve the discretized Fredholm integral eigenvalue problem.

    The symmetrized problem is:
        C_sym @ psi_k = lambda_k * psi_k

    The eigenfunctions on the grid are recovered as:
        phi_k = W^{-1/2} * psi_k

    Parameters
    ----------
    C_weighted : array of shape (N, N)
        Symmetrized weighted covariance matrix (W^{1/2} R W^{1/2}).
    quad_weights : array of shape (N,)
        Quadrature weights.
    num_modes : int or None
        Number of leading eigenmodes to compute. If None, compute all.

    Returns
    -------
    eigenvalues : array of shape (L,)
        Eigenvalues in descending order.
    eigenfunctions : array of shape (N, L)
        Eigenfunctions evaluated at grid points (columns), normalized
        such that int phi_k^2 dx = 1.
    """
    N = C_weighted.shape[0]
    if num_modes is None:
        num_modes = N

    # Solve symmetric eigenvalue problem (returns ascending order)
    # Use subset_by_index to get only the largest eigenvalues
    idx_start = N - num_modes
    eigenvalues, psi = eigh(C_weighted, subset_by_index=[idx_start, N - 1])

    # Reverse to get descending order
    eigenvalues = eigenvalues[::-1].copy()
    psi = psi[:, ::-1].copy()

    # Recover eigenfunctions: phi_k = W^{-1/2} * psi_k
    inv_sqrt_w = 1.0 / np.sqrt(quad_weights)
    eigenfunctions = psi * inv_sqrt_w[:, np.newaxis]

    # Eigenfunctions are already normalized w.r.t. L2 inner product with
    # quadrature weights since psi are orthonormal and phi_k = W^{-1/2} psi_k
    # implies int phi_k * phi_j dx ≈ sum_i phi_k(x_i) phi_j(x_i) w_i
    #   = sum_i (psi_k_i / sqrt(w_i)) (psi_j_i / sqrt(w_i)) w_i
    #   = sum_i psi_k_i psi_j_i = delta_{kj}

    return eigenvalues, eigenfunctions


def kl_expansion(x, theta, eigenvalues, eigenfunctions, a=2.6):
    """
    Evaluate a KL expansion at user-provided coordinates.

    Y_L(x, omega) = sum_{k=1}^{L} sqrt(lambda_k) * theta_k(omega) * phi_k(x)

    Parameters
    ----------
    x : array of shape (2, num_points) or (num_points, 2)
        Coordinates where the KL field is evaluated.
    theta : array of shape (L,)
        Standard normal KL coefficients.
    eigenvalues : array of shape (L,)
        Precomputed KL eigenvalues.
    eigenfunctions : array of shape (N, L)
        Precomputed KL eigenfunctions on a tensor-product grid.
    a : float
        Scaling parameter for permeability contrast.
    Returns
    -------
    Y : array of shape (1, num_points)
        scaled exponential of KL expansion values evaluated at the input coordinates.
    """
    x = np.asarray(x[:2]) # exclude z column which is 0 in 2d cases.
    theta = np.asarray(theta)
    eigenvalues = np.asarray(eigenvalues)
    eigenfunctions = np.asarray(eigenfunctions)

    if x.ndim != 2:
        raise ValueError("x must be a 2D array with shape (2, num_points)")
    if x.shape[0] != 2 and x.shape[1] == 2:
        x = x.T
    if x.shape[0] != 2:
        raise ValueError("x must have shape (2, num_points)")
    if theta.ndim != 1:
        raise ValueError("theta must be a 1D array of length L")
    if eigenvalues.ndim != 1:
        raise ValueError("eigenvalues must be a 1D array")
    if eigenfunctions.ndim != 2:
        raise ValueError("eigenfunctions must be a 2D array of shape (N, L)")

    L = theta.shape[0]
    if L > eigenvalues.shape[0]:
        raise ValueError("len(theta) cannot exceed len(eigenvalues)")
    if L > eigenfunctions.shape[1]:
        raise ValueError("len(theta) cannot exceed eigenfunctions.shape[1]")

    num_grid_points = eigenfunctions.shape[0]
    nx = int(round(np.sqrt(num_grid_points)))
    if nx * nx != num_grid_points:
        raise ValueError(
            "eigenfunctions.shape[0] must correspond to a square 2D tensor grid"
        )

    x_nodes = np.linspace(0.0, 1.0, nx)
    y_nodes = np.linspace(0.0, 1.0, nx)
    query_points = x.T

    phi_at_x = np.empty((query_points.shape[0], L))
    for k in range(L):
        phi_k_grid = eigenfunctions[:, k].reshape(nx, nx)
        interp = RegularGridInterpolator(
            (x_nodes, y_nodes),
            phi_k_grid,
            method='linear',
            bounds_error=False,
            fill_value=None,
        )
        phi_at_x[:, k] = interp(query_points)

    weighted = np.sqrt(np.maximum(eigenvalues[:L], 0.0)) * theta
    value = phi_at_x @ weighted
    return np.exp(a*value) + 1











"""
setup.py
This module provides setup functions for defining boundary conditions, source terms, and coefficient functions for various PDE test problems using FEniCSx/dolfinx. 
It includes utilities for homogeneous Dirichlet and Robin boundaries, as well as spatially varying coefficients such as channels, squares, and random fields. 
"""

import numpy as np
from dolfinx.fem import Function
from KL_expansion import kl_expansion


def getSetupSourceDirichlet(xL, yL, xR, yR, V, msh):
    def robin_boundary(x):
        bool_tmp = np.isclose(x[1], -1)
        return bool_tmp

    def dirichlet_boundary(x):
        bool_tmp = np.logical_or(np.isclose(x[0], xL), np.isclose(x[0], xR))
        bool_tmp = np.logical_or(bool_tmp, np.isclose(x[1], yR))
        bool_tmp = np.logical_or(bool_tmp, np.isclose(x[1], yL))
        
        bool_tmp = np.logical_and(bool_tmp, np.logical_not(robin_boundary(x)))   # Make sure that the point is not on the robin boundary, because we want to have disjoint boundary sets
        return bool_tmp
    
    # Define the Dirichlet boundary condition
    u_D = Function(V)
    u_D.interpolate(lambda x: np.full(x.shape[1], 0.0)) # Only implemented for homogeneous dirichlet BC
    # u_D.interpolate(lambda x : np.cos(direction[0] * k * x[0] + direction[1] * k * x[1]) + imag_unit * np.sin(direction[0] * k * x[0] + direction[1] * k * x[1]))
    # u_D.interpolate(lambda x: x[0] * x[1])

    # Define the Robin boundary condition
    u_R = Function(V)
    u_R.interpolate(lambda x: np.full(x.shape[1], 0.0)) 

    # Define source term
    f = Function(V)
    x0, y0 = 0.7, 0.9
    f.interpolate(lambda x: 10 + 0.0001*x[0])#lambda x: np.exp(-((x[0] - x0)**2 + (x[1] - y0)**2) ))

    # Coeff in PDE
    coeff_V_function = lambda x : np.exp(np.sin(10*np.pi*x[0]))*np.exp(np.sin(10*np.pi*x[1]))# highly oszillating

    return dirichlet_boundary, robin_boundary, u_D, f, coeff_V_function


def getSetupIID(xL, yL, xR, yR, V, contrast):
    def robin_boundary(x):
        bool_tmp = np.isclose(x[1], -1)
        return bool_tmp

    def dirichlet_boundary(x):
        bool_tmp = np.logical_or(np.isclose(x[0], xL), np.isclose(x[0], xR))
        bool_tmp = np.logical_or(bool_tmp, np.isclose(x[1], yR))
        bool_tmp = np.logical_or(bool_tmp, np.isclose(x[1], yL))
        
        bool_tmp = np.logical_and(bool_tmp, np.logical_not(robin_boundary(x)))   # Make sure that the point is not on the robin boundary, because we want to have disjoint boundary sets
        return bool_tmp
    
    # Define the Dirichlet boundary condition
    u_D = Function(V)
    u_D.interpolate(lambda x: np.full(x.shape[1], 0.0)) # Only implemented for homogeneous dirichlet BC

    # Define the Robin boundary condition
    u_R = Function(V)
    u_R.interpolate(lambda x: np.full(x.shape[1], 0.0)) 

    # Define source term
    f = Function(V)
    f.interpolate(lambda x: np.full(x.shape[1], 1.0))

    # Coeff in PDE
    coeff_A_function = getIIDCoefficient(contrast)

    return dirichlet_boundary, robin_boundary, u_D, f, coeff_A_function

def getIIDCoefficient(contrast = 10):
    Nx = 64 # mesh elements for the iid coefficient

    if contrast < 1:
        raise ValueError("Contrast must be greater than 1.")

    # set seed
    np.random.seed(0)
    # Generate an NxN matrix with entries uniformly distributed between 0 and 1
    coeff = np.random.uniform(low=0, high=1, size=(Nx, Nx))

    coeff = coeff  - np.min(coeff)  

    coeff_max = np.max(coeff)

    coeff = 1 + (contrast - 1) * coeff/coeff_max

    coeff_function = lambda x : coeff[np.floor(Nx * x[0]).astype(int), np.floor(Nx * x[1]).astype(int)]

    return coeff_function
    
def getSetupChannel(xL, yL, xR, yR, V, ny, Ny, contrast):
    def robin_boundary(x):
        bool_tmp = np.isclose(x[1], -1)
        return bool_tmp

    def dirichlet_boundary(x):
        bool_tmp = np.logical_or(np.isclose(x[0], xL), np.isclose(x[0], xR))
        bool_tmp = np.logical_or(bool_tmp, np.isclose(x[1], yR))
        bool_tmp = np.logical_or(bool_tmp, np.isclose(x[1], yL))
        
        bool_tmp = np.logical_and(bool_tmp, np.logical_not(robin_boundary(x)))   # Make sure that the point is not on the robin boundary, because we want to have disjoint boundary sets
        return bool_tmp
    
    # Define the Dirichlet boundary condition
    u_D = Function(V)
    u_D.interpolate(lambda x: np.full(x.shape[1], 0.0)) # Only implemented for homogeneous dirichlet BC

    # Define the Robin boundary condition
    u_R = Function(V)
    u_R.interpolate(lambda x: np.full(x.shape[1], 0.0)) 

    # Define source term
    f = Function(V)
    x0, y0 = 0.5, 0.9
    f.interpolate(lambda x: np.full(x.shape[1],  1.0)) 

    # Coeff in PDE
    coeff_A_function = lambda x : 1.0 + (contrast - 1) * channel_grid_pattern(x[0], x[1], ny, Ny)

    return dirichlet_boundary, robin_boundary, u_D, f, coeff_A_function


def vertical_channel(x_midpoint, y_midpoint, length, width, x, y):
    return np.logical_and(np.abs(x - x_midpoint) < width/2 - 1e-6, np.abs(y - y_midpoint) < length/2 - 1e-6)
    
def horizontal_channel(x_midpoint, y_midpoint, length, width, x, y):
    return np.logical_and(np.abs(y - y_midpoint) < width/2 - 1e-6, np.abs(x - x_midpoint) < length/2 - 1e-6)

def on_box(x_midpoint, y_midpoint, sidelength, h, layers, x, y):
    return np.logical_and(
        np.abs(x-x_midpoint) < (sidelength - h * layers)/2, 
        np.abs(y-y_midpoint) < (sidelength - h * layers)/2
    )

def channel_grid_pattern(x,y, ny, N):
    # Take the unit channel from below and repeat it in the four quadrants
    # N = 4
    on_channel = np.full(x.shape, False)
    length = 1
    width = 0.05
    # layers = 4 # Layers aways from dirichlet boundary
    layers = 8
    h = 1/ ny

    checkerboard_bool = True

    for i in range(N):
        for j in range(N):
            on_channel = np.logical_or(on_channel, np.logical_or(horizontal_channel(1/(2*N) + i/N, 1/(2*N) + j/N, length/N, width/N, x, y), np.logical_or(vertical_channel(1/(2*N) -1 /(2*N *5) + i/N, 1/(2*N) + j/N, length/N, width/N, x, y), 
                np.logical_or(vertical_channel(1/(2*N) + i/N, 1/(2*N) + j/N, length/N, width/N, x, y), vertical_channel(1/(2*N) +1 /(2*N *5) + i/N, 1/(2*N) + j/N, length/N, width/N, x, y)))))
            # only connect the channels on each second (checkerboard) domain
            if checkerboard_bool:
                sidelength = 1/ N
                on_channel = np.logical_and(on_channel, np.logical_not(on_box(1/(2*N) + i/N, 1/(2*N) + j/N, sidelength, h, 2 * layers, x, y )))
                # on unconnected regions, connect the three vertical channels by horizontal channels
                # connect upper part:
                on_channel = np.logical_or(on_channel, horizontal_channel(1/(2*N) + i/N,  (j+1)/N - layers * h , length/(3 * N), width/N, x, y))
                # connect lower part:
                on_channel = np.logical_or(on_channel, horizontal_channel(1/(2*N) + i/N,  j/N + layers * h , length/(3 * N), width/N, x, y))


                checkerboard_bool = False
            else:
                checkerboard_bool = True
        checkerboard_bool = not checkerboard_bool

    # Make that channel does not touch the boundary
    layers = 4
    on_channel = np.logical_and(on_channel, x > layers * h -1e-9)
    on_channel = np.logical_and(on_channel, y > layers * h -1e-9)
    on_channel = np.logical_and(on_channel, x < 1 - layers * h + 1e-9)
    on_channel = np.logical_and(on_channel, y < 1 - layers * h + 1e-9)


    return on_channel
    
def unit_channel(x,y):
    # Check whether the point (x,y) is on one of the following 
    # three vertical channels, which have midpoints at  x = 0.4, 0.5, 0.6, y = 0.5, 0.5, 0.5
    # length 0.5; and width 0.05.
    return np.logical_or(horizontal_channel(0.5, 0.5, 0.5, 0.05, x, y), np.logical_or(vertical_channel(0.4, 0.5, 0.5, 0.05, x, y), np.logical_or(vertical_channel(0.5, 0.5, 0.5, 0.05, x, y), vertical_channel(0.6, 0.5, 0.5, 0.05, x, y))))


def getSetupSquareMiddle(xL, yL, xR, yR, V, contrast):
    # Only consider homogeneous dirichlet BC
    def robin_boundary(x):
        bool_tmp = np.isclose(x[1], -1)
        return bool_tmp

    def dirichlet_boundary(x):
        bool_tmp = np.logical_or(np.isclose(x[0], xL), np.isclose(x[0], xR))
        bool_tmp = np.logical_or(bool_tmp, np.isclose(x[1], yR))
        bool_tmp = np.logical_or(bool_tmp, np.isclose(x[1], yL))
        
        bool_tmp = np.logical_and(bool_tmp, np.logical_not(robin_boundary(x)))   # Make sure that the point is not on the robin boundary, because we want to have disjoint boundary sets
        return bool_tmp
    
    # Define the Dirichlet boundary condition
    u_D = Function(V)
    u_D.interpolate(lambda x: np.full(x.shape[1], 0.0)) # Only implemented for homogeneous dirichlet BC

    # Define the Robin boundary condition
    u_R = Function(V)
    u_R.interpolate(lambda x: np.full(x.shape[1], 0.0)) 

    # Define source term
    f = Function(V)
    x0, y0 = 0.5, 0.5
    f.interpolate(lambda x: np.full(x.shape[1], 1.0))

    # Coeff in PDE
    coeff_A_function = lambda x : 1.0 + (contrast - 1) * square_middle(x[0], x[1])

    return dirichlet_boundary, robin_boundary, u_D, f, coeff_A_function

# Standalone functions
def crosspoint_1d(x):
    """
    Compute same-side indicator for thresholds (0.4, 0.35).
    Accepts x as shape (2, num_points) or [X, Y] where X, Y are arrays.
    Returns an array of ±1 with zeros mapped to -1 (for arrays),
    or a scalar ±1 for scalar inputs.
    """
    cond = ((x[0] < 0.4) & (x[1] < 0.35)) | ((x[0] > 0.4) & (x[1] > 0.35))
    same_side = cond.astype(int)
    same_side[same_side == 0] = -1
    return same_side

def crosspoint_1d_coeff_msgfem(x, contrast):
    """
    y can be a scalar or a NumPy array.
    Returns 1 + ((contrast - 1)/(contrast + 1)) * y * crosspoint_1d(x1, x2)
    """
    value = 1 + (contrast - 1)/(contrast + 1)*crosspoint_1d(x)
    return value 

def getSetupCrosspoint_1d(xL, yL, xR, yR, V, contrast):
    # Only consider homogeneous dirichlet BC
    def robin_boundary(x):
        bool_tmp = np.isclose(x[1], -1)
        return bool_tmp

    def dirichlet_boundary(x):
        bool_tmp = np.logical_or(np.isclose(x[0], xL), np.isclose(x[0], xR))
        bool_tmp = np.logical_or(bool_tmp, np.isclose(x[1], yR))
        bool_tmp = np.logical_or(bool_tmp, np.isclose(x[1], yL))
        
        bool_tmp = np.logical_and(bool_tmp, np.logical_not(robin_boundary(x)))   # Make sure that the point is not on the robin boundary, because we want to have disjoint boundary sets
        return bool_tmp
    
    # Define the Dirichlet boundary condition
    u_D = Function(V)
    u_D.interpolate(lambda x: np.full(x.shape[1], 0.0)) # Only implemented for homogeneous dirichlet BC

    # Define the Robin boundary condition
    u_R = Function(V)
    u_R.interpolate(lambda x: np.full(x.shape[1], 0.0)) 

    # Define source term
    f = Function(V)
    x0, y0 = 0.5, 0.5
    f.interpolate(lambda x: np.full(x.shape[1], 1.0))

    # Coeff in PDE
    coeff_A_function = lambda x : crosspoint_1d_coeff_msgfem(x, contrast)

    return dirichlet_boundary, robin_boundary, u_D, f, coeff_A_function

def square_middle(x,y):
    # Check whether the point (x,y) is in the square with midpoint (0.5, 0.5), length 0.1, and width 0.1.
    return np.logical_and(np.abs(x - 0.5) < 0.05 +1e-6, np.abs(y - 0.5) < 0.05 +1e-6)

def getSetupSkyscraper(xL, yL, xR, yR, V, contrast):
    # Only consider homogeneous dirichlet BC
    def robin_boundary(x):
        bool_tmp = np.isclose(x[1], -1)
        return bool_tmp

    def dirichlet_boundary(x):
        bool_tmp = np.logical_or(np.isclose(x[0], xL), np.isclose(x[0], xR))
        bool_tmp = np.logical_or(bool_tmp, np.isclose(x[1], yR))
        bool_tmp = np.logical_or(bool_tmp, np.isclose(x[1], yL))
        
        bool_tmp = np.logical_and(bool_tmp, np.logical_not(robin_boundary(x)))   # Make sure that the point is not on the robin boundary, because we want to have disjoint boundary sets
        return bool_tmp
    
    # Define the Dirichlet boundary condition
    u_D = Function(V)
    u_D.interpolate(lambda x: np.full(x.shape[1], 0.0)) # Only implemented for homogeneous dirichlet BC

    # Define the Robin boundary condition
    u_R = Function(V)
    u_R.interpolate(lambda x: np.full(x.shape[1], 0.0)) 

    # Define source term
    f = Function(V)
    x0, y0 = 0.5, 0.5
    f.interpolate(lambda x: np.full(x.shape[1], 1.0))

    # Coeff in PDE
    coeff_A_function = lambda x : np.array([skyscraper(x[0,i], x[1,i]) for i in range(x.shape[1])])

    return dirichlet_boundary, robin_boundary, u_D, f, coeff_A_function

def skyscraper(x,y):
    # Compute Nx and Ny
    Nx = int(np.floor(x * 8))
    Ny = int(np.floor(y * 8))
    
    # Compute Nx_c and Ny_c
    Nx_c = 8 * x - Nx
    Ny_c = 8 * y - Ny
    
    # Define the r_number array
    r_number = 1.0e5 * np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
        [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        [6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0],
        [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
        [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    ])
    
    # Initial value
    value = 1.0
    
    # Check conditions for Nx_c and Ny_c
    if 0.2 < Nx_c < 0.8 and 0.2 < Ny_c < 0.8:
        value = r_number[Ny, Nx]
    
    # Additional conditions based on rotations
    theta3 = np.pi / 3
    xc, yc = x - 0.825, y - 0.7
    x3 = np.cos(theta3) * xc + np.sin(theta3) * yc + 0.825
    y3 = -np.sin(theta3) * xc + np.cos(theta3) * yc + 0.675
    if 0.8 < x3 < 0.85 and 0.4 < y3 < 0.95:
        value = 1.5e5
    
    theta4 = 3 * np.pi / 4
    xc, yc = x - 0.875, y - 0.25
    x4 = np.cos(theta4) * xc + np.sin(theta4) * yc + 0.875
    y4 = -np.sin(theta4) * xc + np.cos(theta4) * yc + 0.25
    if 0.85 < x4 < 0.9 and 0 < y4 < 0.5:
        value = 1.5e5
    
    theta2 = 3.25 * np.pi / 10
    xc, yc = x - 0.15, y - 0.55
    x2 = np.cos(theta2) * xc + np.sin(theta2) * yc + 0.15
    y2 = -np.sin(theta2) * xc + np.cos(theta2) * yc + 0.55
    if 0.12 < x2 < 0.18 and -0.15 < y2 < 0.6:
        value = 2.0e6
    
    theta1 = np.pi / 10
    xc, yc = x - 0.15, y - 0.55
    x1 = np.cos(theta1) * xc + np.sin(theta1) * yc + 0.15
    y1 = -np.sin(theta1) * xc + np.cos(theta1) * yc + 0.55
    if 0.15 < x1 < 0.67 and 0.56 < y1 < 0.61:
        value = 2.0e6
    
    return value

def channel(x, parameters):
    """
    Global channel coefficient on the unit square [0, 1] x [0, 1].

    The unit square is partitioned into 4 x 4 = 16 equal square subdomains
    of side length ``1/4``. One rectangular channel is placed inside every
    subdomain. Each channel has the same fixed geometry:

        - width  (x-extent) = side / 10 = 1/40
        - height (y-extent) = side / 2  = 1/8

    The channel of subdomain ``k`` is offset from the lower-left corner of
    that subdomain by ``(cx_k, cy_k)`` and its coefficient value on the
    channel is ``h_k``.  Away from every channel the function returns the
    fixed background value ``0.1``.

    To guarantee that no channel intersects the global boundary or the
    interfaces between subdomains, the offsets must satisfy
    ``0 < cx_k < side - width`` and ``0 < cy_k < side - height``.

    Parameters
    ----------
    x : array-like, shape (2, num_gridpoints)
        Spatial coordinates; ``x[0]`` are x-coords, ``x[1]`` are y-coords.
    parameters : array-like, shape (3 * 16,) = (48,)
        Concatenated parameters of the 16 channels. Subdomains are indexed
        in row-major order ``k = i * 4 + j`` with ``i`` the column index
        (x-direction) and ``j`` the row index (y-direction).  For every
        subdomain ``k``:

            parameters[3*k]     = cx_k   (x-offset from subdomain corner)
            parameters[3*k + 1] = cy_k   (y-offset from subdomain corner)
            parameters[3*k + 2] = h_k    (channel value / contrast)

    Returns
    -------
    z : np.ndarray, shape (num_gridpoints,)
        Coefficient field (``0.1`` outside every channel, ``h_k`` inside the
        channel of subdomain ``k``).
    """
    N = 4
    side = 1.0 / N
    width_channel = side / 10.0
    high_channel = side / 2.0

    parameters = np.asarray(parameters).ravel()
    if parameters.size != 3 * N * N:
        raise ValueError(
            f"channel expects {3 * N * N} parameters (3 per each of the "
            f"{N * N} subdomains); got {parameters.size}."
        )

    background = 0.1
    value = np.full(x.shape[1], background)
    for i in range(N):
        for j in range(N):
            k = i * N + j
            x0 = i * side
            y0 = j * side
            channel_x0 = parameters[3 * k]
            channel_y0 = parameters[3 * k + 1]
            height = parameters[3 * k + 2]

            on_channel = np.logical_and(
                np.logical_and(x[0] >= x0 + channel_x0,
                               x[0] <= x0 + channel_x0 + width_channel),
                np.logical_and(x[1] >= y0 + channel_y0,
                               x[1] <= y0 + channel_y0 + high_channel),
            )
            value = np.where(on_channel, height, value)
    return value


def channel_rotated(x, parameters):
    """
    Global rotated-channel coefficient on the unit square [0, 1] x [0, 1].

    Identical to :func:`channel` but every channel additionally carries a
    rotation angle. The unit square is partitioned into 4 x 4 = 16 equal
    square subdomains of side length ``1/4`` and one rectangular channel is
    placed inside every subdomain. Each channel has the same fixed geometry:

        - width  (short axis) = side / 10 = 1/40
        - height (main axis)  = side / 2  = 1/8

    The channel of subdomain ``k`` is offset from the lower-left corner of
    that subdomain by ``(cx_k, cy_k)``, its coefficient value on the channel
    is ``h_k`` and it is rotated about its own center by ``theta_k`` (its
    main/long axis is rotated by that angle).  Away from every channel the
    function returns the fixed background value ``0.1``.

    Parameters
    ----------
    x : array-like, shape (2, num_gridpoints)
        Spatial coordinates; ``x[0]`` are x-coords, ``x[1]`` are y-coords.
    parameters : array-like, shape (4 * 16,) = (64,)
        Concatenated parameters of the 16 channels. Subdomains are indexed
        in row-major order ``k = i * 4 + j`` with ``i`` the column index
        (x-direction) and ``j`` the row index (y-direction).  For every
        subdomain ``k``:

            parameters[4*k]     = cx_k     (x-offset from subdomain corner)
            parameters[4*k + 1] = cy_k     (y-offset from subdomain corner)
            parameters[4*k + 2] = h_k      (channel value / contrast)
            parameters[4*k + 3] = theta_k  (rotation angle in [0, 2*pi])

    Returns
    -------
    z : np.ndarray, shape (num_gridpoints,)
        Coefficient field (``0.1`` outside every channel, ``h_k`` inside the
        rotated channel of subdomain ``k``).
    """
    N = 4
    side = 1.0 / N
    width_channel = side / 10.0
    high_channel = side / 2.0

    parameters = np.asarray(parameters).ravel()
    if parameters.size != 4 * N * N:
        raise ValueError(
            f"channel_rotated expects {4 * N * N} parameters (4 per each of "
            f"the {N * N} subdomains); got {parameters.size}."
        )

    background = 0.1
    value = np.full(x.shape[1], background)
    for i in range(N):
        for j in range(N):
            k = i * N + j
            x0 = i * side
            y0 = j * side
            channel_x0 = parameters[4 * k]
            channel_y0 = parameters[4 * k + 1]
            height = parameters[4 * k + 2]
            theta = parameters[4 * k + 3]

            # Center of the (un-rotated) channel; used as center of rotation.
            center_x = x0 + channel_x0 + width_channel / 2.0
            center_y = y0 + channel_y0 + high_channel / 2.0

            # Translate the point so the channel center is at the origin and
            # apply the inverse rotation to map it into the channel's own
            # (axis-aligned) coordinate frame.
            x_translated = x[0] - center_x
            y_translated = x[1] - center_y
            cos_theta = np.cos(-theta)
            sin_theta = np.sin(-theta)
            x_unrotated = x_translated * cos_theta - y_translated * sin_theta
            y_unrotated = x_translated * sin_theta + y_translated * cos_theta

            on_channel = np.logical_and(
                np.abs(x_unrotated) <= width_channel / 2.0,
                np.abs(y_unrotated) <= high_channel / 2.0,
            )
            value = np.where(on_channel, height, value)
    return value


def channel_sub_dom_five(x, parameters):
    """
    Channel configuration for subdomain five. Works only for subdomains that come from a regular 4x4 partition of the global domain

    Parameters
    ----------
    x : array-like
        Input coordinates.
    parameters : array-like
        Parameters defining the channel position and value 
    """
    x0 = 0.234375 # lower right corner of subdomain
    y1 = 0.515625 # upper left corner

    width_channel = np.abs(y1-x0)/10
    high_channel = np.abs(y1-x0)/2
    channel_x0 = parameters[0]
    channel_y0 = parameters[1]

    on_channel = np.logical_and(np.logical_and( x[0] >= x0 + channel_x0, x[0] <= x0 + channel_x0 + width_channel),
                                np.logical_and( x[1] >= x0 + channel_y0, x[1] <= x0 + channel_y0 + high_channel))
    
    return parameters[2] * on_channel


def rotated_channel_sub_dom_five(x, parameters):
    """
    Channel configuration for subdomain five, with rotation.

    This method defines a channel within a subdomain and allows for it to be
    rotated by a given angle theta. It works for subdomains that originate
    from a regular 4x4 partition of the global domain.

    Parameters
    ----------
    x : array-like
        Input coordinates [x, y] for which to check if they are in the channel.
    parameters : array-like
        Parameters defining the channel's position and value.
        - parameters[0]: x-offset from the corner of the subdomain.
        - parameters[1]: y-offset from the corner of the subdomain.
        - parameters[2]: Rotation angle of the channel

    Returns
    -------
    float
        The channel value if the point x is inside the rotated channel,
        otherwise 0.
    """
    # --- Define Subdomain and Channel Geometry ---
    # These values are based on a regular 4x4 partition of a unit square.
    x0 = 0.234375  # Lower-left x-coordinate of the subdomain corner
    y0 = 0.234375  # Assuming the subdomain is square, so y0 is the same
    y1 = 0.515625  # Upper-right y-coordinate
    
    width_channel = np.abs(y1 - x0) / 10
    high_channel = np.abs(y1 - x0) / 2
    
    channel_x0 = parameters[0]
    channel_y0 = parameters[1]
    theta = parameters[2]

    # --- Rotation Logic ---
    # 1. Calculate the center of the original, un-rotated channel.
    # This will be our center of rotation.
    center_x = x0 + channel_x0 + width_channel / 2.0
    center_y = y0 + channel_y0 + high_channel / 2.0

    # 2. Translate the input point's coordinates so the center of rotation is at the origin.
    x_translated = x[0] - center_x
    y_translated = x[1] - center_y

    # 3. Apply the inverse rotation to the translated point.
    # This brings the point into the coordinate system of the un-rotated channel.
    cos_theta = np.cos(-theta)
    sin_theta = np.sin(-theta)
    x_unrotated = x_translated * cos_theta - y_translated * sin_theta
    y_unrotated = x_translated * sin_theta + y_translated * cos_theta

    # 4. Check if the un-rotated point lies inside the original, un-rotated channel.
    # The original channel is now centered at (0,0) in this new coordinate system.
    on_channel = np.logical_and(
        np.abs(x_unrotated) <= width_channel / 2.0,
        np.abs(y_unrotated) <= high_channel / 2.0
    )

    return 1000 * on_channel


def channel_smooth(x, parameters, sharpness=100):
    """
    Smooth bump function centered at 'parameters[:2]' with height 'parameters[2]'.

    Parameters
    ----------
    x : array-like
        Input coordinates.
    parameters : array-like
        Parameters defining the bump center and height.
    sharpness : float, optional
        Sharpness of the bump function. Default is 100.
    """
    x0 = 0.234375 # lower right corner of subdomain
    y1 = 0.515625 # upper left corner
    center = parameters[:2]
    height = parameters[2]

    w = np.abs(y1-x0)/5

    left = np.array(center) - np.array(w) / 2
    right = np.array(center) + np.array(w) / 2

    bump_x = 1 / (1 + np.exp(-sharpness * (x[0] - left[0]))) - 1 / (1 + np.exp(-sharpness * (x[0] - right[0])))
    bump_y = 1 / (1 + np.exp(-sharpness * (x[1] - left[1]))) - 1 / (1 + np.exp(-sharpness * (x[1] - right[1])))
    return height * bump_x * bump_y


def multiscale_sincos(x, parameters, background=0.1):
    """
    Positive, globally defined multiscale coefficient on [0, 1] x [0, 1].

    The coefficient is the exponential of a truncated sine-cosine (Fourier)
    expansion. Taking the exponential guarantees strict positivity for *any*
    parameter values, while the individual modes set the oscillation
    frequencies and amplitudes of the field:

        g(x, y) = sum_k [ a_k * sin(2*pi*(kx_k*x + ky_k*y))
                          + b_k * cos(2*pi*(kx_k*x + ky_k*y)) ]
        A(x, y) = background + exp(g(x, y))

    A wide spread of the frequencies ``(kx_k, ky_k)`` (coarse and fine modes)
    makes the coefficient genuinely multiscale. The constant ``background``
    provides a guaranteed positive lower bound independent of the amplitudes.

    Parameters
    ----------
    x : array-like, shape (2, num_gridpoints)
        Spatial coordinates; ``x[0]`` are x-coords, ``x[1]`` are y-coords.
    parameters : array-like, shape (4 * K,)
        Concatenated parameters of the ``K`` Fourier modes. For every mode
        ``k``:

            parameters[4*k]     = kx_k   (x-frequency, number of oscillations)
            parameters[4*k + 1] = ky_k   (y-frequency, number of oscillations)
            parameters[4*k + 2] = a_k    (sine amplitude)
            parameters[4*k + 3] = b_k    (cosine amplitude)
    background : float, optional
        Positive constant added to ``exp(g)``. Default is ``0.1``.

    Returns
    -------
    z : np.ndarray, shape (num_gridpoints,)
        The strictly positive coefficient field evaluated at ``x``.
    """
    p = np.asarray(parameters).ravel()
    if p.size % 4 != 0:
        raise ValueError(
            "multiscale_sincos expects 4 parameters per mode "
            f"(kx, ky, a, b); got {p.size} which is not divisible by 4."
        )
    K = p.size // 4

    g = np.zeros(x.shape[1])
    for k in range(K):
        kx, ky, a, b = p[4 * k: 4 * k + 4]
        phase = 2.0 * np.pi * (kx * x[0] + ky * x[1])
        g = g + a * np.sin(phase) + b * np.cos(phase)

    return background + np.exp(g)

# Standalone functions
def crosspoint_1d(x):
    """
    Compute same-side indicator for thresholds (0.4, 0.35).
    Accepts x as shape (2, num_points) or [X, Y] where X, Y are arrays.
    Returns an array of ±1 with zeros mapped to -1 (for arrays),
    or a scalar ±1 for scalar inputs.
    """
    cond = ((x[0] < 0.4) & (x[1] < 0.35)) | ((x[0] > 0.4) & (x[1] > 0.35))
    same_side = cond.astype(int)
    same_side[same_side == 0] = -1
    return same_side

def crosspoint_1d_coeff(x, parameters):
    """
    y can be a scalar or a NumPy array.
    Returns 1 + ((contrast - 1)/(contrast + 1)) * y * crosspoint_1d(x1, x2)
    """
    y = parameters[0]
    contrast = parameters[1]
    value = 1 + (contrast - 1)/(contrast + 1)*y*crosspoint_1d(x)
    np.save('x.npy', x)
    return value 

# random lines
def random_lines(x, p):
    """
    Coefficient function A(x, p) that creates a piecewise-constant field
    with high-conductivity random lines on a low background.

    Parameters
    ----------
    x : np.ndarray, shape (2, num_gridpoints)
        Spatial coordinates; x[0] = x-coords, x[1] = y-coords.
    p : np.ndarray, shape (num_lines * 4,)
        Parameter vector.  For each line i the 4 entries are:
            p[4*i]     – centre x  (cx)
            p[4*i + 1] – centre y  (cy)
            p[4*i + 2] – half-length of the line
            p[4*i + 3] – rotation angle theta (rad)

    Returns
    -------
    z : np.ndarray, shape (num_gridpoints,)
        Coefficient values (0.01 background, 100.0 on lines).
    """
    num_lines = len(p) // 4
    z = np.full(x.shape[1], 0.01)

    half_width = 0.01  # fixed half-width as in the original code

    for i in range(num_lines):
        cx          = p[4 * i]
        cy          = p[4 * i + 1]
        half_length = p[4 * i + 2]
        theta       = p[4 * i + 3]

        dx = x[0] - cx
        dy = x[1] - cy

        local_x =  dx * np.cos(theta) + dy * np.sin(theta)
        local_y = -dx * np.sin(theta) + dy * np.cos(theta)

        mask = (np.abs(local_x) <= half_length) & (np.abs(local_y) <= half_width)
        z[mask] = 100.0

    return z


def bubble_coeff(x, p):
    """
    Coefficient function A(x, p) that creates a piecewise-constant field with
    five circular "bubbles" of random center, radius and height on a low
    background. Bubbles are confined to a square subdomain [x0, y1] x [x0, y1]
    whose bounds are also encoded in the parameter vector p.

    Parameters
    ----------
    x : np.ndarray, shape (2, num_gridpoints)
        Spatial coordinates; x[0] = x-coords, x[1] = y-coords.
    p : np.ndarray, shape (2 + 5 * 4,) = (22,)
        Parameter vector:
            p[0] = x0   subdomain lower bound (used for both x and y)
            p[1] = y1   subdomain upper bound (used for both x and y)
        For each bubble i = 0, ..., 4:
            p[2 + 4 * i]     = cx      center x coordinate
            p[2 + 4 * i + 1] = cy      center y coordinate
            p[2 + 4 * i + 2] = radius
            p[2 + 4 * i + 3] = height  (contrast value inside the bubble)

    The sampling of the bubble parameters (done outside this function, e.g.
    in ex_FNO.py) must ensure that every bubble is fully contained in the
    subdomain, i.e. cx - r >= x0, cx + r <= y1, cy - r >= x0, cy + r <= y1.

    Returns
    -------
    z : np.ndarray, shape (num_gridpoints,)
        Coefficient values (0.1 background, up to 100 inside a bubble).
    """
    background = 0.1
    num_bubbles = (len(p) - 2) // 4

    z = np.full(x.shape[1], background)

    for i in range(num_bubbles):
        cx     = p[2 + 4 * i]
        cy     = p[2 + 4 * i + 1]
        radius = p[2 + 4 * i + 2]
        height = p[2 + 4 * i + 3]

        dx = x[0] - cx
        dy = x[1] - cy
        mask = (dx * dx + dy * dy) <= radius * radius
        z[mask] = height

    return z


def FNO_coeffs(xL, yL, xR, yR, V, msh, parameters, store_tag):
        
        def robin_boundary(x):
            bool_tmp = np.isclose(x[1], -1)
            return bool_tmp

        def dirichlet_boundary(x):
            bool_tmp = np.logical_or(np.isclose(x[0], xL), np.isclose(x[0], xR))
            bool_tmp = np.logical_or(bool_tmp, np.isclose(x[1], yR))
            bool_tmp = np.logical_or(bool_tmp, np.isclose(x[1], yL))
            
            bool_tmp = np.logical_and(bool_tmp, np.logical_not(robin_boundary(x)))   # Make sure that the point is not on the robin boundary, because we want to have disjoint boundary sets
            return bool_tmp
        
        # Define the Dirichlet boundary condition
        u_D = Function(V)
        # TODO: return correct dirichlet boundary for later global solve.
        # Define the Robin boundary condition
        u_R = Function(V)
        u_R.interpolate(lambda x: np.full(x.shape[1], 0.0)) 

        # Define source term
        # TODO: think of meaningful source term.
        f = Function(V)
        x0, y0 = 0.7, 0.9
        f.interpolate(lambda x: np.exp(-((x[0] - x0)**2 + (x[1] - y0)**2) ))
        # coeff in PDE
        if store_tag == 'channel_coeff':
            coeff_A_function = lambda x : channel(x, parameters)
        elif store_tag == 'channel_rotated_coeff':
            coeff_A_function = lambda x : channel_rotated(x, parameters)
        elif store_tag == 'channel_sub_dom_five_coeff':
            coeff_A_function = lambda x : 1 +  channel_sub_dom_five(x, parameters)
        elif store_tag == 'sinus_coeff':
            coeff_A_function = lambda x : np.exp(np.sin(parameters[0]*np.pi*x[0]) + np.sin(parameters[1]*np.pi*x[1]))# highly oszillating
        elif store_tag == 'multiscale_sincos_coeff':
            coeff_A_function = lambda x : multiscale_sincos(x, parameters)
        elif store_tag == 'channel_low_coeff':
            coeff_A_function = lambda x : channel(x, parameters)
        elif (store_tag == 'channel_smooth_coeff') or (store_tag == 'channel_low_smooth_coeff'):
            coeff_A_function = lambda x : 1 + channel_smooth(x, parameters)
        elif store_tag == 'crosspoint_1d_coeff':
            coeff_A_function = lambda x : crosspoint_1d_coeff(x, parameters)
        elif store_tag == 'random_lines':
            coeff_A_function = lambda x : random_lines(x, parameters)
        elif store_tag == 'bubble_coeff':
            coeff_A_function = lambda x : bubble_coeff(x, parameters)
        elif store_tag == 'rotated_channel_coeff':
            coeff_A_function = lambda x : 1 + rotated_channel_sub_dom_five(x, parameters)
        elif store_tag == 'kl_coeff':
            eigenvalues = np.load('kl_data/eigenvalues.npy')
            eigenfunctions = np.load('kl_data/eigenfunctions.npy')
            coeff_A_function = lambda x : kl_expansion(x, parameters, eigenvalues, eigenfunctions)
        else:
            raise ValueError('Coefficient function for %s not defined, please implement here...!'%(store_tag))
        

        return dirichlet_boundary, robin_boundary, u_D, f, coeff_A_function


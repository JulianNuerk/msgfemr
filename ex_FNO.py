"""
Generates Data and uses FNOs to solve the local Eigenvalue problems in the MSGFEM method.
"""

import os as ops

# Limit the number of BLAS/OpenMP threads *before* numpy / dolfinx are
# imported. Every Ray task below requests a single CPU, so we must prevent the
# linear-algebra backends from spawning many threads per task, which would
# oversubscribe the cores allocated on the LSF batch_cpu node.
for _thread_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    ops.environ.setdefault(_thread_var, "1")

import argparse
import experiments_driver as ed
from pathlib import Path
import ray
import numpy as np
import helper
import datetime
from KL_expansion import discretize_covariance_2d, solve_eigenvalue_problem, kl_expansion

# Setup argparse to accept command line arguments
parser = argparse.ArgumentParser(description="Generate data for MSGFEM")
parser.add_argument("--store_tag", type=str, required=True, help="Varying parameter tag")
parser.add_argument("--num_samples", type=int, default=1200, help="Number of samples to generate")
parser.add_argument("--nloc", type=int, default=5, help="Number of local basis functions")
parser.add_argument("--subdom_idx_list", type=int, nargs="+", default=[5, 6, 9, 10],
                    help="Subdomain indices whose eigenproblems are solved in parallel per sample")

args = parser.parse_args()
# Set store_tag from the command line argument
store_tag = args.store_tag
num_samples = args.num_samples
nloc = args.nloc 

# Define the absolute path to your target scratch directory
BASE_DIR = Path("/fs/scratch/rb_bd_dlp_rng_dl01_cr_MSO_employees/students/nuj7rng/msgfem_data/fenics_out_data")
OUT_DIR = BASE_DIR / store_tag / f"samples_{num_samples}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# no need to play with but can be changed
deg = 1
Ny = 4
ny = 2**8
ol = 2
os = 2
rho = 0.0
# Subdomains for which FNO data is generated. For every parameter sample the
# eigenproblem of each of these subdomains is solved in parallel via Ray.
subdom_idx_list = args.subdom_idx_list

# parameters not to play with 
x0 = 0.234375 # lower right corner of subdomain
y1 = 0.515625 # upper left corner 
eps = 1e-08
np.random.seed(42)

if store_tag == 'channel_coeff':
    # Global channel coefficient on [0,1]x[0,1] with a regular 4x4 partition.
    # Each of the 16 subdomains (side = 1/4) hosts one channel of size
    # (width, height) = (side/10, side/2). Per subdomain we sample
    # (channel_x_offset, channel_y_offset, height) so that the channel stays
    # strictly inside the subdomain (and hence away from the global boundary
    # and from all subdomain interfaces).
    N_sub = 4
    side = 1.0 / N_sub
    width_channel = side / 10.0
    high_channel = side / 2.0
    p0_bound = side - width_channel - eps  # max channel_x_offset
    p1_bound = side - high_channel - eps   # max channel_y_offset
    low  = np.tile([eps, eps, 1.0],      N_sub * N_sub)
    high = np.tile([p0_bound, p1_bound, 1e+03], N_sub * N_sub)
    parameters = np.random.uniform(low=low, high=high,
                                   size=(num_samples, 3 * N_sub * N_sub))
elif store_tag == 'channel_rotated_coeff':
    # Same geometry as 'channel_coeff' but every channel additionally carries
    # a rotation angle in [0, 2*pi]. Per subdomain we sample
    # (channel_x_offset, channel_y_offset, height, angle).
    N_sub = 4
    side = 1.0 / N_sub
    width_channel = side / 10.0
    high_channel = side / 2.0
    p0_bound = side - width_channel - eps  # max channel_x_offset
    p1_bound = side - high_channel - eps   # max channel_y_offset
    low  = np.tile([eps, eps, 1.0, 0.0],              N_sub * N_sub)
    high = np.tile([p0_bound, p1_bound, 1e+03, 2 * np.pi], N_sub * N_sub)
    parameters = np.random.uniform(low=low, high=high,
                                   size=(num_samples, 4 * N_sub * N_sub))
elif store_tag == 'sinus_coeff':
    parameters = np.random.uniform(low=[5,10], high=[10, 20], size=(num_samples, 2))
elif store_tag == 'multiscale_sincos_coeff':
    # Positive multiscale coefficient A = 0.1 + exp(sum_k a_k sin(2*pi*(kx*x+ky*y))
    #                                              + b_k cos(2*pi*(kx*x+ky*y))).
    # Per mode we sample integer frequencies spanning coarse-to-fine scales and
    # amplitudes that decay with frequency so high modes stay bounded.
    K = 5                                   # number of Fourier modes
    k_min, k_max = 1, 16                    # frequency range -> multiscale content
    kx = np.random.randint(k_min, k_max + 1, size=(num_samples, K)).astype(float)
    ky = np.random.randint(k_min, k_max + 1, size=(num_samples, K)).astype(float)
    scale = 1.0 / np.sqrt(kx ** 2 + ky ** 2)   # amplitude decay with frequency
    a = np.random.uniform(-1.0, 1.0, size=(num_samples, K)) * scale
    b = np.random.uniform(-1.0, 1.0, size=(num_samples, K)) * scale
    parameters = np.stack([kx, ky, a, b], axis=2).reshape(num_samples, 4 * K)
elif store_tag == 'channel_low_coeff':
    # Same geometry as 'channel_coeff' but with a much lower height range.
    N_sub = 4
    side = 1.0 / N_sub
    width_channel = side / 10.0
    high_channel = side / 2.0
    p0_bound = side - width_channel - eps
    p1_bound = side - high_channel - eps
    low  = np.tile([eps, eps, 1.0],   N_sub * N_sub)
    high = np.tile([p0_bound, p1_bound, 10.0], N_sub * N_sub)
    parameters = np.random.uniform(low=low, high=high,
                                   size=(num_samples, 3 * N_sub * N_sub))
elif store_tag == 'channel_smooth_coeff':
    eps = 1e-04
    wx, wy = 2*[np.abs(y1-x0)/5]
    parameters = np.random.uniform(low=[x0 + eps + wx/2 ,x0 + eps + wy/2,1], high=[y1 - eps - wx/2, y1-eps-wy/2, 100], size=(num_samples, 3))
elif store_tag == 'channel_low_smooth_coeff':
    eps = 1e-04
    wx, wy = 2*[np.abs(y1-x0)/5]
    parameters = np.random.uniform(low=[x0 + eps + wx/2 ,x0 + eps + wy/2,1], high=[y1 - eps - wx/2, y1-eps-wy/2, 10], size=(num_samples, 3))
elif store_tag == 'crosspoint_1d_coeff':
    contrast = 10000
    parameters = contrast*np.ones((num_samples, 2))
    parameters[:,0] = np.linspace(0, 1,num_samples)
elif store_tag == 'random_lines':
    num_lines = 10
    centers = np.random.uniform(0.1, 0.9, (num_samples, num_lines, 2))  
    lengths = np.random.uniform(0.05, 0.4, (num_samples, num_lines, 1)) 
    angles  = np.random.uniform(0, np.pi, (num_samples, num_lines, 1))  
    parameters = np.concatenate([centers, lengths, angles], axis=2).reshape(num_samples, num_lines * 4) 
elif store_tag == 'bubble_coeff':
    # Five circular bubbles per sample confined to the square subdomain
    # [x0, y1] x [x0, y1]. Layout of the parameter vector per sample:
    #   [x0, y1, cx_1, cy_1, r_1, h_1, ..., cx_5, cy_5, r_5, h_5]
    num_bubbles = 5
    side = y1 - x0                          # subdomain side length
    r_min = side / 50                       # minimum bubble radius
    r_max = side / 6                        # keeps bubbles well inside subdomain
    h_min = 1.0                             # minimum bubble height (contrast)
    h_max = 100.0                           # maximum bubble height (contrast)

    parameters = np.zeros((num_samples, 2 + num_bubbles * 4))
    parameters[:, 0] = x0
    parameters[:, 1] = y1
    for s in range(num_samples):
        for b in range(num_bubbles):
            r  = np.random.uniform(r_min, r_max)
            # Sample center so the whole disk lies strictly inside the subdomain
            cx = np.random.uniform(x0 + r + eps, y1 - r - eps)
            cy = np.random.uniform(x0 + r + eps, y1 - r - eps)
            h  = np.random.uniform(h_min, h_max)
            parameters[s, 2 + 4 * b: 2 + 4 * (b + 1)] = [cx, cy, r, h]
elif store_tag == 'rotated_channel_coeff':
    p0_bound = 9/10*(y1-x0) - eps 
    p1_bound = 1/2*(y1-x0) - eps 
    angles  = np.random.uniform(0, np.pi, (num_samples, 1))  
    cy_coords = np.random.uniform(low=[eps,eps], high=[p0_bound, p1_bound], size=(num_samples, 2))
    parameters = np.concatenate([cy_coords, angles], axis=1)
elif store_tag == 'kl_coeff':
    L = 100
    print(f'KL coeff: we assume that L = {L} eigenfunctions and values are already precomputed and stored in /kl_data!')
    parameters = np.random.normal(size=(num_samples, L))
else:
    raise ValueError('store_tag %s not defined!'%(store_tag))

print('Starting data generation for case: %s'%(store_tag))

# Initialise Ray using the cores allocated by LSF. On the RNG LSF cluster the
# number of reserved slots is exposed through LSB_DJOB_NUMPROC; fall back to
# Ray's autodetection when running outside of a batch job.
num_cpus_env = ops.environ.get("LSB_DJOB_NUMPROC")
num_cpus = int(num_cpus_env) if num_cpus_env else None
if not ray.is_initialized():
    ray.init(num_cpus=num_cpus, ignore_reinit_error=True)
print(f"Ray initialised with resources: {ray.cluster_resources()}", flush=True)


@ray.remote(num_cpus=1)
def generate_sample_subdom(sample_idx, parameter, subdom_idx):
    """Solve the local eigenproblem of one subdomain for one parameter sample.

    Each invocation is an independent Ray task (one CPU) and writes its own
    output files, so that the (sample, subdomain) combinations run in parallel
    without transferring the large arrays back to the driver.
    """
    (coeff_A_FNO, basis_funs, vecs_tmp, eig_vals,
     input_indices, eig_solve_time) = ed.run_fno_data_generation(
        deg, Ny, ny, ol, os, nloc, rho, store_tag, parameter, subdom_idx)

    np.save(OUT_DIR / f"phi_sub_dom_{subdom_idx}_sample_{sample_idx}.npy", basis_funs)
    np.save(OUT_DIR / f"coeff_A_sub_dom_{subdom_idx}_sample_{sample_idx}.npy", coeff_A_FNO)
    np.save(OUT_DIR / f"eig_vals_sub_dom_{subdom_idx}_sample_{sample_idx}.npy", eig_vals)
    np.save(OUT_DIR / f"vecs_tmp_sub_dom_{subdom_idx}_sample_{sample_idx}.npy", vecs_tmp)
    np.save(OUT_DIR / f"indices_for_reshape_2d_sub_dom_{subdom_idx}.npy", input_indices)

    return sample_idx, subdom_idx, eig_solve_time


# Dispatch one Ray task per (sample, subdomain) combination. For a fixed sample
# all subdomains in subdom_idx_list are solved in parallel; submitting every
# combination up front lets Ray keep all allocated cores busy across samples.
futures = [
    generate_sample_subdom.remote(idx, parameter, subdom_idx)
    for idx, parameter in enumerate(parameters)
    for subdom_idx in subdom_idx_list
]

# Collect eigsh solve times per subdomain as the tasks complete.
eig_solve_times = {subdom_idx: np.zeros(num_samples) for subdom_idx in subdom_idx_list}
remaining = futures
while remaining:
    done, remaining = ray.wait(remaining, num_returns=1)
    sample_idx, subdom_idx, eig_solve_time = ray.get(done[0])
    eig_solve_times[subdom_idx][sample_idx] = eig_solve_time
    print(f"Completed sample {sample_idx}, subdomain {subdom_idx} "
          f"(eigsh time {eig_solve_time:.4f} s)", flush=True)

for subdom_idx in subdom_idx_list:
    np.save(OUT_DIR / f"eig_solve_times_sub_dom_{subdom_idx}.npy",
            eig_solve_times[subdom_idx])

if store_tag == "crosspoint_1d_coeff":
    np.save(OUT_DIR / f"input_paras_num_samples_{num_samples}.npy", parameters[:,0])
else:
    np.save(OUT_DIR / f"input_paras_num_samples_{num_samples}.npy", parameters)
        
print('FNO data generation completed!')
ray.shutdown()

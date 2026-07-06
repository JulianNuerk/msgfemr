"""
Generates Data and uses FNOs to solve the local Eigenvalue problems in the MSGFEM method.
"""

import argparse
import experiments_driver as ed
from pathlib import Path
import ray
import numpy as np
import helper
import datetime
import os as ops
from KL_expansion import discretize_covariance_2d, solve_eigenvalue_problem, kl_expansion

# Setup argparse to accept command line arguments
parser = argparse.ArgumentParser(description="Generate data for MSGFEM")
parser.add_argument("--store_tag", type=str, required=True, help="Varying parameter tag")
parser.add_argument("--num_samples", type=int, default=1200, help="Number of samples to generate")
parser.add_argument("--nloc", type=int, default=5, help="Number of local basis functions")

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
subdom_idx = 5

# parameters not to play with 
x0 = 0.234375 # lower right corner of subdomain
y1 = 0.515625 # upper left corner 
eps = 1e-08
np.random.seed(42)

if store_tag == 'channel_coeff':
    p0_bound = 9/10*(y1-x0) - eps # guarantees that channel stays in subdomain 
    p1_bound = 1/2*(y1-x0) - eps # guarantees that channel stays in subdomain 
    parameters = np.random.uniform(low=[eps,eps,1], high=[p0_bound, p1_bound, 1e+06], size=(num_samples, 3))
elif store_tag == 'sinus_coeff':
    parameters = np.random.uniform(low=[5,10], high=[10, 20], size=(num_samples, 2))
elif store_tag == 'channel_low_coeff':
    p0_bound = 9/10*(y1-x0) - eps  
    p1_bound = 1/2*(y1-x0) - eps 
    parameters = np.random.uniform(low=[eps,eps,1], high=[p0_bound, p1_bound, 10], size=(num_samples, 3))
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
for idx, parameter in enumerate(parameters):
    print('Sample Nr.:', idx)
    print('Parameters: %s.'%(parameter))
    coeff_A_FNO, basis_funs, vecs_tmp, eig_vals, input_indices = ed.run_fno_data_generation(deg, Ny, ny, ol, os, nloc, rho, store_tag, parameter, subdom_idx)

    np.save(OUT_DIR / f"phi_sub_dom_{subdom_idx}_sample_{idx}.npy", basis_funs)
    np.save(OUT_DIR / f"coeff_A_sub_dom_{subdom_idx}_sample_{idx}.npy", coeff_A_FNO)
    np.save(OUT_DIR / f"eig_vals_sub_dom_{subdom_idx}_sample_{idx}.npy", eig_vals)
    np.save(OUT_DIR / f"vecs_tmp_sub_dom_{subdom_idx}_sample_{idx}.npy", vecs_tmp)
    np.save(OUT_DIR / f"indices_for_reshape_2d_sub_dom_{subdom_idx}.npy", input_indices)

if store_tag == "crosspoint_1d_coeff":
    np.save(OUT_DIR / f"input_paras_num_samples_{num_samples}.npy", parameters[:,0])
else:
    np.save(OUT_DIR / f"input_paras_num_samples_{num_samples}.npy", parameters)
        
print('FNO data generation completed!')
ray.shutdown()

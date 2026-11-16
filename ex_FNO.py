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
import setup
import datetime

# Setup argparse to accept command line arguments
parser = argparse.ArgumentParser(description="Generate data for MSGFEM")
parser.add_argument("--store_tag", type=str, required=True, help="Varying parameter tag")
parser.add_argument("--num_samples", type=int, default=1200, help="Number of samples to generate")
parser.add_argument("--nloc", type=int, default=5, help="Number of local basis functions")
parser.add_argument("--subdom_idx_list", type=int, nargs="+", default=[5, 6, 9, 10],
                    help="Subdomain indices whose eigenproblems are solved in parallel per sample")
parser.add_argument("--param_indices", type=int, nargs="+", default=None,
                    help="Rerun only these sample indices, taking the parameters from the "
                         "stored input_paras file instead of drawing new ones")

args = parser.parse_args()
# Set store_tag from the command line argument
store_tag = args.store_tag
num_samples = args.num_samples
nloc = args.nloc 
param_indices = args.param_indices

# Define the absolute path to your target scratch directory
BASE_DIR = Path("/fs/scratch/rb_bd_dlp_rng_dl01_cr_MSO_employees/students/nuj7rng/msgfem_data/fenics_out_data")
IN_DIR = BASE_DIR / store_tag / f"samples_{num_samples}"
OUT_DIR = (
    BASE_DIR / store_tag / f"samples_{num_samples}_from_parameter"
    if param_indices is not None
    else IN_DIR
)
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

np.random.seed(42)
parameters = setup.sample_FNO_parameters(store_tag, num_samples)

if param_indices is None:
    sample_indices = list(range(len(parameters)))
else:
    param_path = IN_DIR / f"input_paras_num_samples_{num_samples}_nloc_{nloc}.npy"
    if not param_path.exists():
        raise FileNotFoundError(f"No stored parameters at {param_path}")
    stored_parameters = np.load(param_path)
    out_of_range = [i for i in param_indices if not 0 <= i < stored_parameters.shape[0]]
    if out_of_range:
        raise ValueError(
            f"param_indices {out_of_range} out of range for {stored_parameters.shape[0]} "
            f"stored samples in {param_path}"
        )
    sample_indices = list(param_indices)
    parameters = stored_parameters[sample_indices]
    print(f"Loaded {len(sample_indices)} parameter samples from {param_path}")

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

    np.save(OUT_DIR / f"phi_sub_dom_{subdom_idx}_sample_{sample_idx}_nloc_{nloc}.npy", basis_funs)
    np.save(OUT_DIR / f"coeff_A_sub_dom_{subdom_idx}_sample_{sample_idx}_nloc_{nloc}.npy", coeff_A_FNO)
    np.save(OUT_DIR / f"eig_vals_sub_dom_{subdom_idx}_sample_{sample_idx}_nloc_{nloc}.npy", eig_vals)
    np.save(OUT_DIR / f"vecs_tmp_sub_dom_{subdom_idx}_sample_{sample_idx}_nloc_{nloc}.npy", vecs_tmp)
    np.save(OUT_DIR / f"indices_for_reshape_2d_sub_dom_{subdom_idx}_nloc_{nloc}.npy", input_indices)

    return sample_idx, subdom_idx, eig_solve_time


# Dispatch one Ray task per (sample, subdomain) combination. For a fixed sample
# all subdomains in subdom_idx_list are solved in parallel; submitting every
# combination up front lets Ray keep all allocated cores busy across samples.
futures = [
    generate_sample_subdom.remote(idx, parameter, subdom_idx)
    for idx, parameter in zip(sample_indices, parameters)
    for subdom_idx in subdom_idx_list
]

# Collect eigsh solve times per subdomain as the tasks complete.
eig_solve_times = {
    subdom_idx: np.full(max(sample_indices) + 1, np.nan) for subdom_idx in subdom_idx_list
}
remaining = futures
while remaining:
    done, remaining = ray.wait(remaining, num_returns=1)
    sample_idx, subdom_idx, eig_solve_time = ray.get(done[0])
    eig_solve_times[subdom_idx][sample_idx] = eig_solve_time
    print(f"Completed sample {sample_idx}, subdomain {subdom_idx} "
          f"(eigsh time {eig_solve_time:.4f} s)", flush=True)

for subdom_idx in subdom_idx_list:
    np.save(OUT_DIR / f"eig_solve_times_sub_dom_{subdom_idx}_nloc_{nloc}.npy",
            eig_solve_times[subdom_idx])

if param_indices is not None:
    np.save(OUT_DIR / f"sample_indices_nloc_{nloc}.npy", np.asarray(sample_indices))
    np.save(OUT_DIR / f"input_paras_selected_nloc_{nloc}.npy", parameters)
elif store_tag == "crosspoint_1d_coeff":
    np.save(OUT_DIR / f"input_paras_num_samples_{num_samples}_nloc_{nloc}.npy", parameters[:,0])
else:
    np.save(OUT_DIR / f"input_paras_num_samples_{num_samples}_nloc_{nloc}.npy", parameters)
        
print('FNO data generation completed!')
ray.shutdown()

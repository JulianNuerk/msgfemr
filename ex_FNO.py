"""
Generates Data and uses FNOs to solve the local Eigenvalue problems in the MSGFEM method.
"""

import experiments_driver as ed

import ray
import numpy as np
# ray.init(num_cpus=1)  # uncomment if want to run serially
import helper
import datetime
import os as ops

# Generate a timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
np.random.seed(42)

commit_hash = helper.get_commit_hash()

# parameters to play with
deg = 1
Ny = 4
ny = 2**8
ol = 2
os = 2
nloc = 6
rho = 0.0
bool_ring = False
exclude_last_basis_fun = True # if true stores only the last n-1 basis functions to get rid  of the constant one.
subdom_idx = 5
store_tag = 'channel_low_coeff'
num_samples = 10500

# parameters not to play with 
x0 = 0.234375
y1 = 0.515625
eps = 1e-08
p0_bound = 9/10*(y1-x0) - eps # guarantees that channel stays in subdomain 
p1_bound = 1/2*(y1-x0) - eps # guarantees that channel stays in subdomain 

parameters = np.random.uniform(low=[eps,eps,1], high=[p0_bound, p1_bound, 10], size=(num_samples, 3))
#parameters = np.random.uniform(low=[5,10], high=[10, 20], size=(num_samples, 2))
#parameters = [1,1,1,1, 5,5,5,5] #np.linspace(1,5,num_samples)
#p2 = np.linspace(20,30,num_samples)
#parameters = np.array([p1,p2]).T
#parameters = [1, 100, 1000, 10000, 100000, 200000, 300000, 600000, 800000, 1000000] # np.linspace(1, 1e+06, ) np.random.uniform(low=0, high=1e+6, size=num_samples)

for idx, parameter in enumerate(parameters):
    print('Sample Nr.:', idx)
    print('Parameters: %s.'%(parameter))
    coeff_A_FNO, basis_funs, eig_vals = ed.run_fno_data_generation(deg, Ny, ny, ol, os, nloc, rho, bool_ring, parameter, subdom_idx)

    # store data
    path_ = "data_FNO/%s"%(store_tag)
    if not ops.path.exists(path_):
        ops.makedirs(path_)
    if exclude_last_basis_fun:
        basis_funs = basis_funs[..., :-1]

    np.save(ops.path.join(path_, "phi_sub_dom_%s_sample_%s.npy"%(subdom_idx, idx)), basis_funs)
    np.save(ops.path.join(path_, "coeff_A_sub_dom_%s_sample_%s.npy"%(subdom_idx, idx)), coeff_A_FNO)
    np.save(ops.path.join(path_, "eig_vals_sub_dom_%s_sample_%s.npy"%(subdom_idx, idx)), eig_vals)

        
print('FNO data generation completed!')
ray.shutdown()





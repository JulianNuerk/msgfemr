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
exclude_last_basis_fun = True # if true stores only the last n-1 basis functions to get rid  of the constant one.
subdom_idx = 5
store_tag = 'channel_smooth_coeff' # channel_smooth_coeff, sinus_coeff, channel_coeff
num_samples = 10500

# parameters not to play with 
x0 = 0.234375 # lower right corner of subdomain
y1 = 0.515625 # upper left corner 
eps = 1e-08

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

else:
    raise ValueError('store_tag %s not defined!'%(store_tag))

print('Starting data generation for case: %s'%(store_tag))
for idx, parameter in enumerate(parameters):
    print('Sample Nr.:', idx)
    print('Parameters: %s.'%(parameter))
    coeff_A_FNO, basis_funs, eig_vals = ed.run_fno_data_generation(deg, Ny, ny, ol, os, nloc, rho, store_tag, parameter, subdom_idx)

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





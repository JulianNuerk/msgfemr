import experiments_driver as ed
import ray
import numpy as np
# ray.init(num_cpus=1)  # uncomment if want to run serially
import helper
import datetime
import matplotlib.pyplot as plt

import os as ops
# Generate a timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

commit_hash = helper.get_commit_hash()

# Skyscraper experiment
k = 0
deg = 1
Ny = 8
ny = 800
ol = 2
oss = np.arange(2, 7, 2)
nlocs = np.array([10, 20, 30, 40, 50, 60])
rho = 0.0
problem_label = "skyscraper" # other options : "source_dirichlet", "iid", "channel", "blind_ring", "skyscraper"
contrast = 1 # Not relevant for skyscraper problem

iteration_numbers = np.zeros((len(nlocs), len(oss)))
gfem_errors = np.zeros((len(nlocs), len(oss)))
coarse_space_size = np.zeros((len(nlocs), len(oss)))

iteration_numbers_ring = np.zeros((len(nlocs), len(oss)))
gfem_errors_ring = np.zeros((len(nlocs), len(oss)))
coarse_space_size_ring = np.zeros((len(nlocs), len(oss)))

for nloc_id in range(len(nlocs)):
    for os_id in range(len(oss)):
        nloc = nlocs[nloc_id]
        os = oss[os_id]
        print(f"nloc: {nloc}, os: {os}")

        bool_ring = False
        
        iteration_numbers[nloc_id, os_id], gfem_errors[nloc_id, os_id], coarse_space_size[nloc_id, os_id] = ed.run_msgfem(deg, Ny, ny, ol, os, nloc, rho, problem_label, bool_ring, os)

        bool_ring = True

        iteration_numbers_ring[nloc_id, os_id], gfem_errors_ring[nloc_id, os_id], coarse_space_size_ring[nloc_id, os_id] = ed.run_msgfem(deg, Ny, ny, ol, os, nloc, rho, problem_label, bool_ring, os)


        # If directory does not exist, create it
        if not ops.path.exists("data_nloc"):
            ops.makedirs("data_nloc")
        np.savez("data_nloc/" + problem_label + timestamp + ".npz",
                        deg=deg,
                        ny=ny,
                        Ny = Ny,
                        ol = ol,
                        oss = oss,
                        nlocs = nlocs,
                        iteration_numbers = iteration_numbers,
                        iteration_numbers_ring = iteration_numbers_ring,
                        gfem_errors = gfem_errors,
                        gfem_errors_ring = gfem_errors_ring,
                        rho = rho,
                        coarse_space_size = coarse_space_size,
                        coarse_space_size_ring = coarse_space_size_ring,
                        commit_hash = commit_hash
                    )
        

ray.shutdown()

filename = problem_label + timestamp + ".npz"
print("**filename: "+ filename+ "**")
data = np.load("data_nloc/"+ filename, allow_pickle = False)
filename = filename[:-4]

oss = data["oss"].astype(int)
nlocs = data["nlocs"]
iteration_numbers = data["iteration_numbers"].astype(int).T
gfem_errors = data["gfem_errors"]
iteration_numbers_rings = data["iteration_numbers_ring"].astype(int).T
gfem_errors_rings = data["gfem_errors_ring"]

# Get the colormap
colors = ["#0072BD", "#D95319", "#9746A7", "#77AC30", "#4DBEEE", "#A2142F", "#77AC30", "#4DBEEE", "#A2142F"]

# Create a figure and axis with appropriate size for readability
plt.figure(figsize=(8, 6))

# Loop through nlocs and plot the data with enhancements
for os_id in range(len(oss)):
    # color = jet(os_id / len(oss))  # Get a color from the colormap
    plt.semilogy(
        nlocs, 
        gfem_errors[:,os_id], 
        label=fr"$\omega^{{\ast}}, \ell={oss[os_id]}$", 
        marker="o", 
        color=colors[os_id], 
        linestyle='-',
        linewidth=2.5
    )
    plt.semilogy(
        nlocs, 
        gfem_errors_rings[:, os_id], 
        label=fr"$R^{{\ast}}, \ell ={oss[os_id]}$", 
        marker="x", 
        color=colors[os_id], 
        linestyle='--',
        linewidth=2.5
    )

# Add labels and title
plt.xlabel(r"$n$", fontsize=16)
plt.ylabel(r"$\mathbf{err}$", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Add grid for better readability
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

# Add legend outside the plot for better clarity
plt.legend(loc='lower left', fontsize=14)

# Adjust layout for tight fit
plt.tight_layout(rect=[0, 0, 0.85, 1])

# Save figure with high resolution for publication
if not ops.path.exists("plots/"):
    ops.makedirs("plots/")
output_path = f"plots/{filename}.pdf"
plt.savefig(output_path, dpi=600, bbox_inches="tight")

# Display the plot
plt.show()



# Generate a timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

commit_hash = helper.get_commit_hash()

# Skyscraper experiment
k = 0
deg = 1
Ny = 8
ny = 800
ol = 2
oss = np.arange(4, 33, 4)
nlocs = np.array([10, 20, 30])
rho = 0.0
problem_label = "skyscraper" # other options : "source_dirichlet", "iid", "channel", "blind_ring", "skyscraper"
contrast = 1 # Not relevant for skyscraper experiment

iteration_numbers = np.zeros((len(nlocs), len(oss)))
gfem_errors = np.zeros((len(nlocs), len(oss)))
coarse_space_size = np.zeros((len(nlocs), len(oss)))

iteration_numbers_ring = np.zeros((len(nlocs), len(oss)))
gfem_errors_ring = np.zeros((len(nlocs), len(oss)))
coarse_space_size_ring = np.zeros((len(nlocs), len(oss)))

for nloc_id in range(len(nlocs)):
    for os_id in range(len(oss)):
        nloc = nlocs[nloc_id]
        os = oss[os_id]
        print(f"nloc: {nloc}, os: {os}")

        bool_ring = False
        
        iteration_numbers[nloc_id, os_id], gfem_errors[nloc_id, os_id], coarse_space_size[nloc_id, os_id] = ed.run_msgfem(deg, Ny, ny, ol, os, nloc, rho, problem_label, bool_ring, os)

        bool_ring = True

        iteration_numbers_ring[nloc_id, os_id], gfem_errors_ring[nloc_id, os_id], coarse_space_size_ring[nloc_id, os_id] = ed.run_msgfem(deg, Ny, ny, ol, os, nloc, rho, problem_label, bool_ring, os)

        # If directory does not exist, create it
        if not ops.path.exists("data_oversampling"):
            ops.makedirs("data_oversampling")


        np.savez("data_oversampling/" + problem_label + timestamp + ".npz",
                        deg=deg,
                        ny=ny,
                        Ny = Ny,
                        ol = ol,
                        oss = oss,
                        nlocs = nlocs,
                        iteration_numbers = iteration_numbers,
                        iteration_numbers_ring = iteration_numbers_ring,
                        gfem_errors = gfem_errors,
                        gfem_errors_ring = gfem_errors_ring,
                        rho = rho,
                        coarse_space_size = coarse_space_size,
                        coarse_space_size_ring = coarse_space_size_ring,
                        commit_hash = commit_hash
                    )

ray.shutdown()


# Create plot
filename = problem_label + timestamp + ".npz"
print("**filename: "+ filename+ "**")
data = np.load("data_oversampling/"+ filename, allow_pickle = False)
filename = filename[:-4]

oss = data["oss"].astype(int)
# oss = np.array([1, 10, 100, 1000, 10000])
nlocs = data["nlocs"]
iteration_numbers = data["iteration_numbers"].astype(int).T
gfem_errors = data["gfem_errors"]
iteration_numbers_rings = data["iteration_numbers_ring"].astype(int).T
gfem_errors_rings = data["gfem_errors_ring"]

colors = ["#0072BD", "#D95319", "#7E2F8E", "#77AC30", "#4DBEEE", "#A2142F", "#77AC30", "#4DBEEE", "#A2142F"]

# Create a figure and axis with appropriate size for readability
plt.figure(figsize=(8, 6))

# Loop through nlocs and plot the data with enhancements
for nloc_id in range(len(nlocs)):
    plt.semilogy(
        oss, 
        gfem_errors[nloc_id, :], 
        label=fr"$\omega^{{\ast}}, n={nlocs[nloc_id]}$", 
        marker="o", 
        color=colors[nloc_id], 
        linestyle='-',
        linewidth=2.5
    )
    plt.semilogy(
        oss, 
        gfem_errors_rings[nloc_id, :], 
        label=fr"$R^{{\ast}}, n ={nlocs[nloc_id]}$", 
        marker="x", 
        color=colors[nloc_id], 
        linestyle='--',
        linewidth=2.5
    )

# Add labels and title
plt.xlabel(r"$\ell$", fontsize=16)
plt.ylabel(r"$\mathbf{err}$", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Add grid for better readability
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

# Add legend outside the plot for better clarity
plt.legend(loc='lower left', fontsize=14)

# Adjust layout for tight fit
plt.tight_layout(rect=[0, 0, 0.85, 1])

# Save figure with high resolution for publication
if not ops.path.exists("plots/"):
    ops.makedirs("plots/")
output_path = f"plots/{filename}.pdf"
plt.savefig(output_path, dpi=600, bbox_inches="tight")

# Display the plot
plt.show()



"""
eigenproblem_driver.py

This script benchmarks the performance of the eigenproblem solver for 3D subdomains, 
specifically generating the data for Figure 7.4 (left) of the paper. 
"""

import eigenproblem3d as eig

import numpy as np
import helper
import datetime
import matplotlib.pyplot as plt
import os as ops

commit_hash = helper.get_commit_hash()
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

nlocs = np.arange(5,16,5) # Number of eigenvectors per subdomain
nparts = np.arange(10, 26, 5) # Corresponds to m in the paper
ol = 1 # Overlap parameter
oss = np.arange(1,4) # Oversampling parameter

timings = np.zeros((len(nlocs), len(nparts), len(oss))) # timings of eigenproblem
timings_harmonic_extension = np.zeros((len(nlocs), len(nparts), len(oss)))
sizes = np.zeros((len(nlocs), len(nparts), len(oss)))

timings_ring = np.zeros((len(nlocs), len(nparts), len(oss)))
timings_harmonic_extension_ring = np.zeros((len(nlocs), len(nparts), len(oss)))
sizes_ring = np.zeros((len(nlocs), len(nparts), len(oss)))

for nloc_id in range(len(nlocs)):
    nloc = nlocs[nloc_id]
    for n_part_id in range(len(nparts)):
        n_part = nparts[n_part_id]
        for os_id in range(len(oss)):
            os = oss[os_id]
            timings[nloc_id, n_part_id, os_id], sizes[nloc_id, n_part_id, os_id], timings_harmonic_extension[nloc_id, n_part_id, os_id] = eig.computeSubdomain(n_part, nloc, False, ol, os)
            timings_ring[nloc_id, n_part_id, os_id], sizes_ring[nloc_id, n_part_id, os_id], timings_harmonic_extension_ring[nloc_id, n_part_id, os_id] = eig.computeSubdomain(n_part, nloc, True, ol, os)
            
            # If directory does not exist, create it
            if not ops.path.exists("data_3d"):
                ops.makedirs("data_3d")
            np.savez("data_3d/eigenproblem_driver_" + timestamp + ".npz", timings=timings, timings_harmonic_extension = timings_harmonic_extension, sizes=sizes, timings_ring=timings_ring, timings_harmonic_extension_ring = timings_harmonic_extension_ring, sizes_ring=sizes_ring, nlocs=nlocs, nparts=nparts, oss = oss, ol = ol, commit_hash=commit_hash)


# Create plot
filename = "eigenproblem_driver_" + timestamp + ".npz"
data = np.load("data_3d/"+ filename, allow_pickle = False)

plt.figure(figsize=(8, 6))
colors = ["#0072BD", "#D95319", "#7E2F8E", "#77AC30", "#4DBEEE", "#A2142F", "#77AC30", "#4DBEEE", "#A2142F"]

id_nloc = 0     # Change this to select the nloc index you want to plot

for os_id in range(data["oss"].shape[0]):
    # Plot the data with enhancements
    plt.semilogy(
        data["nparts"], 
        data["timings"][id_nloc, :, os_id], 
        label=r"$\omega^*, \ell = $" + str(data["oss"][os_id]), 
        marker="o", 
        linestyle="-", 
        linewidth=2.5, 
        color=colors[os_id]
    )
    plt.semilogy(
        data["nparts"], 
        data["timings_ring"][id_nloc, :, os_id] + data["timings_harmonic_extension_ring"][id_nloc, :, os_id], 
        label=r"$R^*, \ell = $" + str(data["oss"][os_id]),  
        marker="x", 
        linestyle="--", 
        linewidth=2.5, 
        color=colors[os_id]
    )

# Add labels, title, and legend
plt.xlabel(r"$m$", fontsize=16)
plt.ylabel("Time [s]", fontsize=16)

# Customize tick font size
plt.xticks(data["nparts"], fontsize=14)
plt.yticks([1, 1e1, 1e2, 1e3], fontsize=14)

# Add grid for better readability
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

# Add legend outside the plot for better clarity
plt.legend(loc='upper left', fontsize=12)

# Adjust layout for tight fit
plt.tight_layout(rect=[0, 0, 0.85, 1])

# Save figure with high resolution 
if not ops.path.exists("plots/"):
    ops.makedirs("plots/")
output_path = f"plots/{filename[:-4]}.pdf"
plt.savefig(output_path, dpi=600, bbox_inches="tight")

# Display the plot
plt.show()


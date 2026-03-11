"""
Generates and saves the plot for average number of nonzeros per row in subdomain matrices, as shown in Fig. 7.4 of the paper.
"""

import eigenproblem3d as eig
import numpy as np
import helper
import datetime
import matplotlib.pyplot as plt
import os as ops

commit_hash = helper.get_commit_hash()
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

nparts = np.arange(10, 26, 5) # Corresponds to m in the paper
ol = 1 # Overlap parameter
oss = np.arange(1,4) # Oversampling parameter

nnz_superlu = np.zeros((len(nparts), len(oss)))
nnz_MA = np.zeros((len(nparts), len(oss)))
sizes = np.zeros((len(nparts), len(oss)))

nnz_superlu_ring = np.zeros((len(nparts), len(oss)))
nnz_MA_ring = np.zeros((len(nparts), len(oss)))
sizes_ring = np.zeros((len(nparts), len(oss)))

nloc = 5  # Irrelevant for this experiment, but must be set
for n_part_id in range(len(nparts)):
    n_part = nparts[n_part_id]
    for os_id in range(len(oss)):
        os = oss[os_id]
        print("n_part: ", n_part, "os: ", os)
        print("On omega:")
        nnz_superlu[n_part_id, os_id], nnz_MA[n_part_id, os_id], sizes[n_part_id, os_id] = eig.computeSubdomain(n_part, nloc, False, ol, os, nnz_computation=True)
        print("On ring:")
        nnz_superlu_ring[n_part_id, os_id], nnz_MA_ring[n_part_id, os_id], sizes_ring[n_part_id, os_id]  = eig.computeSubdomain(n_part, nloc, True, ol, os, nnz_computation=True)
        
        # If directory does not exist, create it
        if not ops.path.exists("data_3d"):
            ops.makedirs("data_3d")
        
        np.savez("data_3d/eigenproblem_nnz_driver_" + timestamp + ".npz", nnz_superlu = nnz_superlu, nnz_MA = nnz_MA, sizes=sizes, nnz_superlu_ring = nnz_superlu_ring, nnz_MA_ring = nnz_MA_ring, sizes_ring=sizes_ring, nloc=nloc, nparts=nparts, oss = oss, ol = ol, commit_hash=commit_hash)


filename = "eigenproblem_nnz_driver_" + timestamp + ".npz"
data = np.load("data_3d/"+ filename, allow_pickle = False)

# Create a figure and axis with appropriate size for readability
plt.figure(figsize=(8, 6))
colors = ["#0072BD", "#D95319", "#7E2F8E", "#77AC30", "#4DBEEE", "#A2142F", "#77AC30", "#4DBEEE", "#A2142F"]

for os_id in range(data["oss"].shape[0]):
    # Plot the data
    plt.plot(
        data["nparts"], 
        data["nnz_superlu"][:, os_id] / data["sizes"][:,os_id],
        label=r"$\omega^*, \ell = $" + str(data["oss"][os_id]), 
        marker="o", 
        linestyle="-", 
        linewidth=2.5, 
        color=colors[os_id]
    )
    plt.plot(
        data["nparts"], 
        data["nnz_superlu_ring"][:, os_id]/ data["sizes_ring"][:,os_id],
        label=r"$R^*, \ell = $" + str(data["oss"][os_id]),  
        marker="x", 
        linestyle="--", 
        linewidth=2.5, 
        color=colors[os_id]
    )

# Add labels, title, and legend
plt.xlabel(r"$m $", fontsize=16)
plt.ylabel("average number of nonzeros per row", fontsize=16)
plt.yticks([0,1000, 2000, 3000, 4000, 5000], fontsize=14)
plt.xticks([10, 15, 20, 25], fontsize=14)

# Add grid for better readability
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

# Add legend outside the plot for better clarity
plt.legend(loc='upper left', fontsize=12)

# Adjust layout for tight fit
plt.tight_layout(rect=[0, 0, 0.85, 1])

# # Save figure with high resolution for publication
if not ops.path.exists("plots/"):
    ops.makedirs("plots/")
output_path = f"plots/{filename[:-4]}.pdf"
plt.savefig(output_path, dpi=600, bbox_inches="tight")

# Display the plot
plt.show()
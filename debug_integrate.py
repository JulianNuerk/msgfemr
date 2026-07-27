"""
debug_integrate.py
==================

Minimal smoke-test driver for ``integrate_predictions.compute_overall_errors``.

Runs the full pipeline for a single sample with a hand-crafted
``predictions_data`` dictionary so that the code path can be stepped through
with ``pdb`` or the VS Code debugger without needing a ``.npz`` prediction
file produced by ``ex_FNO.py``.

Usage
-----
    python debug_integrate.py                # normal run
    python -m pdb debug_integrate.py         # step through in pdb
    # or press F5 in VS Code with a "Python File" launch config.
"""

import numpy as np

import integrate_predictions as ip


# ---------------------------------------------------------------------------
# MS-GFEM configuration (must match integrate_predictions defaults so that the
# derived local patch size below is correct).
# ---------------------------------------------------------------------------
Ny = 4          # coarse mesh in y (Nx == Ny on the unit square)
ny = 2 ** 8     # fine mesh in y
ol = 2          # overlap
os_ = 2         # oversampling
nloc = 5        # local basis size k

Nx = Ny
nx = ny

sub_ = os_ + ol

# Interior oversampling patch: local cells per subdomain plus (os + ol) padding
# on every side.  For an interior subdomain this matches the shape emitted by
# the FNO prediction pipeline in ex_FNO.py (see the "e.g. 73, 73" comment in
# msgfem_parallel.computeSubdomain).
local_cells = ny // Ny                     # 64 for the defaults above
full_cells = local_cells + 2 * (os_ + ol)  # 72
full_nodes = full_cells + 1                # 73

# ---------------------------------------------------------------------------
# One parameter sample for the "channel_coeff" family.
# setup.channel expects a flat vector of length 3 * 16 = 48 per sample, with
# entries (channel_x_offset, channel_y_offset, height) for each of the 16
# subdomains of the regular 4x4 partition of [0, 1] x [0, 1].
# ---------------------------------------------------------------------------
num_samples = 3
N_sub = 4
side = 1.0 / N_sub
width_channel = side / 10.0
high_channel = side / 2.0
# Place a channel roughly in the middle of every subdomain with unit height.
per_subdomain = np.array(
    [(side - width_channel) / 2.0, (side - high_channel) / 2.0, 1.0]
)
parameters = np.tile(per_subdomain, (num_samples, N_sub * N_sub))  # shape (num_samples, 48)

# ---------------------------------------------------------------------------
# Fake per-subdomain predictions.  Pick an interior subdomain (index 5 in a
# 4x4 partition) so that no edge/corner padding trimming happens; zeros are a
# valid smoke-test input (they produce a degenerate basis but exercise the
# full code path).
# ---------------------------------------------------------------------------
predictions_data = {
    0: np.ones((num_samples, nloc, 73, 73), dtype=np.float64),
    1: np.ones((num_samples, nloc, 73, 73), dtype=np.float64),
    3: np.ones((num_samples, nloc, 73, 73), dtype=np.float64),
    4: np.ones((num_samples, nloc, 73, 73), dtype=np.float64),
    5: np.ones((num_samples, nloc, 73, 73), dtype=np.float64),
    7: np.ones((num_samples, nloc, 73, 73), dtype=np.float64),
    13: np.ones((num_samples, nloc, 73, 73), dtype=np.float64),
    14: np.ones((num_samples, nloc, 73, 73), dtype=np.float64),
}


def main():
    result = ip.compute_overall_errors(
        store_tag="channel_coeff",
        parameters=parameters,
        predictions_data=predictions_data,
        deg=1,
        Ny=Ny,
        ny=ny,
        ol=ol,
        os_=os_,
        nloc=nloc,
        rho=0.0,
        bool_ring=False,
        plot=False,   # skip XDMF output while debugging
    )
    print("\n=== compute_overall_errors result ===")
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()

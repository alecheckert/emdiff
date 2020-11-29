#!/usr/bin/env python
"""
sample_script.py -- Example usage of vbdiff

"""
import pandas as pd
from emdiff import vbdiff

if __name__ == "__main__":

    # Load some tracks
    tracks = pd.read_csv("sample_tracks.csv")

    # Run emdiff
    occs, diff_coefs = vbdiff(
        tracks,
        n_states=5,             # number of diffusive states
        pixel_size_um=0.16,     # microns
        frame_interval=0.00748, # seconds
        loc_error=0.035,        # microns
        dz=0.7,                 # focal depth, microns
    )

    # State the result
    for state in range(len(occs)):
        print("Diffusive state {}:".format(state))
        print("\tDiffusion coefficient {:.4f} um^2 s^-1".format(diff_coefs[state]))
        print("\tOccupancy {:.4}%".format(occs[state] * 100))

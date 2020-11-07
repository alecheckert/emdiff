#!/usr/bin/env python
"""
em.py -- core expectation-maximization routine

"""
from copy import copy
import numpy as np
import pandas as pd
from .defoc import f_remain
from .utils import (
    sum_squared_jumps
)

INIT_DIFF_COEFS = {
    1: np.array([1.0]),
    2: np.array([0.01, 2.0]),
    3: np.array([0.01, 0.5, 5.0]),
    4: np.array([0.01, 0.5, 1.0, 5.0])
}

def emdiff(tracks, n_states=2, pixel_size_um=0.16, frame_interval=0.00748,
    pos_cols=["y", "x"], loc_error=0.035, max_jumps_per_track=None,
    start_frame=0, max_iter=1000, convergence=1.0e-8, dz=np.inf):
    """
    Estimate the occupations and diffusion coefficients for a Brownian
    mixture model using an expectation-maximization routine.

    args
    ----
        tracks          :   pandas.DataFrame with columns "trajectory",
                            "frame", and the contents of *pos_cols*
        n_states        :   int, the number of diffusive states
        pixel_size_um   :   float, the size of pixels in um
        frame_interval  :   float, frame interval in seconds
        pos_cols        :   list of str, the position columns in *tracks*
        loc_error       :   float, the localization error in um
        max_jumps_per_track:    int, the maximum number of jumps to consider
                                from each trajectory
        start_frame     :   int, only include trajectories before this
                            frame
        max_iter        :   int, the maximum number of iterations to do 
                            before stopping
        convergence     :   float, convergence criterion for the occupation
                            estimate
        dz              :   float, focal depth in um

    returns
    -------
        (
            1D ndarray of shape (n_states,), the occupations;
            1D ndarray of shape (n_states,), the corresponding 
                diffusion coefficients in um^2 s^-1
        )

    """
    # Localization variance
    le2 = loc_error ** 2

    # Apparent diffusion coefficient of a completely immobile
    # object due to localization error
    d_err = le2 / frame_interval

    # Only take points after the start frame
    if start_frame > 0:
        tracks = tracks[tracks["frame"] >= start_frame]

    # Calculate the sum of squared displacements for every 
    # trajectory in this dataset
    L = sum_squared_jumps(
        tracks,
        pixel_size_um=pixel_size_um,
        pos_cols=pos_cols,
        max_jumps_per_track=max_jumps_per_track
    )
    sum_r2 = np.asarray(L["sum_r2"])
    n_jumps = np.asarray(L["n_jumps"]) * len(pos_cols) / 2.0
    n_tracks = L["trajectory"].nunique()

    # Initial state occupations
    occs = np.ones(n_states, dtype=np.float64) / n_states

    # Initial diffusion coefficients
    diff_coefs = copy(INIT_DIFF_COEFS[n_states])

    # Likelihoods of each state, given each trajectory
    T = np.zeros((n_states, n_tracks), dtype=np.float64)

    # Previous iteration's estimate, to check for convergence
    prev_occs = occs.copy()

    # Iterate until convergence or *max_iter* is reached
    for iter_idx in range(max_iter):

        # Determine the log likelihoods of each trajectory under
        # the present model
        for j, d in enumerate(diff_coefs):
            var2 = 4 * (d * frame_interval + le2)
            T[j,:] = -sum_r2 / var2 - n_jumps * np.log(var2)
        T = T - T.max(axis=0)
        T = np.exp(T)

        # Scale to current state occupation estimate
        T = (T.T * occs).T

        # Normalize over diffusion coefficients for each
        # trajectory
        T = T / T.sum(axis=0)

        # Calculate the new vector of diffusion coefficients
        diff_coefs = ((T @ sum_r2) / (T @ n_jumps)) / \
            (2 * len(pos_cols) * frame_interval)
        diff_coefs -= d_err

        # Calculate the new state occupation vector
        occs = (T * n_jumps).sum(axis=1)
        occs /= occs.sum()

        # Check for convergence
        if (np.abs(occs - prev_occs) <= convergence).all():
            break
        else:
            prev_occs[:] = occs[:]

    # Correct for defocalization
    if (not dz is None) and (not dz is np.inf):
        corr = np.zeros(n_states)
        for i, D in enumerate(diff_coefs):
            corr[i] = f_remain(D, 1, frame_interval, dz)[0]
        occs /= corr
        occs /= occs.sum()

    return occs, diff_coefs

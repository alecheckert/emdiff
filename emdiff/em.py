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
from .plot import plot_jump_length_dist, spatial_dist

INIT_DIFF_COEFS = {
    1: np.array([1.0]),
    2: np.array([0.01, 2.0]),
    3: np.array([0.01, 0.5, 5.0]),
    4: np.array([0.01, 0.5, 1.0, 5.0])
}

def emdiff(tracks, n_states=2, pixel_size_um=0.16, frame_interval=0.00748,
    pos_cols=["y", "x"], loc_error=0.035, max_jumps_per_track=None,
    start_frame=0, max_iter=1000, convergence=1.0e-8, dz=np.inf,
    plot=False, plot_prefix="emdiff_default_out"):
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
        plot            :   bool, make plots of the result
        plot_prefix     :   str, prefix for output plots

    returns
    -------
        (
            1D ndarray of shape (n_states,), the occupations;
            1D ndarray of shape (n_states,), the corresponding 
                diffusion coefficients in um^2 s^-1;
            pandas.DataFrame, the original trajectories with extra
                columns corresponding to the likelihoods of each 
                final diffusive state, given the corresponding
                trajectory. Trajectories with only a single point
                (``singlets'') are assigned NaN in these columns.
        )

    """
    # Check for incompatible inputs
    check_input(tracks, start_frame=start_frame, pos_cols=pos_cols,
        max_jumps_per_track=max_jumps_per_track)

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

    # Make sure we actually have jumps to work with
    assert not L.empty, "emdiff.em: no jumps found in dataset"

    # Sum of squared radial displacements
    sum_r2 = np.asarray(L["sum_r2"])

    # Number of jumps corresponding to each trajectory (1D ndarray)
    n_jumps = np.asarray(L["n_jumps"]) * len(pos_cols) / 2.0

    # Total number of trajectories in the dataset (int)
    n_tracks = L["trajectory"].nunique()

    # Initial state occupations
    occs = np.ones(n_states, dtype=np.float64) / n_states

    # Initial diffusion coefficients
    diff_coefs = copy(INIT_DIFF_COEFS[n_states])

    # Likelihoods of each state, given each trajectory
    T = np.zeros((n_states, n_tracks), dtype=np.float64)

    # Previous iteration's estimate, to check for convergence
    prev_occs = occs.copy()
    prev_diff_coefs = diff_coefs.copy()

    # Defocalization correction
    correct_defoc = (not dz is None) and (not dz is np.inf)
    corr = np.ones(n_states, dtype=np.float64)

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

        # Weight each trajectory by the number of jumps
        T = T * n_jumps

        # Calculate the new vector of diffusion coefficients
        diff_coefs = ((T @ sum_r2) / (T @ n_jumps)) / \
            (2 * len(pos_cols) * frame_interval)
        diff_coefs -= d_err

        # Naive way to guard against negative diffusion coefficients
        diff_coefs = np.maximum(diff_coefs, 1.0e-8 * np.ones(n_states))

        # Calculate the new state occupation vector
        occs = T.sum(axis=1)

        # Correct for defocalization
        if correct_defoc:
            for i, D in enumerate(diff_coefs):
                corr[i] = f_remain(D, 1, frame_interval, dz)[0]
            occs = occs / corr

        # Normalize
        occs /= occs.sum()

        # Check for convergence
        if (np.abs(occs - prev_occs) <= convergence).all() and \
            (np.abs(diff_coefs - prev_diff_coefs) <= convergence).all():
            break
        else:
            prev_occs[:] = occs[:]
            prev_diff_coefs[:] = diff_coefs[:]

    # Order the diffusion coefficients so that they're strictly increasing
    order = np.argsort(diff_coefs)
    occs = occs[order]
    diff_coefs = diff_coefs[order]
    T = T[order, :]

    # Map the state likelihoods back to the origin trajectories, if 
    # desired
    T = T / T.sum(axis=0)
    L = L.set_index("trajectory")
    for j in range(n_states):
        c = "likelihood_state_%d" % j 
        L[c] = T[j,:]
        tracks[c] = tracks["trajectory"].map(L[c])

    # Plot the result, if desired
    if plot:
        use_entire_track = (not max_jumps_per_track is None) and \
            (not max_jumps_per_track is np.inf)

        # Plot jump length histograms
        plot_jump_length_dist(tracks, occs, diff_coefs, plot_prefix, 
            pos_cols=pos_cols, n_frames=4, frame_interval=frame_interval,
            pixel_size_um=pixel_size_um, loc_error=loc_error, dz=dz,
            max_jump=5.0, max_jump_pmf=2.0, cmap="gray", figsize_mod=1.0,
            n_gaps=0, use_entire_track=use_entire_track,
            max_jumps_per_track=max_jumps_per_track)

        # Plot the spatial distribution
        spatial_dist_png = "{}_spatial_dist.png".format(plot_prefix)
        attrib_cols = ["likelihood_state_%d" % j for j in range(n_states)]
        spatial_dist(tracks, attrib_cols, spatial_dist_png, pixel_size_um=pixel_size_um,
            bin_size=0.01, kde_width=0.1, cmap="magma", cmap_perc=99.5)

    return occs, diff_coefs, tracks 

def check_input(tracks, **kwargs):
    """
    Check some of the arguments to emdiff.em.

    args
    ----
        Arguments and keyword arguments to emdiff.em().

    returns
    -------
        None. Raises no errors if correct input.

    """
    pos_cols = kwargs.get("pos_cols", ["y", "x"])
    max_jumps_per_track = kwargs.get("max_jumps_per_track", None)
    start_frame = kwargs.get("start_frame", 0)

    # Check that required columns exist
    for c in (["trajectory", "frame"] + pos_cols):
        assert c in tracks.columns, "emdiff.em.check_input: column {} " \
            "not present".format(c)

    # Check that the start frame does not exclude all trajectories
    if start_frame > 0:
        assert (tracks["frame"] >= start_frame).sum() > 0, "emdiff.em.check_input: " \
            "no localizations after the start frame {}".format(start_frame)

    # Check that maximum jumps per trajectory is sane
    if (not max_jumps_per_track is None) and (not max_jumps_per_track is np.inf):
        assert max_jumps_per_track > 0, "emdiff.em.check_input: " \
            "max_jumps_per_track must be positive (given: {})".format(max_jumps_per_track)



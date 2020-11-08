#!/usr/bin/env python
"""
plot.py

"""
import sys
import os
import numpy as np
from scipy.special import gammainc, gamma
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

from .defoc import f_remain 
from .utils import rad_disp_histogram, coarsen_histogram

# Use Arial font
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Arial'

def savefig(out_png, dpi=800, show_result=True):
    """
    Save a matplotlib Figure to a PNG.

    args
    ----
        out_png         :   str, save path
        dpi             :   int

    """
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi)
    plt.close()
    if show_result and sys.platform == "darwin":
        os.system("open {}".format(out_png))

def plot_jump_length_dist(tracks, occs, diff_coefs, out_prefix, pos_cols=["y", "x"],
    n_frames=4, frame_interval=0.00748, pixel_size_um=0.16, loc_error=0.035,
    dz=None, max_jump=5.0, max_jump_pmf=2.0, cmap="gray", figsize_mod=1.0,
    n_gaps=0, use_entire_track=True, max_jumps_per_track=10):

    # Calculate radial displacement histograms for the trajectories
    # in this dataset
    H, bin_edges = rad_disp_histogram(tracks, n_frames=n_frames, 
        pos_cols=pos_cols, bin_size=0.001, max_jump=max_jump,
        pixel_size_um=pixel_size_um, n_gaps=n_gaps, 
        use_entire_track=True, max_jumps_per_track=max_jumps_per_track)

    # Empirical PMF
    H = H.astype(np.float64)
    H = (H.T / H.sum(axis=1)).T

    # Aggregate into bins
    H_agg, bin_edges_agg = coarsen_histogram(H, bin_edges, 20)

    # Empirical CDF
    cdf = np.cumsum(H, axis=1)

    # Calculate the model PMF and CDF
    model_pmf, model_cdf = evaluate_diffusion_model(bin_edges, occs, diff_coefs,
        len(pos_cols), frame_interval=frame_interval, loc_error=loc_error,
        dz=dz, n_frames=n_frames)

    # Plot the jump PMFs
    out_png_pmf = "{}_pmf.png".format(out_prefix)
    plot_jump_length_pmf(bin_edges_agg, H_agg, model_pmfs=model_pmf, model_bin_edges=bin_edges,
        frame_interval=frame_interval, max_jump=max_jump_pmf, cmap=cmap, 
        figsize_mod=1.0, out_png=out_png_pmf)

    # Plot the jump CDFs
    out_png_cdf = "{}_cdf.png".format(out_prefix)
    plot_jump_length_cdf(bin_edges, cdf, model_cdfs=model_cdf, model_bin_edges=bin_edges,
        frame_interval=frame_interval, max_jump=max_jump, cmap=cmap,
        figsize_mod=1.0, out_png=out_png_cdf, fontsize=8)

def plot_jump_length_pmf(bin_edges, pmfs, model_pmfs=None, model_bin_edges=None, 
    frame_interval=0.01, max_jump=2.0, cmap="gray", figsize_mod=1.0, out_png=None):
    """
    Plot jump length histograms at different frame intervals, possibly with a model
    overlay.

    args
    ----
        bin_edges       :   1D ndarray of shape (n_bins+1), the edges of each jump
                            length bin in um
        pmfs            :   2D ndarray of shape (n_frames, n_bins), the jump length
                            histogram. This is normalized, if not already normalized.
        model_pmfs      :   2D ndarray of shape (n_frames, n_bins_model), the
                            model PMFs for each frame interval in um
        model_bin_edges :   1D ndarray of shape (n_bins_model+1), the edges of each 
                            jump length bin for the model PMFs in um. If not given,
                            this function defaults to *bin_edges*.
        frame_interval  :   float, the time between frames in seconds
        max_jump        :   float, the maximum jump length to show in um
        cmap            :   str, color palette to use for each jump length. If a hex color
                            (for instance, "#A1A1A1"), then each frame interval is 
                            colored the same.
        figsize_mod     :   float, modifier for the default figure size
        out_png         :   str, a file to save this plot to. If not specified, the plot
                            is not saved.

    returns
    -------
        (
            matplotlib.pyplot.Figure,
            1D ndarray of matplotlib.axes.Axes
        )

    """
    # Check user inputs and get the number of bins and bin size
    assert len(pmfs.shape) == 2
    n_frames, n_bins = pmfs.shape 
    exp_bin_size = bin_edges[1] - bin_edges[0]
    bin_centers = bin_edges[:-1] + 0.5 * exp_bin_size    
    assert bin_edges.shape[0] == n_bins + 1
    if not model_pmfs is None:
        assert model_pmfs.shape[0] == n_frames 
        _, n_bins_model = model_pmfs.shape 
        if not model_bin_edges is None:
            assert n_bins_model == model_bin_edges.shape[0] - 1
            model_bin_size = model_bin_edges[1] - model_bin_edges[0]
            model_bin_centers = model_bin_edges[:-1] + model_bin_size * 0.5
        else:
            assert n_bins_model == n_bins
            model_bin_centers = bin_centers
            model_bin_size = exp_bin_size

        # PMF scaling, accounting for disparate bin sizes
        scale_factor = exp_bin_size / model_bin_size 


    # Bar width for bar plot
    width = exp_bin_size * 0.8

    # Generate the plot axes
    fig, axes = plt.subplots(n_frames, 1, figsize=(4.2*figsize_mod, 0.75*n_frames*figsize_mod),
        sharex=True)
    if n_frames == 1:
        axes = np.array([axes])

    # Make colors for each frame interval
    assert isinstance(cmap, str)
    if cmap[0] == "#":
        palette = [cmap for j in range(n_frames)]
    else:
        palette = sns.color_palette(cmap, n_frames)

    # Plot the PMF for each frame interval
    for t in range(n_frames):

        # Plot the experimental data
        if pmfs[t,:].sum() == 0:
            exp_pmf = np.zeros(pmfs[t,:].shape, dtype=np.float64)
        else:
            exp_pmf = pmfs[t,:].astype(np.float64) / pmfs[t,:].sum()
        axes[t].bar(bin_centers, exp_pmf, color=palette[t], edgecolor="k", linewidth=1, 
            width=width, label=None)

        # Plot the model 
        if not model_pmfs is None:
            axes[t].plot(model_bin_centers, model_pmfs[t,:]*scale_factor, linestyle='-',
                linewidth=1.5, color='k', label=None)

        # For labels
        axes[t].plot([], [], linestyle="", marker=None, color="w",
            label="$\Delta t = ${:.4f} sec".format((t+1)*frame_interval))

        axes[t].legend(frameon=False, prop={"size": 6}, loc="upper right")
        axes[t].set_yticks([])

        # Kill some of the plot spines
        for s in ["top", "right", "left"]:
            axes[t].spines[s].set_visible(False)

    # Only show jumps up to the max jump length 
    if not max_jump is None:
        axes[0].set_xlim((0, max_jump))
    axes[-1].set_xlabel("2D radial displacement ($\mu$m)", fontsize=10)

    # Save to a file, if desired
    if not out_png is None:
        savefig(out_png)

    return fig, axes 

def plot_jump_length_cdf(bin_edges, cdfs, model_cdfs=None, model_bin_edges=None,
    frame_interval=0.01, max_jump=5.0, cmap='gray', figsize_mod=1.0, out_png=None,
    fontsize=8):
    """
    Plot jump length cumulative distribution functions at different frame intervals,
    potentially with a model overlay.

    args
    ----
        bin_edges       :   1D ndarray of shape (n_bins+1), the edges of each jump
                            length bin in um
        cdfs            :   2D ndarray of shape (n_frames, n_bins), the jump length
                            CDFs
        model_cdfs      :   2D ndarray of shape (n_frames, n_bins_model), the
                            model CDFs for each frame interval in um
        model_bin_edges :   1D ndarray of shape (n_bins_model+1), the edges of each 
                            jump length bin for the model CDFs in um. If not given,
                            this function defaults to *bin_edges*.
        frame_interval  :   float, the time between frames in seconds
        max_jump        :   float, the maximum jump length to show in um
        cmap            :   str, color palette to use for each jump length. If a hex color
                            (for instance, "#A1A1A1"), then each frame interval is 
                            colored the same.
        figsize_mod     :   float, modifier for the default figure size
        out_png         :   str, a file to save this plot to. If not specified, the plot
                            is not saved.

    returns
    -------
        (
            matplotlib.pyplot.Figure,
            list of matplotlib.axes.Axes
        )

    """
    # Check user inputs and figure out what kind of plot to make. 
    # plot_case == 0: plot the experimental CDFs, model overlay, and model residuals
    # plot_case == 1: plot the experimental CDFs and model overlay, but no residuals
    # plot_case == 2: plot only the experimental CDFs
    n_frames, n_bins = cdfs.shape 
    assert bin_edges.shape[0] == n_bins + 1
    bins_right = bin_edges[1:]
    bin_size = bin_edges[1] - bin_edges[0]
    if not model_cdfs is None:
        n_frames_model, n_bins_model = model_cdfs.shape
        if not model_bin_edges is None:
            assert model_bin_edges.shape[0] == n_bins_model + 1
            model_bin_size = model_bin_edges[1] - model_bin_edges[0]
            model_bins_right = model_bin_edges[1:]
        else:
            assert model_cdfs.shape == cdfs.shape 
            model_bins_right = bins_right 

        # Choose whether or not to plot the residuals
        if model_bins_right.shape == bins_right.shape:
            plot_case = 0
        else:
            plot_case = 1
    else:
        plot_case = 2

    # Configure the colors to use during plotting
    assert isinstance(cmap, str)
    if cmap[0] == "#":
        palette = [cmap for j in range(n_frames)]
    else:
        palette = sns.color_palette(cmap, n_frames)

    # Plot the experimental CDFs with a model overlay and residuals below
    if plot_case == 0:
        fig, ax = plt.subplots(2, 1, figsize=(3*figsize_mod, 3*figsize_mod),
            gridspec_kw={'height_ratios': [3,1]}, sharex=True)

    # Plot the experimental CDFs, potentially with a model overlay, and no residuals
    else:
        fig, ax = plt.subplots(figsize=(3*figsize_mod, 2*figsize_mod))
        ax = [ax]

    # Plot the experimental CDFs
    for t in range(n_frames):
        ax[0].plot(bins_right, cdfs[t,:], color=palette[t], linestyle='-',
            label="{:.4f} sec".format((t+1)*frame_interval))

    # Plot the model CDFs
    if plot_case == 0 or plot_case == 1:
        for t in range(n_frames):
            ax[0].plot(model_bins_right, model_cdfs[t,:], color="k", 
                linestyle="--", label=None)
        ax[0].plot([], [], color="k", linestyle="--", label="Model")

    # Plot the model residuals
    if plot_case == 0:
        residuals = cdfs - model_cdfs 
        for t in range(n_frames):
            ax[1].plot(bins_right, residuals[t,:], color=palette[t], linestyle='-',
                label="{:.4f} sec".format((t+1)*frame_interval), linewidth=1)
        ax[1].set_xlabel("Jump length ($\mu$m)", fontsize=fontsize)
        ax[1].set_ylabel("Residuals", fontsize=fontsize)

        # Center the residuals on zero
        ax1_ylim = np.abs(residuals).max() * 1.5
        ax[1].set_ylim((-ax1_ylim, ax1_ylim))
        ax[1].set_xlim((0, max_jump))
        ax[1].tick_params(labelsize=fontsize)

    # Axis labels and legend
    ax[0].set_ylabel("CDF", fontsize=fontsize)
    ax[0].set_xlim((0, max_jump))
    ax[0].legend(frameon=False, prop={'size': fontsize}, loc="lower right")
    ax[0].tick_params(labelsize=fontsize)

    # Save to a file, if desired
    if not out_png is None:
        savefig(out_png)

    return fig, ax 

def evaluate_diffusion_model(bin_edges, occs, diff_coefs, n_dimensions,
    frame_interval=0.00748, loc_error=0.0, dz=None, n_frames=4):
    """
    Evaluate the jump length distribution for a normal diffusive 
    mixture model at some number of frame intervals.

    args
    ----
        bin_edges           :   1D ndarray of shape (n_bins,), the edge
                                of each spatial bin in um
        occs                :   1D ndarray of shape (n_states,), the 
                                fractional occupations of each diffusive state
        diff_coefs          :   1D ndarray of shape (n_states,), the 
                                diffusion coefficients corresponding to 
                                each state in um^2 s^-1
        n_dimensions        :   int, the number of spatial dimensions
        frame_interval      :   float, time between frames in seconds
        loc_error           :   float, localization error in um
        dz                  :   float, focal depth in um
        n_frames            :   int, the number of frame delays over which 
                                to compute the PMF and CDF

    returns
    -------
        (
            2D ndarray of shape (n_frames, n_bins), the normalized PMF;
            2D ndarray of shape (n_frames, n_bins), the normalized CDF
        )

    """
    n_states = len(occs)
    le2 = loc_error ** 2

    # Size of the radial displacement bins in um
    bin_size = bin_edges[1] - bin_edges[0]
    r2 = bin_edges[1:] ** 2

    # Centers of each radial displacement bin
    bin_c = bin_edges[:-1] + bin_size * 0.5
    r2_c = bin_c ** 2
    rdim_c = np.power(bin_c, n_dimensions-1)
    n_bins = bin_c.shape[0]

    pmf = np.zeros((n_frames, n_bins), dtype=np.float64)
    cdf = np.zeros((n_frames, n_bins), dtype=np.float64)

    # Calculate the contribution to the PMF and CDF from each state
    for j in range(n_states):
        D = diff_coefs[j]
        occ = occs[j]

        defoc_mod = f_remain(D, n_frames, frame_interval, dz)
        for i in range(n_frames):

            # Spatial variance of this state at this frame interval
            var2 = 4 * (D * frame_interval * (i+1) + le2)

            # Contribution to the PMF
            pmf[i,:] += (occ * defoc_mod[i] * 2 * rdim_c * np.exp(-r2_c / var2) / \
                (np.power(var2, n_dimensions/2.0) * gamma(n_dimensions/2.0)))

            # Contribution to the CDF
            cdf[i,:] += (occ * defoc_mod[i] * gammainc(n_dimensions/2.0, \
                r2 / var2))

    # Normalize
    pmf = (pmf.T / pmf.sum(axis=1)).T
    cdf = (cdf.T / cdf[:,-1]).T

    return pmf, cdf 

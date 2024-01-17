#!/usr/bin/env python
"""
plot.py

"""
import sys
import os

# Numeric
import numpy as np

# Gaussian filtering for KDE
from scipy.ndimage import gaussian_filter

# DataFrames
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# Scalebars
try:
    from matplotlib_scalebar.scalebar import ScaleBar

    scalebar_active = True
except ModuleNotFoundError:
    scalebar_active = False

# Internal utilities
from .defoc import f_remain
from .utils import rad_disp_histogram, evaluate_diffusion_model, coarsen_histogram

# Use Arial font
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = "Arial"


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


def kill_ticks(axes, spines=True):
    """
    Remove ticks on a matplotlib.axes.Axes object. If *spines*,
    also remove the spines.

    """
    axes.set_xticks([])
    axes.set_yticks([])
    if spines:
        for s in ["top", "bottom", "left", "right"]:
            axes.spines[s].set_visible(False)


def plot_jump_length_dist(
    tracks,
    occs,
    diff_coefs,
    out_prefix,
    pos_cols=["y", "x"],
    n_frames=4,
    frame_interval=0.00748,
    pixel_size_um=0.16,
    loc_error=0.035,
    dz=None,
    max_jump=5.0,
    max_jump_pmf=2.0,
    cmap="gray",
    figsize_mod=1.0,
    n_gaps=0,
    use_entire_track=True,
    max_jumps_per_track=10,
):
    """
    Given a set of trajectories and a particular diffusive mixture model,
    plot the observed and model radial jump histograms alongside each other.

    args
    ----
        tracks          :   pandas.DataFrame, with columns "trajectory", "frame",
                            and the contents of *pos_cols*
        occs            :   1D ndarray, fractional state occupations
        diff_coefs      :   1D ndarray, diffusion coefficients for each
                            state in um^2 s^-1
        out_prefix      :   str, the prefix for the output plots
        pos_cols        :   list of str, positional coordinate columns
        n_frames        :   int, the number of time points to consider
        frame_interval  :   float, the time between frames in seconds
        pixel_size_um   :   float, size of pixels in um
        loc_error       :   float, localization error in um
        dz              :   float, focal depth in um
        max_jump        :   float, the maximum jump length to show in um
        max_jump_pmf    :   float, the maximum jump length to show in PMF plots
        cmap            :   str, color palette to use for each jump length
        figsize_mod     :   float, modifier for the default figure size
        n_gaps          :   int, the number of gaps allowed during tracking
        use_entire_track:   bool, use all jumps from every trajectory
        max_jumps_per_track:    int. If *use_entire_track* is False, the maximum
                                number of jumps per trajectory to consider

    returns
    -------
        None; plots directly to output plots

    """
    # Calculate radial displacement histograms for the trajectories
    # in this dataset
    H, bin_edges = rad_disp_histogram(
        tracks,
        n_frames=n_frames,
        pos_cols=pos_cols,
        bin_size=0.001,
        max_jump=max_jump,
        pixel_size_um=pixel_size_um,
        n_gaps=n_gaps,
        use_entire_track=True,
        max_jumps_per_track=max_jumps_per_track,
    )

    # Empirical PMF
    H = H.astype(np.float64)
    H = (H.T / H.sum(axis=1)).T

    # Aggregate into bins
    H_agg, bin_edges_agg = coarsen_histogram(H, bin_edges, 20)

    # Empirical CDF
    cdf = np.cumsum(H, axis=1)

    # Calculate the model PMF and CDF
    model_pmf, model_cdf = evaluate_diffusion_model(
        bin_edges,
        occs,
        diff_coefs,
        len(pos_cols),
        frame_interval=frame_interval,
        loc_error=loc_error,
        dz=dz,
        n_frames=n_frames,
    )

    # Plot the jump PMFs
    out_png_pmf = "{}_pmf.png".format(out_prefix)
    plot_jump_length_pmf(
        bin_edges_agg,
        H_agg,
        model_pmfs=model_pmf,
        model_bin_edges=bin_edges,
        frame_interval=frame_interval,
        max_jump=max_jump_pmf,
        cmap=cmap,
        figsize_mod=1.0,
        out_png=out_png_pmf,
    )

    # Plot the jump CDFs
    out_png_cdf = "{}_cdf.png".format(out_prefix)
    plot_jump_length_cdf(
        bin_edges,
        cdf,
        model_cdfs=model_cdf,
        model_bin_edges=bin_edges,
        frame_interval=frame_interval,
        max_jump=max_jump,
        cmap=cmap,
        figsize_mod=1.0,
        out_png=out_png_cdf,
        fontsize=8,
    )


def plot_jump_length_pmf(
    bin_edges,
    pmfs,
    model_pmfs=None,
    model_bin_edges=None,
    frame_interval=0.01,
    max_jump=2.0,
    cmap="gray",
    figsize_mod=1.0,
    out_png=None,
):
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
    fig, axes = plt.subplots(
        n_frames,
        1,
        figsize=(4.2 * figsize_mod, 0.75 * n_frames * figsize_mod),
        sharex=True,
    )
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
        if pmfs[t, :].sum() == 0:
            exp_pmf = np.zeros(pmfs[t, :].shape, dtype=np.float64)
        else:
            exp_pmf = pmfs[t, :].astype(np.float64) / pmfs[t, :].sum()
        axes[t].bar(
            bin_centers,
            exp_pmf,
            color=palette[t],
            edgecolor="k",
            linewidth=1,
            width=width,
            label=None,
        )

        # Plot the model
        if not model_pmfs is None:
            axes[t].plot(
                model_bin_centers,
                model_pmfs[t, :] * scale_factor,
                linestyle="-",
                linewidth=1.5,
                color="k",
                label=None,
            )

        # For labels
        axes[t].plot(
            [],
            [],
            linestyle="",
            marker=None,
            color="w",
            label="∆t = {:.4f} sec".format((t + 1) * frame_interval),
        )

        axes[t].legend(frameon=False, prop={"size": 6}, loc="upper right")
        axes[t].set_yticks([])

        # Kill some of the plot spines
        for s in ["top", "right", "left"]:
            axes[t].spines[s].set_visible(False)

    # Only show jumps up to the max jump length
    if not max_jump is None:
        axes[0].set_xlim((0, max_jump))
    axes[-1].set_xlabel("2D radial displacement (µm)", fontsize=10)

    # Save to a file, if desired
    if not out_png is None:
        savefig(out_png)

    return fig, axes


def plot_jump_length_cdf(
    bin_edges,
    cdfs,
    model_cdfs=None,
    model_bin_edges=None,
    frame_interval=0.01,
    max_jump=5.0,
    cmap="gray",
    figsize_mod=1.0,
    out_png=None,
    fontsize=8,
):
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
        fig, ax = plt.subplots(
            2,
            1,
            figsize=(3 * figsize_mod, 3 * figsize_mod),
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True,
        )

    # Plot the experimental CDFs, potentially with a model overlay, and no residuals
    else:
        fig, ax = plt.subplots(figsize=(3 * figsize_mod, 2 * figsize_mod))
        ax = [ax]

    # Plot the experimental CDFs
    for t in range(n_frames):
        ax[0].plot(
            bins_right,
            cdfs[t, :],
            color=palette[t],
            linestyle="-",
            label="{:.4f} sec".format((t + 1) * frame_interval),
        )

    # Plot the model CDFs
    if plot_case == 0 or plot_case == 1:
        for t in range(n_frames):
            ax[0].plot(
                model_bins_right,
                model_cdfs[t, :],
                color="k",
                linestyle="--",
                label=None,
            )
        ax[0].plot([], [], color="k", linestyle="--", label="Model")

    # Plot the model residuals
    if plot_case == 0:
        residuals = cdfs - model_cdfs
        for t in range(n_frames):
            ax[1].plot(
                bins_right,
                residuals[t, :],
                color=palette[t],
                linestyle="-",
                label="{:.4f} sec".format((t + 1) * frame_interval),
                linewidth=1,
            )
        ax[1].set_xlabel("Jump length (µm)", fontsize=fontsize)
        ax[1].set_ylabel("Residuals", fontsize=fontsize)

        # Center the residuals on zero
        ax1_ylim = np.abs(residuals).max() * 1.5
        ax[1].set_ylim((-ax1_ylim, ax1_ylim))
        ax[1].set_xlim((0, max_jump))
        ax[1].tick_params(labelsize=fontsize)

    # Axis labels and legend
    ax[0].set_ylabel("CDF", fontsize=fontsize)
    ax[0].set_xlim((0, max_jump))
    ax[0].legend(frameon=False, prop={"size": fontsize}, loc="lower right")
    ax[0].tick_params(labelsize=fontsize)

    # Save to a file, if desired
    if not out_png is None:
        savefig(out_png)

    return fig, ax


def spatial_dist(
    tracks,
    attrib_cols,
    out_png,
    pixel_size_um=0.16,
    bin_size=0.01,
    kde_width=0.08,
    cmap="magma",
    cmap_perc=99.5,
):
    """
    Plot the spatial distribution of states for a set of trajectories.

    args
    ----
        tracks          :   pandas.DataFrame, trajectories
        attrib_cols     :   list of str, a set of columns in *tracks*, each
                            corresponding to a diffusive state, giving
                            the likelihood of that state given the corresponding
                            trajectory
        out_png         :   str, output plot path
        pixel_size_um   :   float, size of pixels in um
        bin_size        :   float, size of the pixels in um
        kde_width       :   float, size of the KDE kernel in um
        cmap            :   str, color map

    """
    n_states = len(attrib_cols)

    # Spatial limits
    n_pixels_y = int(tracks["y"].max()) + 1
    n_pixels_x = int(tracks["x"].max()) + 1

    # Spatial binning strategy
    n_bins_y = int(n_pixels_y * pixel_size_um / bin_size) + 2
    n_um_y = n_bins_y * bin_size
    n_bins_x = int(n_pixels_x * pixel_size_um / bin_size) + 2
    n_um_x = n_bins_x * bin_size
    bin_edges_y = np.arange(0, n_um_y + bin_size, bin_size)
    bin_edges_x = np.arange(0, n_um_x + bin_size, bin_size)

    # Plot layout
    M = n_states // 2 + 1 if (n_states % 2 == 1) else n_states // 2
    fig, ax = plt.subplots(2, M + 1, figsize=((n_states + 1) * 2.5, 6))

    # Raw localization density
    H = np.histogram2d(
        tracks["y"] * pixel_size_um,
        tracks["x"] * pixel_size_um,
        bins=(bin_edges_y, bin_edges_x),
    )[0].astype(np.float64)

    # KDE
    loc_density = gaussian_filter(H, kde_width / bin_size)
    ax[0, 0].imshow(
        loc_density,
        cmap="gray",
        vmin=0,
        vmax=np.percentile(loc_density, cmap_perc),
        origin="bottom",
    )
    ax[0, 0].set_title("Localization density")

    # The maximum likelihood state for each trajectory
    X = np.asarray(tracks[attrib_cols])
    max_l_states = np.argmax(X, axis=1)

    # Scatter plot of maximum likelihood states
    colors = sns.color_palette(cmap, n_states + 1)
    for j, attrib_col in enumerate(attrib_cols):
        k = n_states - j - 1
        exclude = pd.isnull(tracks[attrib_col])
        include = np.logical_and(~exclude, max_l_states == k)
        ax[1, 0].scatter(
            tracks.loc[include, "x"] * pixel_size_um,
            tracks.loc[include, "y"] * pixel_size_um,
            color=colors[j],
            s=1.5,
        )
        ax[1, 0].set_aspect("equal")
    ax[1, 0].set_title("Maximum likelihood state")

    # Spatial distribution of likelihoods for each state
    for j, attrib_col in enumerate(attrib_cols):
        # Do not include singlets
        exclude = pd.isnull(tracks[attrib_col])

        # Make a histogram of the likelihood
        H = np.histogram2d(
            tracks.loc[~exclude, "y"] * pixel_size_um,
            tracks.loc[~exclude, "x"] * pixel_size_um,
            bins=(bin_edges_y, bin_edges_x),
            weights=tracks.loc[~exclude, attrib_col],
        )[0].astype(np.float64)

        # Kernel density estimate
        kde = gaussian_filter(H, kde_width / bin_size)
        ax_y = j % 2
        ax_x = j // 2
        ax[ax_y, ax_x + 1].imshow(
            kde,
            cmap="inferno",
            vmin=0,
            vmax=np.percentile(kde, cmap_perc),
            origin="bottom",
        )
        ax[ax_y, ax_x + 1].set_title("State %d" % (j + 1))

    # Add scale bars, if matplotlib_scalebar is installed
    if scalebar_active:
        s = ScaleBar(bin_size, "um", frameon=False, color="w", location="lower right")
        ax[0, 0].add_artist(s)

        for j in range(n_states):
            ax_y = j % 2
            ax_x = j // 2
            s = ScaleBar(
                bin_size, "um", frameon=False, color="w", location="lower right"
            )
            ax[ax_y, ax_x + 1].add_artist(s)

    # Remove the ticks
    for i in range(2):
        for j in range(M + 1):
            kill_ticks(ax[i, j])

    # Save figure
    savefig(out_png)

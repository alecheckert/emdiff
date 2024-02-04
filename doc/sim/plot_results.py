#!/usr/bin/env python
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = "Arial"


def savefig(out_png: str):
    plt.tight_layout()
    plt.savefig(out_png, dpi=600)
    plt.close()
    os.system(f"open {out_png} -a Preview")


def open_spine(ax: matplotlib.axes.Axes):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)


def add_log_ticks(ax: matplotlib.axes.Axes, side="x"):
    """Force log-spaced major and minor ticks, regardless of
    figure size."""
    if side == "x":
        lo, hi = ax.get_xlim()
    else:
        lo, hi = ax.get_ylim()
    log_lo = int(np.floor(np.log10(lo)))
    log_hi = int(np.ceil(np.log10(hi)))
    major_ticks = []
    minor_ticks = []
    for major in range(log_lo, log_hi + 1):
        x = 10**major
        major_ticks.append(x)
        for i in range(10):
            minor_ticks.append(i * x)
    major_ticks = np.array(major_ticks)
    minor_ticks = np.array(minor_ticks)
    major_ticks = major_ticks[np.logical_and(major_ticks >= lo, major_ticks <= hi)]
    minor_ticks = minor_ticks[np.logical_and(minor_ticks >= lo, minor_ticks <= hi)]
    if side == "x":
        ax.set_xticks(major_ticks, minor=False)
        ax.set_xticks(minor_ticks, minor=True)
    else:
        ax.set_yticks(major_ticks, minor=False)
        ax.set_yticks(minor_ticks, minor=True)


def figure1():
    """Show distribution of the max-ELBO state for each
    simulation."""
    # 30000 simulated particles ~= 7000 observed tracks, since most
    # simulated particles do not pass through the simulated focus.
    conditions = [
        ("Mixture 1: $K_{true} = 1$", "results_1states_ntracks30000.csv"),
        ("Mixture 2: $K_{true} = 2$", "results_2states_ntracks30000.csv"),
        ("Mixture 3: $K_{true} = 3$", "results_3states_ntracks30000.csv"),
        ("Mixture 4: $K_{true} = 4$", "results_4states_ntracks30000.csv"),
        ("Mixture 5: $K_{true} = 5$", "results_5states_ntracks30000.csv"),
        ("Mixture 6: $K_{true} = 6$", "results_6states_ntracks30000.csv"),
    ]

    n_conditions = len(conditions)
    fig, axes = plt.subplots(
        n_conditions, 1, figsize=(2.5, n_conditions * 1), sharex=True
    )
    fontsize = 12
    ind = np.arange(1, 8)

    for i, (title, csv) in enumerate(conditions):
        ax = axes[i]
        f = pd.read_csv(csv)
        f["replicate"] = np.repeat(np.arange(len(f) // 7), 7)
        by_state = pd.Series(np.zeros(len(ind), dtype=np.float64), index=ind)
        obs = f.loc[f.groupby("replicate")["elbo"].idxmax()].groupby("n_states").size()
        by_state.loc[obs.index] = obs
        ax.bar(by_state.index, by_state, color="w", edgecolor="k", width=0.8)
        open_spine(ax)
        ax.tick_params(labelsize=fontsize)
        # ax.set_ylabel("Frequency", fontsize=fontsize)
        ax.set_title(title, fontsize=fontsize)

    axes[-1].set_xlabel("# states in model ($K$)", fontsize=fontsize)
    axes[-1].set_xticks(by_state.index)

    savefig("figure1.png")


def figure2():
    """Show the ground truth models (diffusion coefficients
    and occupations) for each of the simulations considered
    in figure1."""
    models = [
        # 1 states
        (np.array([5.0]), np.array([1.0])),
        # 2 states
        (np.array([5.0, 20.0]), np.array([0.5, 0.5])),
        # 3 states
        (np.array([0.1, 1.0, 5.0]), np.array([0.2, 0.4, 0.4])),
        # 4 states
        (np.array([0.02, 0.3, 2.0, 8.0]), np.array([0.1, 0.3, 0.2, 0.4])),
        # 5 states
        (np.array([0.04, 0.43, 1.2, 4.9, 11.0]), np.array([0.1, 0.15, 0.3, 0.25, 0.2])),
        # 6 states
        (
            np.array([0.01, 0.2, 0.9, 2.3, 6.6, 15.0]),
            np.array(
                [0.01278217, 0.31094062, 0.0178494, 0.44588117, 0.11149864, 0.101048]
            ),
        ),
    ]

    n_models = len(models)
    fig, axes = plt.subplots(n_models, 1, figsize=(2.5, 1 * n_models), sharex=True)
    fontsize = 12
    xlim = (10 ** (-2.5), 10**2.5)

    for i, (diff_coefs, occs) in enumerate(models):
        ax = axes[i]
        for d, o in zip(diff_coefs, occs):
            ax.plot([d, d], [0, o], color="r")
        ax.set_xscale("log")
        ax.set_xlim(xlim)
        ax.set_ylim((0, ax.get_ylim()[1]))
        ax.tick_params(labelsize=fontsize)
        open_spine(ax)
        ax.set_title("Mixture %d: $K_{true} = %d$" % (i + 1, i + 1), fontsize=fontsize)
        add_log_ticks(ax, side="x")

    axes[-1].set_xlabel("Diff. coef. ($\mu$m$^{2}$/s)", fontsize=fontsize)

    savefig("figure2.png")


if __name__ == "__main__":
    figure1()
    figure2()

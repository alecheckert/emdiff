#!/usr/bin/env python
import dask
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar
from itertools import chain
from typing import List, Tuple
from emdiff.vb import vbdiff
from strobesim import strobe_multistate


FRAME_INTERVAL = 0.005
FOCAL_DEPTH_UM = 0.7
LOC_ERROR = 0.02


def fit_vbdiff(
    spots: pd.DataFrame, n_states: int
) -> Tuple[np.ndarray, np.ndarray, float, dict]:
    """Fit an observed set of trajectories to a multi-state
    Brownian mixture model using vbdiff. We run the algorithm
    5 times with randomly initialized starting diffusion
    coefficients and take the resulting model with the highest
    ELBO.

    Parameters
    ----------
    spots       :   pandas.DataFrame with columns "frame",
                    "trajectory", "x", and "y"
    n_states    :   number of states in the mixture model

    Returns
    -------
    1D ndarray of shape *n_states*, diffusion coefficients in
        squared microns per sec;
    1D ndarray of shape *n_states*, state occupations;
    float, evidence lower bound (ELBO);
    dict with keys "A", "B", "C", "D", "E", "F", and "G",
        individual terms contributing to ELBO
    """
    diff_coefs = None
    occs = None
    elbo = -np.inf
    evidence_terms = None
    for _ in range(5):
        guess_diff_coefs = np.random.gamma(1.0, 8.0, size=n_states)
        (
            r,
            n,
            A,
            B,
            occs_,
            diff_coefs_,
            elbo_,
            evA,
            evB,
            evC,
            evD,
            evE,
            evF,
            evG,
        ) = vbdiff(
            spots.copy(),
            n_states=n_states,
            pixel_size_um=1.0,
            frame_interval=FRAME_INTERVAL,
            loc_error=LOC_ERROR,
            pseudocounts=2.0,
            return_posterior=True,
            dz=FOCAL_DEPTH_UM,
            guess=guess_diff_coefs,
        )
        if elbo_ > elbo:
            diff_coefs = diff_coefs_
            occs = occs_
            elbo = elbo_
            evidence_terms = {
                "A": evA,
                "B": evB,
                "C": evC,
                "D": evD,
                "E": evE,
                "F": evF,
                "G": evG,
                "n_states": n_states,
                "elbo": elbo,
            }

    return diff_coefs, occs, elbo, evidence_terms


def simulate_and_fit_1states():
    n_replicates = 64
    n_tracks = 10000
    results = []

    @dask.delayed
    def run_replicate(rep_idx: int) -> List[dict]:
        spots = strobe_multistate(
            n_tracks,
            np.array([5.0]),
            np.array([1.0]),
            motion="brownian",
            geometry="sphere",
            radius=5.0,
            dz=FOCAL_DEPTH_UM,
            frame_interval=FRAME_INTERVAL,
            loc_error=LOC_ERROR,
            track_len=100,
            bleach_prob=0.1,
        )
        results = []
        for n_states in [1, 2, 3, 4, 5, 6, 7]:
            print(f"\nfitting replicate {rep_idx} to K={n_states}...")
            _, _, _, evidence_terms = fit_vbdiff(spots, n_states=n_states)
            results.append(evidence_terms)
        return results

    tasks = map(run_replicate, range(n_replicates))
    with ProgressBar():
        results = dask.compute(*tasks, num_workers=16, scheduler="processes")

    results = pd.DataFrame(list(chain(*results)))
    print(results)
    results.to_csv(f"results_1states_ntracks{n_tracks}.csv", index=False)


def simulate_and_fit_2states():
    n_replicates = 64
    n_tracks = 10000
    results = []

    @dask.delayed
    def run_replicate(rep_idx: int) -> List[dict]:
        spots = strobe_multistate(
            n_tracks,
            np.array([5.0, 20.0]),
            np.array([0.5, 0.5]),
            motion="brownian",
            geometry="sphere",
            radius=5.0,
            dz=FOCAL_DEPTH_UM,
            frame_interval=FRAME_INTERVAL,
            loc_error=LOC_ERROR,
            track_len=100,
            bleach_prob=0.1,
        )
        results = []
        for n_states in [1, 2, 3, 4, 5, 6, 7]:
            print(f"\nfitting replicate {rep_idx} to K={n_states}...")
            _, _, _, evidence_terms = fit_vbdiff(spots, n_states=n_states)
            results.append(evidence_terms)
        return results

    tasks = map(run_replicate, range(n_replicates))
    with ProgressBar():
        results = dask.compute(*tasks, num_workers=16, scheduler="processes")

    results = pd.DataFrame(list(chain(*results)))
    print(results)
    results.to_csv(f"results_2states_ntracks{n_tracks}.csv", index=False)


def simulate_and_fit_3states():
    n_replicates = 64
    n_tracks = 10000
    results = []

    @dask.delayed
    def run_replicate(rep_idx: int) -> List[dict]:
        spots = strobe_multistate(
            n_tracks,
            np.array([0.1, 1.0, 5.0]),
            np.array([0.2, 0.4, 0.4]),
            motion="brownian",
            geometry="sphere",
            radius=5.0,
            dz=FOCAL_DEPTH_UM,
            frame_interval=FRAME_INTERVAL,
            loc_error=LOC_ERROR,
            track_len=100,
            bleach_prob=0.1,
        )
        results = []
        for n_states in [1, 2, 3, 4, 5, 6, 7]:
            print(f"\nfitting replicate {rep_idx} to K={n_states}...")
            _, _, _, evidence_terms = fit_vbdiff(spots, n_states=n_states)
            results.append(evidence_terms)
        return results

    tasks = map(run_replicate, range(n_replicates))
    with ProgressBar():
        results = dask.compute(*tasks, num_workers=16, scheduler="processes")

    results = pd.DataFrame(list(chain(*results)))
    print(results)
    results.to_csv(f"results_3states_ntracks{n_tracks}.csv", index=False)


def simulate_and_fit_4states():
    n_replicates = 64
    n_tracks = 10000
    results = []

    @dask.delayed
    def run_replicate(rep_idx: int) -> List[dict]:
        spots = strobe_multistate(
            n_tracks,
            np.array([0.02, 0.3, 2.0, 8.0]),
            np.array([0.1, 0.3, 0.2, 0.4]),
            motion="brownian",
            geometry="sphere",
            radius=5.0,
            dz=FOCAL_DEPTH_UM,
            frame_interval=FRAME_INTERVAL,
            loc_error=LOC_ERROR,
            track_len=100,
            bleach_prob=0.1,
        )
        results = []
        for n_states in [1, 2, 3, 4, 5, 6, 7]:
            print(f"\nfitting replicate {rep_idx} to K={n_states}...")
            _, _, _, evidence_terms = fit_vbdiff(spots, n_states=n_states)
            results.append(evidence_terms)
        return results

    tasks = map(run_replicate, range(n_replicates))
    with ProgressBar():
        results = dask.compute(*tasks, num_workers=16, scheduler="processes")

    results = pd.DataFrame(list(chain(*results)))
    print(results)
    results.to_csv(f"results_4states_ntracks{n_tracks}.csv", index=False)


def simulate_and_fit_5states():
    n_replicates = 64
    n_tracks = 10000
    results = []

    @dask.delayed
    def run_replicate(rep_idx: int) -> List[dict]:
        spots = strobe_multistate(
            n_tracks,
            np.array([0.04, 0.43, 1.2, 4.9, 11.0]),
            np.array([0.1, 0.15, 0.3, 0.25, 0.2]),
            motion="brownian",
            geometry="sphere",
            radius=5.0,
            dz=FOCAL_DEPTH_UM,
            frame_interval=FRAME_INTERVAL,
            loc_error=LOC_ERROR,
            track_len=100,
            bleach_prob=0.1,
        )
        results = []
        for n_states in [1, 2, 3, 4, 5, 6, 7]:
            print(f"\nfitting replicate {rep_idx} to K={n_states}...")
            _, _, _, evidence_terms = fit_vbdiff(spots, n_states=n_states)
            results.append(evidence_terms)
        return results

    tasks = map(run_replicate, range(n_replicates))
    with ProgressBar():
        results = dask.compute(*tasks, num_workers=16, scheduler="processes")

    results = pd.DataFrame(list(chain(*results)))
    print(results)
    results.to_csv(f"results_5states_ntracks{n_tracks}.csv", index=False)


def simulate_and_fit_6states():
    n_replicates = 64
    n_tracks = 10000
    results = []

    @dask.delayed
    def run_replicate(rep_idx: int) -> List[dict]:
        spots = strobe_multistate(
            n_tracks,
            np.array([0.01, 0.2, 0.9, 2.3, 6.6, 15.0]),
            np.array(
                [0.01278217, 0.31094062, 0.0178494, 0.44588117, 0.11149864, 0.101048]
            ),
            motion="brownian",
            geometry="sphere",
            radius=5.0,
            dz=FOCAL_DEPTH_UM,
            frame_interval=FRAME_INTERVAL,
            loc_error=LOC_ERROR,
            track_len=100,
            bleach_prob=0.1,
        )
        results = []
        for n_states in [1, 2, 3, 4, 5, 6, 7]:
            print(f"\nfitting replicate {rep_idx} to K={n_states}...")
            _, _, _, evidence_terms = fit_vbdiff(spots, n_states=n_states)
            results.append(evidence_terms)
        return results

    tasks = map(run_replicate, range(n_replicates))
    with ProgressBar():
        results = dask.compute(*tasks, num_workers=16, scheduler="processes")

    results = pd.DataFrame(list(chain(*results)))
    print(results)
    results.to_csv(f"results_6states_ntracks{n_tracks}.csv", index=False)


if __name__ == "__main__":
    simulate_and_fit_1states()
    simulate_and_fit_2states()
    simulate_and_fit_3states()
    simulate_and_fit_4states()
    simulate_and_fit_5states()
    simulate_and_fit_6states()
    if False:
        spots = strobe_multistate(
            10000,
            np.array([0.02, 0.3, 2.0, 8.0]),
            np.array([0.1, 0.3, 0.2, 0.4]),
            motion="brownian",
            geometry="sphere",
            radius=5.0,
            dz=FOCAL_DEPTH_UM,
            frame_interval=FRAME_INTERVAL,
            loc_error=LOC_ERROR,
            track_len=100,
            bleach_prob=0.1,
        )
        print(spots)
        print(spots["trajectory"].nunique())
        print(spots.groupby("trajectory").size().mean())

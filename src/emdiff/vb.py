#!/usr/bin/env python
"""
vb.py -- variational Bayes version of the EM algorithm

"""
import sys
import os
from time import sleep
from copy import copy
import numpy as np
import pandas as pd
from scipy.special import digamma, loggamma

from .defoc import f_remain
from .utils import sum_squared_jumps, logbeta

INIT_DIFF_COEFS = {
    1: np.array([1.0]),
    2: np.array([0.01, 2.0]),
    3: np.array([0.01, 0.5, 5.0]),
    4: np.array([0.01, 0.5, 1.0, 5.0]),
    5: np.array([0.01, 0.2, 1.0, 3.0, 10.0]),
}


def vbdiff(
    tracks,
    n_states=2,
    pixel_size_um=0.16,
    frame_interval=0.01,
    pos_cols=["y", "x"],
    loc_error=0.035,
    max_jumps_per_track=None,
    start_frame=0,
    max_iter=10000,
    convergence=1.0e-8,
    dz=np.inf,
    guess=None,
    pseudocounts=2.0,
    return_posterior=False,
    allow_neg_diff_coef=False,
):
    """
    Evaluate a variational Bayesian approximation to the posterior
    distribution of a finite-state regular Brownian mixtures, given
    some observed trajectories.

    model description
    -----------------
        likelihood of each trajectory   :   gamma random variable

        prior on state assignments  :   categorical random variables
                                        governed by prior on state occupations

        prior on state occupations  :   Dirichlet random variable with
                                        all coefficients equal to *n0*

        prior on spatial variances  :   inverse gamma random variable
                                        with alpha parameter *n0*
                                        and a beta parameter set by *guess*

        variational approximation: the posterior is assumed to be
            separable in the state assignments and the other random
            variables. There are additional induced factorizations
            of the posterior.

        The posteriors over the state assignments, state occupations, and
        spatial variances :are all conjugate to the corresponding priors
        and have parameters that are described in the *returns* section below.

    args
    ----
        tracks              :   pandas.DataFrame
        n_states            :   int, number of different states to consider
        pixel_size_um       :   float, size of pixels in um
        frame_interval      :   float, time between frames in seconds
        pos_cols            :   list of str, columns with the spatial
                                coordinates of each localization ~~in pixels~~!!
        loc_error           :   float, 1D localization error in um
        max_jumps_per_track :   int
        start_frame         :   int, disregard points before this frame
        max_iter            :   int, maximum number of iterations to do
        convergence         :   float
        dz                  :   float, depth of field in um
        guess               :   1D ndarray of shape (n_states), an initial guess
                                for the diffusion coefficients of each state
                                in um^2 s^-1. This determines the prior over the
                                spatial variance.
        pseudocounts        :   float, the number of pseudocounts in the prior
                                over the mixing coefficients and the spatial
                                variances. Higher means the algorithm requires more
                                data but is more conservative.
                                ~~Do not set below 2.0.~~
        return_posterior    :   bool. In addition to returning the mean diffusion
                                coefficients and state occupancies, also return
                                the parameters for the full posterior distribution
                                over occupancies, diffusion coefficients, and
                                assignments of each trajectory to a state
        allow_neg_diff_coef :   bool, allow the diffusion coefficient to assume
                                negative values if the observed motion is
                                actually slower than the user-provided localization
                                error

    returns
    -------
        if *return_posterior*:
            (
                r, 2D ndarray of shape (n_states, n_tracks); where
                    r[j,i] is the mean probability for trajectory i to
                    inhabit state j under the posterior model;

                n, 1D ndarray of shape (n_states); the Dirichlet parameter
                    over the mixing coefficients under the posterior model;

                A, 1D ndarray of shape (n_states); the alpha parameter over
                    the spatial variances (phi) under the posterior model;

                B, 1D ndarray of shape (n_states); the beta parameter over
                    the spatial variances (phi) under the posterior model;

                occs, 1D ndarray of shape (n_states); the mean state
                    occupations under the posterior model;

                D_mean, 1D ndarray of shape (n_states); the mean diffusion
                    coefficients under the posterior model;

                elbo, float; the evidence lower bound (higher indicates that
                    the model (i.e. the number of states) describes the data
                    better);

                model_likelihood, float; the likelihood of the posterior
                    model given the data
            )

        otherwise:
            (
                occs, 1D ndarray of shape (n_states); the mean state
                    occupations under the posterior model;

                D_mean, 1D ndarray of shape (n_states); the mean diffusion
                    coefficients under the posterior model
            )

    """
    # For internal convenience
    le2 = loc_error**2
    K = n_states
    consider_defoc = (not dz is None) and (not dz is np.inf)

    # Only take points after the start frame
    if start_frame > 0:
        tracks = tracks[tracks["frame"] >= start_frame]

    # Calculate the sum of squared displacements for every trajectory
    # in the dataset
    L = sum_squared_jumps(
        tracks,
        pixel_size_um=pixel_size_um,
        pos_cols=pos_cols,
        max_jumps_per_track=max_jumps_per_track,
    )

    # Make sure we actually have jumps
    if L.empty:
        raise RuntimeError("no jumps in dataset")

    # Sum of squared radial displacements
    sum_r2 = np.asarray(L["sum_r2"])

    # Number of jumps corresponding to each trajectory
    n_jumps = np.asarray(L["n_jumps"] * len(pos_cols) / 2.0)

    # Total number of trajectories
    N = L["trajectory"].nunique()

    # Function to correct state occupations for defocalization bias
    def corr_defoc(A, B, n):
        """
        args
        ----
            A       :   posterior alpha parameters for each state
            B       :   posterior beta parameters for each state
            n       :   counts to correct

        returns
        -------
            version of *n*, corrected

        """
        phi = B / (A - 1)
        corr = np.zeros(K)
        diff_coefs = np.maximum(0, (phi / 4 - le2) / frame_interval)
        for j in range(K):
            corr[j] = f_remain(diff_coefs[j], 1, frame_interval, dz)[0]
        corr = 1.0 / corr
        corr /= corr.max()
        return n * corr

    ## PRIOR DEFINITION

    # Treat the initial guess at the diffusion coefficients as
    # the mean of an inverse gamma prior over the diffusion
    # coefficient
    if guess is None:
        if n_states <= max(INIT_DIFF_COEFS.keys()):
            diff_coefs = copy(INIT_DIFF_COEFS[n_states])
        else:
            diff_coefs = np.logspace(-2, 2, n_states)
    else:
        diff_coefs = np.asarray(guess)

    # Offset by localization error
    phi = 4 * (diff_coefs * frame_interval + le2)

    # Prior over the mixing coefficients
    n0 = np.ones(K) * pseudocounts

    # Expectations of the state assignment for each trajectory.
    # r[i,j] is the probability that trajectory i is in state j,
    # given the current model
    r = np.zeros((K, N), dtype=np.float64)

    # alpha factor for the inverse gamma posterior over phi
    A0 = np.ones(K) * pseudocounts
    A = A0.copy()

    # beta factor for the inverse gamma posterior over phi.
    # Initially, we set this to the beta parameter corresponding
    # to the max a priori value of phi
    B0 = phi * A
    B = B0.copy()

    ## INITIALIZATION

    # We don't know the state occupations in advance, so we'll
    # guess the distribution of initial state assignments without
    # incorporating any knowledge of them.
    exp_log_phi = np.log(B) - digamma(A)
    exp_inv_phi = A / B
    for j in range(K):
        r[j, :] = -sum_r2 * exp_inv_phi[j] - n_jumps * exp_log_phi[j]

    # Normalize
    r = r - r.max(axis=0)
    r = np.exp(r)
    r = r / r.sum(axis=0)

    # Calculate the initial posterior estimate over the mixing
    # coefficients (via the Dirichlet parameter *n*) and the
    # phi values (via the inverse gamma parameters *A* and *B*)
    nr = (r * n_jumps).sum(axis=1)
    sr = (r * sum_r2).sum(axis=1)

    # Initial posterior estimates
    n = n0 + nr
    A = A0 + nr
    B = B0 + sr

    # Correct for defocalization
    if consider_defoc:
        n = corr_defoc(A, B, n)

    ## CORE REFINEMENT

    prev_n = n.copy()
    for iter_idx in range(max_iter):
        # Expectation of log(occs[j]) under the
        # current model, size K
        exp_log_occs = digamma(n) - digamma(n.sum())

        # Expectation of log(phi[j]) under the
        # current model
        exp_log_phi = np.log(B) - digamma(A)

        # Expectation of 1/phi[j] under the current
        # model
        exp_inv_phi = A / B

        # Determine the expectations of the state assignment
        # vector, given the current model
        for j in range(K):
            r[j, :] = (
                exp_log_occs[j] - sum_r2 * exp_inv_phi[j] - n_jumps * exp_log_phi[j]
            )
        r = r - r.max(axis=0)
        r = np.exp(r)

        # Normalize over the phi values for each trajectory
        r = r / r.sum(axis=0)

        # Scale to the number of jumps (size K)
        nr = (r * n_jumps).sum(axis=1)

        # Scale to the sum of squared displacements (size K)
        sr = (r * sum_r2).sum(axis=1)

        # Posterior over state occupations (parameter for a Dirichlet distribution)
        n = n0 + nr

        # Correct for defocalization, if desired
        if consider_defoc:
            n = corr_defoc(A, B, n)

        # Posterior over phi (alpha and beta parameters for an inverse gamma distribution)
        A = A0 + nr
        B = B0 + sr

        # Call convergence
        delta = np.abs(n - prev_n)
        if (delta < convergence).all():
            break
        else:
            prev_n = n

    # Evaluate the variational lower bound for the model evidence
    # under the posterior distribution
    exp_log_occs = digamma(n) - digamma(n.sum())
    exp_log_phi = np.log(B) - digamma(A)
    exp_inv_phi = A / B
    elbo = calc_elbo(
        sum_r2, n_jumps, r, n, A, B, n0, A0, B0, exp_log_occs, exp_inv_phi, exp_log_phi
    )

    # Mean occupation for each state under the posterior distribution
    occs = n / n.sum()

    # Mean diffusion coefficients under the posterior distribution
    phi_mean = B / (A - 1)
    D_mean = (phi_mean / 4.0 - le2) / frame_interval

    # Arrange the states by increasing diffusion coefficient
    order = np.argsort(D_mean)
    occs = occs[order]
    D_mean = D_mean[order]
    A = A[order]
    B = B[order]
    n = n[order]
    r = r[order, :]

    # Set negative values for the diffusion coefficient to zero,
    # if desired
    if not allow_neg_diff_coef:
        D_mean[D_mean < 0] = 0

    # Return the parameters for the mean field approximation to the posterior
    # distribution
    if return_posterior:
        return r, n, A, B, occs, D_mean, elbo
    else:
        return occs, D_mean


def calc_elbo(
    sum_r2,
    n_jumps,
    r,
    n,
    A,
    B,
    n0,
    A0,
    B0,
    exp_log_occs,
    exp_inv_phi,
    exp_log_phi,
    identifiability_corr=True,
) -> dict:
    """
    Calculate the variational evidence lower bound for
    a particular mixture model. For more details on this
    calculation, see `emdiff/doc/vbdiff.pdf`.

    args
    ----
        Most of the arguments are those produced internally
        by the vbdiff algorithm.

        identifiability_corr        :   bool, apply a
            correction for the identifiability deflation
            in the ELBO, which originates from the
            label switching problem

    returns
    -------
    dict with keys
        "elbo": actual evidence lower bound,
        "A": E[log p(X|Z,tau,phi)]
        "B": E[log p(Z|tau)]
        "C": E[log p(tau)]
        "D": E[log p(phi)]
        "E": E[log q(Z)]
        "F": E[log q(tau)]
        "G": E[log q(phi)]
    """
    K, N = r.shape

    # Due to the model likelihood (increases with higher K; large magnitude)
    _evA = np.zeros(r.shape)
    for j in range(K):
        _evA[j, :] = (
            (n_jumps - 1) * np.log(sum_r2)
            - exp_inv_phi[j] * sum_r2
            - loggamma(n_jumps)
            - exp_log_phi[j] * n_jumps
        )
    evA = (r * _evA).sum()

    # Due to the prior over state assignments (decreases with higher K; large magnitude)
    evB = (exp_log_occs * r.sum(axis=1)).sum()

    # Due to the prior over mixing coefficients (decreases with higher K; low magnitude)
    evC = (n0[0] - 1) * exp_log_occs.sum() - logbeta(*n0)

    # Due to the prior over phi (increases with higher K; low magnitude)
    evD = (A0 * np.log(B0) - loggamma(A0) - B0 * A / B - (A0 + 1) * exp_log_phi).sum()

    # Due to the posterior over state assignments (decreases with higher K; high magnitude)
    r_adj = r + 1e-8
    evE = (r_adj * np.log(r_adj)).sum()

    # Due to the posterior over mixing coefficients (increases with higher K; low magnitude)
    evF = ((n - 1) * exp_log_occs).sum() - logbeta(*n)

    # Due to the posterior over phi (increases with higher K; low magnitude)
    evG = (A * np.log(B) - A - loggamma(A) - (A + 1) * exp_log_phi).sum()

    # Lower bound
    elbo = evA + evB + evC + evD - evE - evF - evG

    # For checking numerical accuracy
    if False:
        print(f"K = {K}")
        print(f"evA:\t{evA:.3f}")
        print(f"evB:\t{evB:.3f}")
        print(f"evC:\t{evC:.3f}")
        print(f"evD:\t{evD:.3f}")
        print(f"evE:\t{evE:.3f}")
        print(f"evF:\t{evF:.3f}")
        print(f"evG:\t{evG:.3f}")

    # Apply an identifiability correction (usually small in magnitude
    # compare to the other terms)
    if identifiability_corr:
        elbo += loggamma(K + 1)

    return {
        "elbo": elbo,
        "A": evA,
        "B": evB,
        "C": evC,
        "D": evD,
        "E": evE,
        "F": evF,
        "G": evG,
    }

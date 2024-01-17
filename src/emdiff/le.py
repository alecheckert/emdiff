"""
le.py -- estimate localization error in a
set of trajectories using the covariance 
between subsequent jumps

"""
import numpy as np
from .utils import track_length


def calc_le(
    tracks,
    pixel_size_um=0.16,
    pos_cols=["y", "x"],
    frame_interval=0.01,
    method="covariance",
):
    """
    Calculate localization error using the negative covariance
    between subsequent jumps in a trajectory.

    principle
    ---------
        Consider every three points in a trajectory originating
        from a Markov process:

            (point) A         B         C
                    .  -----> .  -----> .
            (error) e1        e2        e3

        The Markov process condition means that the real underlying
        jump from A to B is independent of the jump from B to C.

        However, each point has associated with it some localization
        error, which is assumed to be independent for each point.
        In this way, the measured vector A->B is equal to the true
        jump length plus e1 plus e2. Likewise, the measured vector
        B->C is equal to the true jump from B to C plus e2 plus e3.

        The shared dependence of the measured jumps on e2 induces
        a negative covariance between the measured jump lengths. This
        covariance is negative with absolute magnitude equal to
        the variance involved in estimating the second point's position
        (that is, the variance of e2).

        This provides an experimental route to measuring the
        localization error for moving particles when the motion is
        a Markov process. If the motion is not a Markov process,
        this won't work.

    args
    ----
        tracks          :   pandas.DataFrame with columns 'trajectory',
                                'frame', and the contents of *pos_cols*
        pixel_size_um   :   float, pixel size in um
        pos_cols        :   list of str, the positional information
                                for the trajectory
        method          :   str, either "covariance" or "msd"

    returns
    -------
        float, 1D localization root variance in um

    """
    # Exclude trajectories with fewer than three points
    tracks = track_length(tracks)
    tracks = tracks[tracks["track_length"] >= 3]

    # Sort first by trajectory, then frame
    tracks = tracks.sort_values(by=["trajectory", "frame"])

    # Convert to ndarray and from pixels to um
    cols = ["trajectory", "frame"] + list(pos_cols)
    T = np.asarray(tracks[cols]) * pixel_size_um

    if method == "covariance":
        # Calculate jumps
        D = T[1:, :] - T[:-1, :]

        # Product between subsequent jumps
        P = D[1:, :] * D[:-1, :]

        # Sum of the differential trajectory indices for subsequent
        # jumps. This is 0 if the two jumps originate from the same
        # trajectory, and greater than 0 otherwise.
        sum_indices = D[1:, 0] + D[:-1, 0]

        # Only consider jumps that originate from the same trajectory
        same_track = sum_indices == 0

        # Only consider jumps over a single frame interval
        gap_one = np.logical_and(D[1:, 1] == 1, D[:-1, 1] == 1)

        # Take the covariance in the Y and X directions
        take = np.logical_and(same_track, gap_one)
        jump_cov = np.minimum(P[take, 2:].mean(axis=0), 0)

        return np.sqrt(-jump_cov).mean()

    elif method == "msd":
        # Calculate mean squared displacement
        M = len(pos_cols)
        msd = np.zeros(4)
        for j in range(1, msd.shape[0] + 1):
            jumps = T[j:, :] - T[:-j, :]
            jumps = jumps[np.logical_and(jumps[:, 0] == 0, jumps[:, 1] == j), :]
            msd[j - 1] = (jumps[:, 2:] ** 2).mean()

        # Fit to linear model
        X = np.zeros((msd.shape[0], 2))
        X[:, 0] = 2 * M * frame_interval * np.arange(1, msd.shape[0] + 1)
        X[:, 1] = 2 * M
        fit_pars = np.linalg.inv(X.T @ X) @ (X.T @ msd)

        return np.sqrt(np.maximum(fit_pars[1], 0))

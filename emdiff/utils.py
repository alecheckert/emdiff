"""
utils.py

"""
import numpy as np
import pandas as pd
from scipy.special import gamma, gammainc
from .defoc import f_remain 

def drop_col(df, col):
    """
    If a pandas.DataFrame has a particular column, drop it.

    args
    ----
        df      :   pandas.DataFrame
        col     :   str

    returns
    -------
        pandas.DataFrame, same DataFrame without column
    
    """
    if col in df.columns:
        df = df.drop(col, axis=1)
    return df

def track_length(tracks):
    """
    Calculate the length in frames of each trajectory.

    args
    ----
        tracks      :   pandas.DataFrame, with columns
                        "trajectory" and "frame"

    returns
    -------
        pandas.DataFrame, the same set of trajectories
            with a new column, "track_length"

    """
    tracks = drop_col(tracks, "track_length")
    return tracks.join(
        tracks.groupby("trajectory").size().rename("track_length"),
        on="trajectory"
    )

def assign_index_in_track(tracks):
    """
    For every row of a DataFrame, assign a number that indicates
    its index in the context of the corresponding trajectory.

    args
    ----
        tracks      :   pandas.DataFrame with column "trajectory"

    returns
    -------
        same DataFrame with "index_in_track" column, 
            starts at 1 for each trajectory

    """
    tracks = drop_col(tracks, "index_in_track")
    tracks["one"] = 1
    tracks["index_in_track"] = tracks.grouby("trajectory")["one"].cumsum()
    tracks = drop_col(tracks, "one")
    return tracks

def squared_jumps(tracks, pixel_size_um=0.16, pos_cols=['y', 'x']):
    """
    Calculate every squared radial displacement over a single
    frame interval present in a set of trajectories.

    args
    ----
        tracks          :   pandas.DataFrame, with columns 
                            "trajectory", "frame", and the 
                            contents of *pos_cols*
        pixel_size_um   :   float, size of pixels in um
        pos_cols        :   list of str, the spatial coordinates

    returns
    -------
        (
            1D ndarray of shape (n_tracks,), the sum of
                squared radial displacements for each 
                trajectory in um^2;
            1D ndarray of shape (n_tracks,), the corresponding
                trajectory indices;
            1D ndarray of shape (n_tracks,), the length of 
                each trajectory in frames
        )

    """
    # Remove singlets
    tracks = track_length(tracks)
    tracks = tracks[tracks["track_length"] > 1].copy()

    # Sort first by trajectory, then by frame
    tracks = tracks.sort_values(by=["trajectory", "frame"])

    # Convert to ndarray
    cols = ["trajectory", "frame", "track_length"] + list(pos_cols)
    T = np.asarray(tracks[cols])

    # Convert from pixels to um
    T[:,3:] *= pixel_size_um 

    # Calculate displacements
    D = T[1:,:] - T[:-1,:]

    # Only consider vectors between points originating 
    # from the same trajectory and one frame interval
    same_track = D[:,0] == 0
    one_interval = D[:,1] == 1
    take = np.logical_and(same_track, one_interval)

    # Calculate the corresponding squared radial displacements
    r2 = (D[take,3:]**2).sum(axis=1)

    return r2, T[:-1,0][take].astype(np.int64), T[:-1,2][take]

def sum_squared_jumps(tracks, pixel_size_um=0.16, 
    pos_cols=["y", "x"], max_jumps_per_track=None):
    """
    Calculate the sum of squared radial displacements
    corresponding to each trajectory in a dataset.
    Only include displacements over single frame intervals.

    args
    ----
        tracks              :   pandas.DataFrame
        pixel_size_um       :   float, size of pixels in um
        pos_cols            :   list of str, position coordinates
        max_jumps_per_track :   int, the maximum number of 
                                jumps to include from each trajectory

    returns
    -------
        pandas.DataFrame with columns ["trajectory", "sum_r2",
            "n_jumps"]

    """
    # Calculate squared displacements
    r2, track_indices, track_lengths = squared_jumps(
        tracks, pixel_size_um=pixel_size_um, pos_cols=pos_cols)
    n_jumps = r2.shape[0]

    # Convert to DataFrame
    df = pd.DataFrame(index=np.arange(n_jumps),
        columns=["r2", "trajectory", "track_length"])
    df["r2"] = r2
    df["trajectory"] = track_indices
    df["track_length"] = track_lengths

    # Truncate trajectories if desired
    if (not max_jumps_per_track is None) and \
        (not max_jumps_per_track is np.inf):

        df = assign_index_in_track(df)
        df = df[df["index_in_track"] < max_jumps_per_track]

    # Calculate the number of displacements corresponding
    # to each trajectory
    df = df.join(
        df.groupby("trajectory").size().rename("n_jumps"),
        on="trajectory"
    )

    # Aggregate
    n_tracks = df["trajectory"].nunique()
    L = pd.DataFrame(index=np.arange(n_tracks),
        columns=["sum_r2", "trajectory", "n_jumps"])
    L["sum_r2"] = np.asarray(df.groupby("trajectory")["r2"].sum())
    L["trajectory"] = np.asarray(df.groupby("trajectory")["trajectory"].first())
    L["n_jumps"] = np.asarray(df.groupby("trajectory")["n_jumps"].first())

    return L 

#########################
## UTILITIES FOR PLOTS ##
#########################

def rad_disp_histogram(tracks, n_frames=4, pos_cols=["y", "x"], bin_size=0.001, 
    max_jump=5.0, pixel_size_um=0.160, n_gaps=0, use_entire_track=False,
    max_jumps_per_track=10):
    """
    Compile a histogram of radial displacements for a set of trajectories.

    args
    ----
        tracks          :   pandas.DataFrame
        n_frames        :   int, the number of frame delays to consider.
                            A separate histogram is compiled for each
                            frame delay.
        pos_cols        :   list of str, the spatial coordinate columns
        bin_size        :   float, the size of the bins in um. For typical
                            experiments, this should not be changed because
                            some diffusion models (e.g. Levy flights) are 
                            contingent on the default binning parameters.
        max_jump        :   float, the max radial displacement to consider in 
                            um
        pixel_size_um   :   float, the size of individual pixels in um
        n_gaps          :   int, the number of gaps allowed during tracking
        use_entire_track:   bool, use every displacement in the dataset
        max_jumps_per_track:   int, the maximum number of displacements
                            to consider per trajectory. Ignored if 
                            *use_entire_track* is *True*.

    returns
    -------
        (
            2D ndarray of shape (n_frames, n_bins), the distribution of 
                displacements at each time point;
            1D ndarray of shape (n_bins+1), the edges of each bin in um
        )

    """
    # Sort by trajectory, then frame
    tracks = tracks.sort_values(by=["trajectory", "frame"])

    # Assign track lengths
    if "track_length" not in tracks.columns:
        tracks = track_length(tracks)

    # Filter out unassigned localizations and singlets
    cols = ["frame", "trajectory"] + list(pos_cols)
    T = tracks[
        np.logical_and(tracks["trajectory"]>=0, tracks["track_length"]>1)
    ][cols]

    # Convert to ndarray for speed
    convert_cols = ["frame", "trajectory", "trajectory"] + list(pos_cols)
    T = np.asarray(T[convert_cols]).astype(np.float64)

    # Sort first by track, then by frame
    T = T[np.lexsort((T[:,0], T[:,1])), :]

    # Convert from pixels to um
    T[:,3:] = T[:,3:] * pixel_size_um 

    # Format output histogram
    bin_edges = np.arange(0.0, max_jump+bin_size, bin_size)
    n_bins = bin_edges.shape[0] - 1
    H = np.zeros((n_frames, n_bins), dtype=np.int64)

    # Consider gap frames
    if n_gaps > 0:

        # The maximum trajectory length, including gap frames
        max_len = (n_gaps + 1) * n_frames + 1

        # Consider every shift up to the maximum trajectory length
        for l in range(1, max_len+1):

            # Compute the displacement for all possible jumps
            diff = T[l:,:] - T[:-l,:]

            # Map the trajectory index corresponding to the first point in 
            # each trajectory
            diff[:,2] = T[l:,1]

            # Only consider vectors between points originating from the same track
            diff = diff[diff[:,1] == 0.0, :]

            # Look for jumps corresponding to each frame interval being considered
            for t in range(1, n_frames+1):

                # Find vectors that match the delay being considered
                subdiff = diff[diff[:,0] == t, :]

                # Only consider a finite number of displacements from each trajectory
                if not use_entire_track:
                    _df = pd.DataFrame(subdiff[:,2], columns=["traj"])
                    _df["ones"] = 1
                    _df["index_in_track"] = _df.groupby("traj")["ones"].cumsum() 
                    subdiff = subdiff[np.asarray(_df["index_in_track"]) <= max_jumps_per_track, :]

                # Calculate radial displacements
                r_disps = np.sqrt((subdiff[:,3:]**2).sum(axis=1))
                H[t-1,:] = H[t-1,:] + np.histogram(r_disps, bins=bin_edges)[0]

    # No gap frames
    else:

        # For each frame interval and each track, calculate the vector change in position
        for t in range(1, n_frames+1):
            diff = T[t:,:] - T[:-t,:]

            # Map trajectory indices back to the first localization of each trajectory
            diff[:,2] = T[t:,1]

            # Only consider vectors between points originating in the same track
            diff = diff[diff[:,1] == 0.0, :]

            # Only consider vectors that match the delay being considered
            diff = diff[diff[:,0] == t, :]

            # Only consider a finite number of displacements from each trajectory
            if not use_entire_track:
                _df = pd.DataFrame(diff[:,2], columns=["traj"])
                _df["ones"] = 1
                _df["index_in_track"] = _df.groupby("traj")["ones"].cumsum()
                diff = diff[np.asarray(_df["index_in_track"]) <= max_jumps_per_track, :]

            # Calculate radial displacements
            r_disps = np.sqrt((diff[:,3:]**2).sum(axis=1))
            H[t-1,:] = np.histogram(r_disps, bins=bin_edges)[0]

    return H, bin_edges

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

def coarsen_histogram(jump_length_histo, bin_edges, factor):
    """
    Given a jump length histogram with many small bins, aggregate into a 
    histogram with a small number of larger bins.

    args
    ----
        jump_length_histo       :   2D ndarray, the jump length histograms
                                    indexed by (frame interval, jump length bin)
        bin_edges               :   1D ndarray, the edges of each jump length
                                    bin in *jump_length_histo*
        factor                  :   int, the number of bins in the old histogram
                                    to aggregate for each bin of the new histogram

    returns
    -------
        (
            2D ndarray, the aggregated histogram,
            1D ndarray, the edges of each jump length bin the aggregated histogram
        )

    """
    # Get the new set of bin edges
    n_frames, n_bins_orig = jump_length_histo.shape 
    bin_edges_new = bin_edges[::factor]
    n_bins_new = bin_edges_new.shape[0] - 1

    # May need to truncate the histogram at the very end, if *factor* doesn't
    # go cleanly into the number of bins in the original histogram
    H_old = jump_length_histo[:, (bin_edges<bin_edges_new[-1])[:-1]]

    # Aggregate the histogram
    H = np.zeros((n_frames, n_bins_new), dtype=jump_length_histo.dtype)
    for j in range(factor):
        H = H + H_old[:, j::factor]

    return H, bin_edges_new 

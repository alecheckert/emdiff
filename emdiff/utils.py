"""
utils.py

"""
import numpy as np
import pandas as pd

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


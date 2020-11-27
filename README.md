# emdiff
Maximum likelihood and variational Bayes estimation for
mixtures of regular Brownian motions

## What does it do?

`emdiff` is a Python analysis tool for trajectories generated by 
stroboscopic single particle tracking (SPT) experiments.

It takes a set of trajectories in CSV format, then returns the 
estimated fraction of molecules in distinct states and the 
diffusion coefficient corresponding to each state. 

There are actually two commands in `emdiff`:

 - `emdiff` is an expectation-maximization routine that only 
    estimates the maximum likelihood state occupancies and 
    diffusion coefficients of each state.
 - `vbdiff` is a variational Bayes treatment of the same problem.
    It estimates the full posterior distribution over 
    occupancies, diffusion coefficients, and the assignment of 
    each trajectory to a state.

`emdiff` is simpler, but `vbdiff` is highly recommended. `vbdiff` performs
better when the number of states in the model doesn't match the true
number of states in the data. Both run at about the same speed.

## What doesn't it do?

`emdiff` and `vbdiff` do not check the quality of your raw data.

`emdiff` and `vbdiff` do not filter out any data. They will consider whatever
data you give them. There is a single optional filter to exclude
trajectories before a start frame, but that's it.

`emdiff` does not decide how many diffusive states are present
in your data. You decide that.

`emdiff` and `vbdiff` do not consider transitions between states. For that,
we highly recommend [vbSPT](http://vbspt.sourceforge.net/ref/persson-nmeth-2013.pdf)
from the Elf group.

`emdiff` and `vbdiff` do not provide any visualizations of the result beyond
some simple histograms provided by the `plot` option to `emdiff`.

`emdiff` and `vbdiff` expect you to understand your own experiment. You need to
know the frame interval, approximate localization error, and pixel size.

`emdiff` and `vbdiff` only launch the estimator from a single initial parameter
guess. They do not identify when the estimator has converged to a 
local rather than global maximum. The user can set the initial guess
via the `guess` argument.

`emdiff` and `vbdiff` do not deal with identifiability issues arising from 
interchangeability of the states. For instance, if we fit 
with `n` states, there will be at least `n!` maxima in parameter space
that come from permutations of the state identities. `emdiff` and `vbdiff` 
will simply converge to one of these maxima, then order the output
by increasing diffusion coefficient.

## Is there a limit on the number of states I can fit?

No. `emdiff` and `vbdiff` place no limit on the number of states.

Of course, more states mean more error involved in the estimation.

## Is there a limit on the number of spatial dimensions?

No, with one fat caveat. `emdiff` and `vbdiff` support analysis in any
number of spatial dimensions. Exactly how many is set by the `pos_cols`
argument, which tells the tools where to look for the spatial coordinates.
See the ``Example usage'' section below.

The fat caveat is that both tools model localization error
as identical along each spatial dimension. This is generally true
for 1D and 2D analyses. But for most SPT experiments in 3D, tracking
the position of moving objects in the axial (Z) direction is harder
than in the lateral (XY) directions. So the assumption of identical
localization error along each axis will be wrong for those
setups.

## Dependencies

`numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`. These are all
part of the standard [Anaconda](https://www.anaconda.com/products/individual) distribution. If you have 
[`matplotlib_scalebar`](https://pypi.org/project/matplotlib-scalebar/), 
the plots will have scalebars. So that's recommended too.

## Install

1. Clone the repository: `git clone https://github.com/alecheckert/emdiff.git`. 

2. Navigate to the top-level `emdiff` directory.

3. From a `conda` environment with the dependencies above, run `python setup.py develop`. 

`emdiff` is in active development. The `develop` option 
will track changes in the source files as new versions become available.
To pull new versions, navigate to the `emdiff` repository and use
`git pull`. 

## Expected input

Both `emdiff` and `vbdiff` take trajectories as a `pandas.DataFrame`. Each row of 
the dataframe should represent one detection from an SPT experiment.
The dataframe must contain at minimum:

 - a `trajectory` column, with the index of the trajectory to which that localization corresponds;
 - a `frame` column, with the index of the corresponding frame;
 - columns such as `y` and `x` with the spatial coordinates of the localization. **These must be specified in pixels.**

See `samples/sample_tracks.csv` for a sample input.

## Example usage of `emdiff`

```
    from emdiff import emdiff
    state_occs, diff_coefs = emdiff(
        tracks,
        n_states=2,
        pos_cols=["y", "x"],     # columns in *tracks* with 
                                 # the spatial coordinates
        pixel_size_um=0.16,      # microns
        frame_interval=0.00748,  # seconds
        loc_error=0.035          # microns
    )
```

`state_occs[j]` and `diff_coefs[j]` are the fractional occupancy
and diffusion coefficient (in microns squared per second)
of the jth state.

The states are always ordered in terms of increasing
diffusion coefficient.

You can find another example at `samples/sample_script.py`.

## Example usage of `vbdiff`

```
    from emdiff import vbdiff
    state_occs_mean, diff_coefs_mean = vbdiff(
        tracks,
        n_states=2,
        pos_cols=["y", "x"],     # columns in *tracks* with 
                                 # the spatial coordinates
        pixel_size_um=0.16,      # microns
        frame_interval=0.00748,  # seconds
        loc_error=0.035          # microns
    )
```

`state_occs_mean[j]` and `diff_coefs_mean[j]` are the mean fractional 
occupancy and mean diffusion coefficient for the jth state under the 
posterior model. 

The states are always ordered in terms
of increasing diffusion coefficient. 

## Can I use `vbdiff` to get the full posterior distribution, rather than just the posterior mean?

Yes. This can be useful, for instance, if you want the probability that each 
trajectory is in each diffusive state. Use the `return_posterior` argument:

```
    from emdiff import vbdiff
    r, n, A, B, occs_mean, diff_coefs_mean, elbo, likelihood = vbdiff(
        tracks,
        n_states=2,
        pos_cols=["y", "x"],     # columns in *tracks* with 
                                 # the spatial coordinates
        pixel_size_um=0.16,      # microns
        frame_interval=0.00748,  # seconds
        loc_error=0.035,         # microns
        return_posterior=True
    )
```

As you can see, we get some more outputs. What do these mean?

 - `r` is a 2D `ndarray` of shape `(n_states, n_tracks)`, where `r[j,i]` 
    is the posterior probability for trajectory `i` to inhabit diffusive state `j`.
 - `n` is a 1D `ndarray` of shape `(n_states,)`. This is the parameter for 
    the posterior (Dirichlet) distribution over state occupancies.
 - `A` and `B` are the parameters for the posterior (inverse gamma) distribution
    over the spatial variance of each state. Both are 1D `ndarray`s, where 
    `A[j]`, `B[j]` are the parameters for the jth state. The spatial variance
    is defined as `4 * (D * frame_interval + loc_error**2)`.
 - `elbo` is the evidence lower bound for the posterior model.
 - `likelihood` is the likelihood of the posterior model given the data. 

## Example usage of `emdiff` with plots
```
    from emdiff import emdiff

    # The output prefix for the plots, which are saved as PNGs
    plot_prefix = "my_dataset_name_emdiff_fits"

    # Run the core EM routine
    state_occupations, diff_coefs = emdiff(
        tracks,
        n_states=2,
        pos_cols=["y", "x"],
        pixel_size_um=0.16,
        frame_interval=0.00748,
        loc_error=0.035,
        plot=True,
        plot_prefix=plot_prefix
    )
```

## What do I do with the result?

Up to you.

## But I don't know the localization error for my experiment.

There are various ways to estimate the localization
error in an SPT experiment. `emdiff` provides one:
```
    from emdiff import calc_le

    # Use the covariance method to calculate localization error
    loc_error = calc_le(
        tracks, 
        pixel_size_um=0.16,
        pos_cols=['y', 'x'],
        method="covariance"
    )

    # Use the MSD method to calculate localization error
    loc_error = calc_le(
        tracks, 
        pixel_size_um=0.16,
        pos_cols=['y', 'x'],
        method="msd"
    )

```

The localization error is defined as the root 1D variance associated
with estimating a particle's position at any given frame. The two
methods (`covariance` and `msd`) are two different ways to go about
estimating it.

If this localization error seems large, probably that's
because it actually is. Moving molecules produce a lot more error
than in traditional fixed cell PALM/STORM experiments where all 
the molecules are stationary.

## Does `emdiff` have a defocalization correction similar to [Spot-On](https://gitlab.com/tjian-darzacq-lab/Spot-On)?

Yes. Use the `dz` argument to `emdiff` to set the microscope's focal
depth in microns.

For example:
```
    from emdiff import vbdiff
    state_occs_mean, diff_coefs_mean = vbdiff(
        tracks,
        n_states=2,
        pos_cols=["y", "x"],     # columns in *tracks* with 
                                 # the spatial coordinates
        pixel_size_um=0.16,      # microns
        frame_interval=0.00748,  # seconds
        loc_error=0.035,         # microns
        dz=0.7                   # depth of field in microns
    )
```


## Where can I find a description of the parameters to `emdiff`?

For now, see the docstrings to the functions `emdiff.em.emdiff` and
`emdiff.vb.vbdiff`. 
In the future, you'll find a user guide in the `doc` folder to 
this repository.

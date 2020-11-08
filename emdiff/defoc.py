"""
defoc.py

"""
import numpy as np

def f_remain(D, n_frames, frame_interval, dz, mode="inside"):
    """
    Calculate the fraction of Brownian particles that 
    remain in a microscope's depth of field after a
    few time points.

    Specifically:

    Generate a Brownian motion with diffusion coefficient
    *D* at a random position between [-dz/2 and dz/2]. 
    The particle is observed at time 0 and then at regular
    intervals after that. The time between each interval
    is *frame_interval*.

    If the particle lies outside the range [-dz/2, dz/2]
    for a single frame interval, it is counted ``lost''.

    This function calculates the probability that such a
    particle is NOT lost at each frame.

    args
    ----
        D               :   float, diffusion coefficient
                            in um^2 s^-1
        n_frames        :   int, the number of frames
        frame_interval  :   float, seconds
        dz              :   float, depth of field in um

    returns
    -------
        1D ndarray of shape (n_frames,), the probability
            to remain at each frame interval

    """
    if (dz is np.inf) or (dz is None):
        return np.ones(n_frames, dtype=np.float64)

    # Support for the calculations
    s = (int(dz//2.0)+1) * 2
    support = np.linspace(-s, s, int(((2*s)//0.001)+2))[:-1]
    hz = 0.5 * dz 
    inside = np.abs(support) <= hz 
    outside = ~inside 

    # Define the transfer function for this BM
    g = np.exp(-(support ** 2)/ (4 * D * frame_interval))
    g /= g.sum()
    g_rft = np.fft.rfft(g)   

    # Set up the initial probability density
    if mode == "inside":
        pmf = inside.astype(np.float64)
        pmf /= pmf.sum()
    elif mode == "outside":
        pmf = outside.astype(np.float64)
        pmf /= pmf.sum()
        pmf = np.fft.fftshift(np.fft.irfft(
            np.fft.rfft(pmf) * g_rft, n=pmf.shape[0]))
        pmf[outside] = 0.0
        pmf /= pmf.sum()

    # Propagate over subsequent frame intervals
    result = np.zeros(n_frames, dtype=np.float64)
    for t in range(n_frames):
        pmf = np.fft.fftshift(np.fft.irfft(
            np.fft.rfft(pmf) * g_rft, n=pmf.shape[0]))
        pmf[outside] = 0.0
        result[t] = pmf.sum()

    return result 



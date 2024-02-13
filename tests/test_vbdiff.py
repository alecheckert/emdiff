import numpy as np
import pandas as pd
from pathlib import Path
from unittest import TestCase
from emdiff.vb import vbdiff


FIXTURES = Path(__file__).absolute().parent / "fixtures"


class TestVBDiff(TestCase):
    def test_end_to_end(self):
        target = FIXTURES / "sample_tracks.zip"
        assert target.is_file()
        spots = pd.read_csv(str(target))
        n_states = 5
        occs, diff_coefs = vbdiff(
            spots,
            n_states=n_states,  # number of diffusive states
            pixel_size_um=0.16,  # microns
            frame_interval=0.00748,  # seconds
            loc_error=0.035,  # microns
            dz=0.7,  # focal depth, microns
        )
        assert len(occs) == len(diff_coefs)
        assert len(occs) == n_states
        assert np.isfinite(occs).all()
        assert np.isfinite(diff_coefs).all()
        assert (occs >= 0.0).all()
        assert (diff_coefs >= 0.0).all()
        assert abs(occs.sum() - 1.0) < 1e-5

    def test_empty(self):
        spots = pd.DataFrame(
            {
                "x": np.zeros(0, dtype=np.float64),
                "y": np.zeros(0, dtype=np.float64),
                "trajectory": np.zeros(0, dtype=np.int64),
                "frame": np.zeros(0, dtype=np.int64),
            }
        )
        n_states = 2
        with self.assertRaises(RuntimeError):
            occs, diff_coefs = vbdiff(
                spots,
                n_states=n_states,
                pixel_size_um=0.16,
                frame_interval=0.00748,
                loc_error=0.035,
                dz=0.7,
            )

    def test_retries(self):
        """Multiple runs with different initial guesses
        should return the same answer if we call np.random.seed
        in advance."""
        target = FIXTURES / "sample_tracks.zip"
        assert target.is_file()
        spots = pd.read_csv(str(target))
        n_states = 7
        retries = 5
        np.random.seed(1)
        occs1, diffcoefs1 = vbdiff(
            spots,
            n_states=n_states,  # number of diffusive states
            pixel_size_um=0.16,  # microns
            frame_interval=0.00748,  # seconds
            loc_error=0.035,  # microns
            dz=0.7,  # focal depth, microns
            max_iter=10,
            retries=retries,
        )
        np.random.seed(1)
        occs2, diffcoefs2 = vbdiff(
            spots,
            n_states=n_states,  # number of diffusive states
            pixel_size_um=0.16,  # microns
            frame_interval=0.00748,  # seconds
            loc_error=0.035,  # microns
            dz=0.7,  # focal depth, microns
            max_iter=10,
            retries=retries,
        )
        np.testing.assert_allclose(occs1, occs2, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(diffcoefs1, diffcoefs2, atol=1e-6, rtol=1e-6)

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

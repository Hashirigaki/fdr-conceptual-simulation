"""Fast structural and deterministic smoke tests for the conceptual simulation."""
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import fdr_v0_5_showcase as fdr


class FDRSmokeTests(unittest.TestCase):
    def test_stationary_minima_and_gradient(self):
        self.assertTrue(fdr.check_model())

    def test_barrier_order(self):
        heights = [fdr.trench_barrier(a) for a in fdr.BARRIER_LEVELS.values()]
        self.assertTrue(all(x < y for x, y in zip(heights, heights[1:])))

    def test_readout_partition(self):
        x = np.vstack((fdr.BASIN_CENTERS, np.array([[99.0, 99.0]])))
        actual = fdr.fractions(x)
        np.testing.assert_allclose(actual, np.array([.25, .25, .25, .25]))
        self.assertAlmostEqual(float(np.sum(actual)), 1.0)

    def test_seed_reproduces_trajectories(self):
        t1, a, _, _ = fdr.simulate_suite()
        t2, b, _, _ = fdr.simulate_suite()
        np.testing.assert_array_equal(t1, t2)
        for scenario in fdr.SCENARIOS:
            for barrier in fdr.BARRIER_LEVELS:
                np.testing.assert_array_equal(a[scenario][barrier], b[scenario][barrier])


if __name__ == "__main__":
    unittest.main()

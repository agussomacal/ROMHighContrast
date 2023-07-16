import unittest

import numpy as np

from lib.SolutionsManagers import SolutionsManagerFEM


class TestSolutionsManager(unittest.TestCase):
    def setUp(self) -> None:
        self.sm = SolutionsManagerFEM(blocks_geometry=(2, 2), N=10, num_cores=1, method="lsq")

    def test_riesz(self):
        assert np.shape(self.sm.generate_riesz([(0, 0)], norm="l2")) == (1, self.sm.vspace_dim)
        assert np.shape(self.sm.generate_riesz([(0, 0)], norm="h10")) == (1, self.sm.vspace_dim)
        assert np.shape(self.sm.generate_riesz([(0, 0), (1, 1)], norm="h10")) == (2, self.sm.vspace_dim)

import unittest

from lib.SolutionsManagers import SolutionsManagerFEM


class TestSolutionsManager(unittest.TestCase):
    def setUp(self) -> None:
        self.sm = SolutionsManagerFEM(blocks_geometry=(2, 2), N=10, num_cores=1, method="lsq")

    def test_riesz(self):
        self.sm.generate_riesz([(0, 0)], norm="l2")
        self.sm.generate_riesz([(0, 0)], norm="h10")

import unittest

import numpy as np

from src.lib.Functions.PolynomialFunctions import PolynomialFunctions
from src.lib.Functions.RegionsFunctionsDefault import RegionsFunctionsDefault
from src.lib.Regions.Regions import Regions, RectangularRegion


class TestRegionsFunctionsDefault(unittest.TestCase):
    def setUp(self) -> None:
        self.coeffs_1d = np.ones(2)
        self.coeffs_2d = np.ones((2, 2))

        regions = Regions([
            RectangularRegion(lower_limits=[0], upper_limits=[0.5]),
            RectangularRegion(lower_limits=[0.5], upper_limits=[1])
        ])
        self.rfd_1d = RegionsFunctionsDefault(
            regions=regions,
            functions=[
                PolynomialFunctions(coefficients=self.coeffs_1d),
                PolynomialFunctions(coefficients=self.coeffs_1d + 1)
            ]
        )
        # TODO: test 2d

    def test_evaluate(self):
        assert self.rfd_1d.evaluate([0]) == 1
        assert self.rfd_1d.evaluate([1]) == 4

    def test_add(self):
        assert (self.rfd_1d + self.rfd_1d).evaluate([0]) == 2
        assert (self.rfd_1d + self.rfd_1d).evaluate([1]) == 8

    def test_mult(self):
        assert (self.rfd_1d * self.rfd_1d).evaluate([0]) == 1
        assert (self.rfd_1d * self.rfd_1d).evaluate([1]) == 16

    def test_gradient(self):
        grad = self.rfd_1d.gradient()
        assert grad[0].evaluate([0]) == 1
        assert grad[0].evaluate([1]) == 2

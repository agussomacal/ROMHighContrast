import unittest

import numpy as np

from src.lib.Functions.PolynomialFunctions import PolynomialFunctions
from src.lib.Functions.RectangularRegionsFunctions import RectangularRegionsFunctions
from src.lib.Regions.Regions import RectangularRegion, RectangularRegions


class TestRectangularRegionsFunctions(unittest.TestCase):
    def setUp(self) -> None:
        self.coeffs_1d = np.ones(2)
        self.coeffs_2d = np.ones((2, 2))

        regions = RectangularRegions([
            RectangularRegion(lower_limits=[0], upper_limits=[0.5]),
            RectangularRegion(lower_limits=[0.5], upper_limits=[1])
        ])
        self.rrf_1d = RectangularRegionsFunctions(
            regions=regions,
            functions=[
                PolynomialFunctions(coefficients=self.coeffs_1d),
                PolynomialFunctions(coefficients=self.coeffs_1d + 1)
            ]
        )
        # TODO: test 2d

    def test_integrate(self):
        assert self.rrf_1d.integrate() == 0
        assert self.rrf_1d.integrate(lower_limits=[0], upper_limits=[0.5]) == (1 + 1.5) / 2 * 0.5
        assert self.rrf_1d.integrate(lower_limits=[0.5], upper_limits=[1]) == (4 + 3) / 2 * 0.5
        assert self.rrf_1d.integrate(lower_limits=[0], upper_limits=[1]) == (1 + 1.5) / 2 * 0.5 + (4 + 3) / 2 * 0.5

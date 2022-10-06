import unittest

import numpy as np

from src.lib.Functions.PolynomialFunctions import PolynomialFunctions


class TestPolynomialFuctions(unittest.TestCase):
    def setUp(self) -> None:
        self.coeffs_1d = np.ones(3)
        self.coeffs_2d = np.ones((3, 3))
        self.coeffs_2d_2 = np.ones((2, 2))

        self.poly1d = PolynomialFunctions(coefficients=self.coeffs_1d)
        self.poly2d = PolynomialFunctions(coefficients=self.coeffs_2d)
        self.poly2d_2 = PolynomialFunctions(coefficients=self.coeffs_2d_2)

    def test_evaluate(self):
        assert self.poly1d.evaluate([0]) == 1
        assert self.poly1d.evaluate([1]) == 3
        assert self.poly1d.evaluate([2]) == 7

        assert self.poly2d.evaluate([0, 0]) == 1
        assert self.poly2d.evaluate([0, 1]) == 3
        assert self.poly2d.evaluate([1, 0]) == 3
        assert self.poly2d.evaluate([1, 1]) == 9

    def test_add(self):
        assert np.allclose((self.poly1d + 1).coefficients, [2, 1, 1])
        assert np.allclose((self.poly1d + self.poly1d).coefficients, 2 * self.poly1d.coefficients)

        assert np.allclose((self.poly2d + 1).coefficients, [[2, 1, 1],
                                                            [1, 1, 1],
                                                            [1, 1, 1]])
        assert np.allclose((self.poly2d + self.poly2d).coefficients, 2 * self.poly2d.coefficients)

    def test_mult(self):
        assert np.allclose((self.poly1d * 1).coefficients, self.coeffs_1d)
        assert np.allclose((self.poly1d * 2).coefficients, self.coeffs_1d * 2)
        assert np.allclose((self.poly1d * self.poly1d).coefficients, [1, 2, 3, 2, 1])

        assert np.allclose((self.poly2d * 1).coefficients, self.coeffs_2d)
        assert np.allclose((self.poly2d * 2).coefficients, self.coeffs_2d * 2)
        assert np.allclose((self.poly2d * self.poly2d).coefficients, [[1, 2, 3, 2, 1],
                                                                      [2, 4, 6, 4, 2],
                                                                      [3, 6, 9, 6, 3],
                                                                      [2, 4, 6, 4, 2],
                                                                      [1, 2, 3, 2, 1]])
        assert np.allclose((self.poly2d_2 * self.poly2d_2).coefficients, [[1, 2, 1],
                                                                          [2, 4, 2],
                                                                          [1, 2, 1]])

    def test_gradient(self):
        grad = self.poly1d.gradient()
        assert np.allclose(grad[0].coefficients, [1, 2])

        grad = self.poly2d.gradient()
        assert np.allclose(grad[0].coefficients, [[1, 1, 1],
                                                  [2, 2, 2]])
        assert np.allclose(grad[1].coefficients, [[1, 2],
                                                  [1, 2],
                                                  [1, 2]])

    def test_integrate(self):
        assert self.poly1d.integrate([0], [1]) == 1 + 1 / 2 + 1 / 3
        assert self.poly1d.integrate([-1], [1]) == 2 + 0 + 2 / 3

        assert self.poly2d_2.integrate([0, 0], [1, 1]) == 1+1/2+1/2+1/4
#
# class TestLagrangeInterpolationPolynomialTensorial(unittest.TestCase):
#     def test_init(self):
#         lipt = LagrangeInterpolationPolynomialTensorial(
#             coordinates_per_dim=([-1, 0, 1], [-1, 0, 1]),
#             not_null_idx=(1, 1)
#         )
#
#         assert np.shape(lipt.coefficients) == (3, 3)
#         assert np.all(lipt.coefficients == np.array([
#             [1, 0, -1],
#             [0, 0, 0],
#             [-1, 0, 1]
#         ]))
#
#
#         lipt = LagrangeInterpolationPolynomialTensorial(
#             coordinates_per_dim=([-1, 0, 1], [-1, 0, 1]),
#             not_null_idx=(0, 2)
#         )
#
#         assert np.all(lipt.coefficients == np.array([
#             [0.25, 0.25, 0],
#             [-0.25, -0.25, 0],
#             [0, 0, 0]
#         ]))

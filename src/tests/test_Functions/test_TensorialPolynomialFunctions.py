import unittest

import numpy as np

from src.lib.Functions.TensorialPolynomialFunctions import TensorizedPolynomialFunctions


class TestTensorialPolynomialFuctions(unittest.TestCase):
    def setUp(self) -> None:
        self.coeffs_1d = [np.ones(3)]
        self.coeffs_2d = [np.ones(3), np.ones(3)]
        self.coeffs_2d_2 = [np.ones(2), np.ones(2)]

        self.poly1d = TensorizedPolynomialFunctions(coefficients=self.coeffs_1d)
        self.poly2d = TensorizedPolynomialFunctions(coefficients=self.coeffs_2d)
        self.poly2d_2 = TensorizedPolynomialFunctions(coefficients=self.coeffs_2d_2)

    def test_detensorize(self):
        assert np.allclose(
            TensorizedPolynomialFunctions(coefficients=[np.ones(3), np.ones(3)]).detensorize().coefficients,
            np.ones((3, 3)))

        assert np.allclose(
            TensorizedPolynomialFunctions(coefficients=[np.ones(2), np.ones(3)]).detensorize().coefficients,
            np.ones((2, 3)))


        assert np.allclose(
            TensorizedPolynomialFunctions(coefficients=[np.ones(1), np.ones(3)]).detensorize().coefficients,
            np.ones((1, 3)))

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
        assert (self.poly1d + 1).evaluate([0]) == 2
        assert np.allclose((self.poly1d + self.poly1d).coefficients, np.array(self.coeffs_1d) * 2)

        assert np.allclose((self.poly2d + 1).coefficients, [[2, 1, 1],
                                                            [1, 1, 1],
                                                            [1, 1, 1]])
        assert np.allclose((self.poly2d + self.poly2d).coefficients, 2 * self.poly2d.detensorize().coefficients)

    def test_mult(self):
        assert np.allclose((self.poly1d * 1).coefficients, self.coeffs_1d)
        assert np.allclose((self.poly1d * 2).coefficients, np.array(self.coeffs_1d) * 2)
        assert np.allclose((self.poly1d * self.poly1d).coefficients, [1, 2, 3, 2, 1])

        assert np.allclose((self.poly2d * 1).coefficients, self.coeffs_2d)
        assert np.allclose((self.poly2d * 2).detensorize().coefficients, self.poly2d.detensorize().coefficients * 2)
        assert np.allclose((self.poly2d * self.poly2d).detensorize().coefficients, [[1, 2, 3, 2, 1],
                                                                                    [2, 4, 6, 4, 2],
                                                                                    [3, 6, 9, 6, 3],
                                                                                    [2, 4, 6, 4, 2],
                                                                                    [1, 2, 3, 2, 1]])
        assert np.allclose((self.poly2d_2 * self.poly2d_2).detensorize().coefficients, [[1, 2, 1],
                                                                                        [2, 4, 2],
                                                                                        [1, 2, 1]])

    def test_gradient(self):
        grad = self.poly1d.gradient()
        assert np.allclose(grad[0].coefficients, [1, 2])

        grad = self.poly2d.gradient()
        assert np.allclose(grad[0].detensorize().coefficients, [[1, 1, 1],
                                                                [2, 2, 2]])
        assert np.allclose(grad[1].detensorize().coefficients, [[1, 2],
                                                                [1, 2],
                                                                [1, 2]])

    def test_integrate(self):
        assert self.poly1d.integrate([0], [1]) == 1 + 1 / 2 + 1 / 3
        assert self.poly1d.integrate([-1], [1]) == 2 + 0 + 2 / 3

        assert self.poly2d_2.integrate([0, 0], [1, 1]) == 2.25

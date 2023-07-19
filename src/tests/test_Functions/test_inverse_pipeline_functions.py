import unittest

import numpy as np
from scipy.stats import qmc


def measurements_sampling_method_sobol_points(number_of_measures, xlim, ylim, seed=42, **kwargs) -> np.ndarray:
    # sobol points
    measurement_points = qmc.Sobol(d=2, scramble=True).random_base2(m=int(np.ceil(np.log2(number_of_measures))))
    # translate and shrink to be inside the domain
    measurement_points[:, 0] = measurement_points[:, 0] * (np.diff(xlim)) + xlim[0]
    measurement_points[:, 1] = measurement_points[:, 1] * (np.diff(ylim)) + ylim[0]
    # get exactly number_of_measures quantity
    return measurement_points[np.random.choice(len(measurement_points), size=number_of_measures, replace=False)]


def measurements_sampling_method_latin_square(number_of_measures, xlim, ylim, seed=42, **kwargs) -> np.ndarray:
    # choose points in a matrix so no one share rows nor columns with respect to others.
    measurement_points = np.transpose([np.random.choice(number_of_measures, size=number_of_measures, replace=False),
                                       np.random.choice(number_of_measures, size=number_of_measures, replace=False)])
    # put in [0, 1] interval
    measurement_points = measurement_points / number_of_measures
    # add a random perturbation inside the square
    measurement_points += np.random.uniform(size=(number_of_measures, 2)) / number_of_measures
    # translate and shrink to be inside the domain
    measurement_points[:, 0] = measurement_points[:, 0] * np.diff(xlim) + xlim[0]
    measurement_points[:, 1] = measurement_points[:, 1] * np.diff(ylim) + ylim[0]
    return measurement_points


class TestSamplingMethods(unittest.TestCase):
    def setUp(self) -> None:
        self.m = 50

    def test_sobol_points(self):
        points = measurements_sampling_method_sobol_points(self.m, (-1, 1), (-2, 2), seed=42)
        assert np.shape(points)[0] == self.m

    def test_latin_square(self):
        measurements_sampling_method_latin_square(self.m, (-1, 1), (-2, 2), seed=42)

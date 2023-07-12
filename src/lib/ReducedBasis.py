from logging import warning
from typing import List

import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

from src.lib.Estimators import EstimatorInv, EstimatorLinear
from src.lib.SolutionsManagers import SolutionsManager

INFINIT_A = 1e10  # 1e50


def get_high_contrast_coefficient(a):
    return np.array([np.max(coefs, axis=(-1, -2)) for coefs in a])


def orthonormalize_base(rb):
    q, r = np.linalg.qr(np.array(rb).T)
    rb = q.T
    return rb


def sort_orthogonalize_base(a_selected, rb):
    order = np.argsort(1 / a_selected)
    a_selected = a_selected[order]
    rb = rb[order, :]
    rb = orthonormalize_base(rb[order, :])
    return a_selected, rb


class BaseReducedBasis:
    def __init__(self):
        self.basis = None
        self.a = None
        self.inverse_parameter_estimator = None
        self.linear_parameter_estimator = None

    def build(self, **kwargs):
        raise Exception("Not implemented.")

    def set(self, basis, a):
        self.basis = basis
        self.a = a
        self.inverse_parameter_estimator = EstimatorInv(a)
        self.linear_parameter_estimator = EstimatorLinear(a)

    @property
    def dim(self):
        return np.shape(self.basis)[0]

    @property
    def ambient_space_dim(self):
        return np.shape(self.basis)[1]

    def __str__(self):
        return self.__class__.__name__

    def forward_modeling(self, sm: SolutionsManager, a: np.ndarray):
        return sm.generate_fm_solutions(a=a, coefficients_rom=self.basis)

    def projection(self, sm: SolutionsManager, true_solutions: np.ndarray):
        return sm.project_solutions(true_solutions, self.basis)

    def state_estimation(self, sm: SolutionsManager, measurement_points: np.ndarray, measurements: np.ndarray,
                         return_coefs=False):
        rb_evaluations_in_points = sm.evaluate_solutions(measurement_points, self.basis)
        c = np.linalg.lstsq(rb_evaluations_in_points.T, measurements.T, rcond=-1)[0]
        solution_estimations = c.T @ np.array(self.basis)
        return (c, solution_estimations) if return_coefs else solution_estimations

    def parameter_estimation_inverse(self, c):
        """

        :param c: coefficients obtained from state estimation fit
        :return:
        """
        return self.inverse_parameter_estimator.estimate_parameter(c_values=c)

    def parameter_estimation_linear(self, c):
        """

        :param c: coefficients obtained from state estimation fit
        :return:
        """
        return self.linear_parameter_estimator.estimate_parameter(c_values=c)

    def __getitem__(self, item):
        # get a new reduced basis with the sub-sampled elements given by the slicing.
        rb = BaseReducedBasis()
        rb.set(basis=self.basis[item], a=self.a[item])
        return rb

    def orthonormalize(self):
        _, self.basis = sort_orthogonalize_base(
            get_high_contrast_coefficient(self.a),
            np.reshape(self.basis, (-1, self.ambient_space_dim))
        )


GREEDY_FOR_H10 = r"$H^1_0$"
GREEDY_FOR_GALERKIN = "galerkin"


class ReducedBasisGreedy(BaseReducedBasis):
    def __init__(self, greedy_for=GREEDY_FOR_GALERKIN):
        self.greedy_for = greedy_for
        self.name = "Greedy " + self.greedy_for
        self.linestyle = "solid" if greedy_for == GREEDY_FOR_H10 else "dashed"
        super().__init__()

    def build(self, n: int, sm: SolutionsManager, solutions2train, a2train: List[np.ndarray] = (()),
              solutions2train_h1norm=1, **kwargs):
        high_contrast_a = get_high_contrast_coefficient(a2train)

        basis = np.empty((0, 0))
        basis_orth = basis.copy()
        a_selected = []
        a = []
        for _ in tqdm(range(n), desc="Obtaining greedy basis."):
            if self.greedy_for == GREEDY_FOR_H10:
                approx_solutions_coefs = sm.project_solutions(solutions=solutions2train, coefficients_rom=basis_orth)
            elif self.greedy_for == GREEDY_FOR_GALERKIN:
                approx_solutions_coefs = sm.generate_fm_solutions(a=a2train, coefficients_rom=basis_orth)
            else:
                raise Exception(f"Not implemented greedy for {self.greedy_for}, "
                                f"should be one of [{GREEDY_FOR_H10}, {GREEDY_FOR_GALERKIN}]")

            max_error_index = np.argmax(sm.H10norm(approx_solutions_coefs - solutions2train) / solutions2train_h1norm)
            max_element = np.reshape(solutions2train[max_error_index], (1, -1))
            basis = max_element if len(basis) == 0 else np.concatenate((basis, max_element), axis=0)
            a.append(a2train[max_error_index])

            # orthonormalize for stability and choose the ordering by the contrast of the higher coefficient.
            a_selected = np.append(a_selected, np.ravel(high_contrast_a[max_error_index]))
            a_selected, basis_orth = sort_orthogonalize_base(a_selected, np.reshape(basis, (len(basis), -1)))

        super().set(basis=basis, a=a)
        return self


def get_inf_solutions_starting_basis(solutions2train, a2train, only_one_block=True):
    """
    Choose among the solutions presented those that have only one of the subdomains going to infinity.
    The combinations should be approximating by linear combination of this basic ones.
    """
    num_hc_blocks = np.sum(np.array(a2train) == INFINIT_A, axis=(-1, -2))
    chosen_ix = np.ravel(np.where(num_hc_blocks == 1 if only_one_block else num_hc_blocks != 0))
    free_ix = np.ravel(np.where(num_hc_blocks != 1 if only_one_block else num_hc_blocks == 0))
    return solutions2train[chosen_ix], a2train[chosen_ix], solutions2train[free_ix], a2train[free_ix]


def get_starting_basis(solutions2train, a2train, add_inf_solutions=True):
    if add_inf_solutions:
        # choose the high contrast solutions available in the training set
        basis, a, solutions2train, a2train = get_inf_solutions_starting_basis(solutions2train, a2train,
                                                                              only_one_block=False)
    else:
        basis, a, solutions2train, a2train = get_inf_solutions_starting_basis(solutions2train, a2train,
                                                                              only_one_block=False)
        # do nothing, just initialize
        basis = np.empty((0, np.shape(solutions2train)[1]))
        a = np.empty((0,) + np.shape(a2train)[1:])
    return basis, a, solutions2train, a2train


class ReducedBasisRandom(BaseReducedBasis):
    def __init__(self, add_inf_solutions=True):
        self.add_inf_solutions = add_inf_solutions
        self.name = "Random" + (r" $\infty$" if add_inf_solutions else "")
        super().__init__()

    def build(self, n: int, sm: SolutionsManager, solutions2train, a2train: List[np.ndarray] = (()),
              solutions2train_h1norm=1, seed=42, **kwargs):
        basis, a, solutions2train, a2train = get_starting_basis(solutions2train, a2train, self.add_inf_solutions)
        np.random.seed(seed)
        chosen_ix = np.random.choice(len(solutions2train), size=n, replace=False)
        super().set(basis=np.vstack((basis, solutions2train[chosen_ix]))[:n],
                    a=np.vstack((a, a2train[chosen_ix]))[:n])
        return self


class ReducedBasisPCA(BaseReducedBasis):
    def __init__(self, add_inf_solutions=True):
        self.add_inf_solutions = add_inf_solutions
        self.name = "PCA" + (r" $\infty$" if add_inf_solutions else "")
        super().__init__()

    def build(self, n: int, sm: SolutionsManager, solutions2train, a2train: List[np.ndarray] = (()),
              solutions2train_h1norm=1, add_inf_solutions=True, seed=42, **kwargs):
        basis, a, solutions2train, a2train = get_starting_basis(solutions2train, a2train, self.add_inf_solutions)
        # log-transform data before performing PCA?
        # pca = PCA(n_components=n).fit(np.log(solutions2train))
        # super().set(basis=np.vstack((basis, np.exp(pca.components_)))[:n],
        #             a=np.vstack((a, a2train))[:n])
        pca = PCA(n_components=n).fit(solutions2train)
        super().set(basis=np.vstack((basis, pca.components_))[:n],
                    a=np.vstack((a, a2train))[:n])
        warning("PCA method has not been adapted for inverse parameter estimation, the a coefficients are not correct.")
        return self

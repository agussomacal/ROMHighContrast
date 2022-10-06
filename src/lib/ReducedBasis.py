from typing import List

import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

from lib.Estimators import EstimatorInv, EstimatorLinear
from lib.SolutionsManagers import SolutionsManager

INFINIT_A = 1e10  # 1e50

LINESTYLES = {
    "greedy": "solid",
    "mixed_refinement_inf": "dashed",
    "mixed": "dashdot",
    "uniform_refinement_inf": "dotted",
    "uniform": "dotted",
    "uniform_refinement": "dotted",
    "dyadic_refinement_inf": "dashdot",
    "dyadic": "dashed",
    "random_inf": (0, (3, 5, 1, 5, 1, 5)),
    "random": "solid"
}

MARKERS = {
    "greedy": ".",
    "mixed_refinement_inf": "*",
    "mixed": "*",
    "uniform_refinement_inf": "o",
    "uniform": "o",
    "uniform_refinement": "s",
    "dyadic_refinement_inf": "^",
    "dyadic": "^",
    "random_inf": "x",
    "random": "x"
}

COLORS = {
    "greedy": "green",
    "mixed_refinement": "blueviolet",
    "mixed_refinement_inf": "mediumblue",
    "mixed": "blueviolet",
    "uniform_refinement": "royalblue",
    "uniform_refinement_inf": "darkblue",
    "uniform": "blue",
    "dyadic_refinement": "indianred",
    "dyadic_refinement_inf": "darkred",
    "dyadic": "red",
    "random": "grey",
    "random_inf": "darkgoldenrod",
    "blocks": "slategrey"
}


def get_high_contrast_coefficient(a):
    return np.array([np.max(coefs, axis=(-1, -2)) for coefs in a])


def sort_orthogonalize_base(a_selected, rb):
    order = np.argsort(1 / a_selected)
    a_selected = a_selected[order]
    rb = rb[order, :]
    q, r = np.linalg.qr(rb.T)
    rb = q.T
    return a_selected, rb


class BaseReducedBasis:
    def __init__(self, basis, a, **kwargs):
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
        return sm.project_solutions(true_solutions, self.basis, optim_method="lsq")

    def state_estimation(self, sm: SolutionsManager, measurement_points: np.ndarray, measurements: np.ndarray,
                         return_coefs=False):
        rb_evaluations_in_points = sm.evaluate_solutions(measurement_points, self.basis)
        c = np.linalg.lstsq(rb_evaluations_in_points.T, measurements.T, rcond=-1)[0]
        solution_estimations = c.T @ np.array(self.basis)
        return c, solution_estimations if return_coefs else solution_estimations

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
        return BaseReducedBasis(basis=self.basis[item], a=self.a[item])

    def orthonormalize(self):
        _, self.basis = sort_orthogonalize_base(
            get_high_contrast_coefficient(self.a),
            np.reshape(self.basis, (-1, self.ambient_space_dim))
        )


class ReducedBasisGreedy(BaseReducedBasis):
    name = "Greedy"
    color = "Green"
    linestyle = "solid"
    markers = "."

    def __init__(self, n: int, sm: SolutionsManager, solutions2train, a2train: List[np.ndarray] = (()),
                 optim_method="lsq", greedy_for="projection", solutions2train_h1norm=1, **kwargs):
        high_contrast_a = get_high_contrast_coefficient(a2train)

        basis = np.empty((0, 0))
        basis_orth = basis.copy()
        a_selected = []
        a = []
        for _ in tqdm(range(n), desc="Obtaining greedy basis."):
            if greedy_for == "projection":
                approx_solutions_coefs = sm.project_solutions(solutions=solutions2train, coefficients_rom=basis_orth,
                                                              optim_method=optim_method)
            elif greedy_for == "forward_modeling":
                approx_solutions_coefs = sm.generate_fm_solutions(a=a2train, coefficients_rom=basis_orth)
            else:
                raise Exception(f"Not implemented greedy for {greedy_for}, should be one of ['projection']")

            max_error_index = np.argmax(sm.H10norm(approx_solutions_coefs - solutions2train) / solutions2train_h1norm)
            max_element = np.reshape(solutions2train[max_error_index], (1, -1))
            basis = max_element if len(basis) == 0 else np.concatenate((basis, max_element), axis=0)
            a.append(a2train[max_error_index])

            # orthonormalize for stability and choose the ordering by the contrast of the higher coefficient.
            a_selected = np.append(a_selected, np.ravel(high_contrast_a[max_error_index]))
            a_selected, basis_orth = sort_orthogonalize_base(a_selected, np.reshape(basis, (len(basis), -1)))

        super().__init__(basis=basis, a=a)


def get_inf_solutions_starting_basis(solutions2train, a2train):
    num_hc_blocks = np.sum(np.array(a2train) == INFINIT_A, axis=(-1, -2))
    chosen_ix = np.ravel(np.where(num_hc_blocks == 1))
    free_ix = np.ravel(np.where(num_hc_blocks != 1))
    return solutions2train[chosen_ix], a2train[chosen_ix], solutions2train[free_ix], a2train[free_ix]


def get_starting_basis(solutions2train, a2train, add_inf_solutions=True):
    if add_inf_solutions:
        # choose the high contrast solutions available in the training set
        basis, a, solutions2train, a2train = get_inf_solutions_starting_basis(solutions2train, a2train)
    else:
        # do nothing, just initialize
        basis = np.array([])
        a = np.array([])
    return basis, a, solutions2train, a2train


class ReducedBasisRandom(BaseReducedBasis):
    name = "Random"
    color = "blue"
    linestyle = "solid"
    markers = "*"

    def __init__(self, n: int, solutions2train, a2train: List[np.ndarray] = (()), add_inf_solutions=True, seed=42,
                 **kwargs):
        basis, a, solutions2train, a2train = get_starting_basis(solutions2train, a2train, add_inf_solutions)
        np.random.seed(seed)
        chosen_ix = np.random.choice(len(solutions2train), size=n, replace=False)
        super().__init__(basis=np.vstack((basis, solutions2train[chosen_ix]))[:n],
                         a=np.vstack((a, a2train[chosen_ix]))[:n])


class ReducedBasisPCA(BaseReducedBasis):
    name = "Random"
    color = "grey"
    linestyle = "solid"
    markers = "x"

    def __init__(self, n: int, solutions2train, a2train: List[np.ndarray] = (()), **kwargs):
        raise Exception("Not implemented.")
        pca = PCA(n_components=n).fit(solutions2train)
        super().__init__(basis=pca.components_, a=a2train[chosen_ix])

from functools import partial

from pathos.multiprocessing import Pool, cpu_count
from typing import Union, List, Tuple

import numpy as np
from scipy import linalg
from scipy.interpolate import lagrange
from scipy.sparse.linalg import spsolve
from sklearn.linear_model import Ridge


def h1_error(v: List[np.ndarray]):
    return np.sqrt(np.mean(np.sum(np.power(np.gradient(v, axis=(1, 2)), 2), axis=0), axis=(1, 2)))


def galerkin(a, B_total, A_preassembled, method="lsq"):
    # create A matrix
    A_assembled = np.einsum(
        "pqij,pq->ij",  # sum over regions (quarter contraction)
        A_preassembled,
        a
    )
    # inf_coefs = np.isinf(A_assembled.max(axis=0)) | np.isnan(A_assembled.max(axis=0))
    # coefs = np.zeros(len(B_total))
    if method.lower() == "lsq":
        # coefs[~inf_coefs] = np.linalg.lstsq(A_assembled[~inf_coefs, :][:, ~inf_coefs], B_total[~inf_coefs], rcond=-1)[0]
        # coefs[~inf_coefs] = linalg.solve(A_assembled[~inf_coefs, :][:, ~inf_coefs], B_total[~inf_coefs], assume_a='pos')
        coefs = linalg.solve(A_assembled, B_total, assume_a='pos')
    elif method.lower() == "lsqsparse":
        coefs = spsolve(A_assembled, B_total)
        # lsmr()
        # lsqr
    elif method.lower() == "ridge":
        # coefs[~inf_coefs] = Ridge(alpha=1e-15, fit_intercept=False).fit(A_assembled[~inf_coefs, :][:, ~inf_coefs],
        #                                                                 B_total[~inf_coefs]).coef_
        coefs = Ridge(alpha=1e-15, fit_intercept=False).fit(A_assembled, B_total).coef_
    else:
        raise Exception(f"Method {method} Not implemented.")
    return coefs


class SolutionsManager:
    def __init__(self, A_preassembled, B_total, num_cores=1, method="lsq"):
        self.method = method
        self.vspace_dim = len(B_total)
        self.blocks_geometry = np.shape(A_preassembled)[:2]
        self.A_preassembled = A_preassembled
        self.A_preassembled4h1_norm = np.einsum("abij->ij", self.A_preassembled)
        self.B_total = B_total
        self.mapfunction = map if num_cores == 1 else Pool(min((num_cores, cpu_count() - 1))).map

    def __str__(self):
        return self.__class__.__name__

    def H10norm(self, solutions: List[np.ndarray]):
        # return np.sqrt(np.einsum("abij,ki,kj->k", self.A_preassembled, solutions, solutions))
        return np.sqrt(np.einsum("ij,ki,kj->k", self.A_preassembled4h1_norm, solutions, solutions))

    @staticmethod
    def l2norm(solutions: List[np.ndarray]):
        return np.sqrt(np.sum(np.square(solutions), axis=1))

    def generate_solutions(self, a2try):
        return np.array(
            list(self.mapfunction(
                partial(galerkin, B_total=self.B_total, A_preassembled=self.A_preassembled, method=self.method),
                a2try)))

    def generate_riesz(self, x, norm="h10"):
        """
        param: x: list of measurement points [(x1, y1), ... (xm, ym)]
        returns Riesz representers: shape (m, N)
        """
        B_total = self.evaluate_solutions(points=x, solutions=np.eye(self.vspace_dim)).T  # (N, m)
        if norm == "l2":
            return B_total  # (m, N)
        elif norm.lower() == "h10":
            # the evaluation operator (B_total used) in truth does not belong to H10,
            # we need a kernel (like a gaussean) to compute the B_total correctly.
            return np.array([np.squeeze(
                galerkin(a=np.ones(self.blocks_geometry), B_total=B_total[i],
                         A_preassembled=self.A_preassembled, method=self.method)) for i, xi in enumerate(x)])  # (m, N)
        else:
            raise Exception("Not implemented.")

    def generate_fm_solutions(self, a: Union[np.ndarray, List[np.ndarray]], coefficients_rom: List[np.ndarray]):
        if len(coefficients_rom) == 0:
            # in case no reduced basis then return 0 always.
            return np.zeros((len(a), self.vspace_dim))
        else:
            A_kl = np.einsum(
                "...jk,dk->...jd",  # right multiplication (grad u_k)
                np.einsum(
                    "...jk,dj->...dk",  # left multiplication (grad u_l)
                    self.A_preassembled,
                    coefficients_rom
                ),
                coefficients_rom
            )

            B_k = np.array(coefficients_rom) @ self.B_total
            c_i = np.array(
                list(self.mapfunction(partial(galerkin, B_total=B_k, A_preassembled=A_kl, method=self.method), a)))
            return np.einsum("id,dj->ij", c_i, coefficients_rom)

    def project_solutions(self, solutions: List[np.ndarray], coefficients_rom: List[np.ndarray]):
        if len(coefficients_rom) == 0:
            # in case no reduced basis then return 0 always.
            return np.zeros((len(solutions), self.vspace_dim))
        else:
            B_km = np.sum(
                np.einsum(
                    "...jk,dk->...jd",  # right multiplication (grad u)
                    np.einsum(
                        "...jk,dj->...dk",  # left multiplication (grad u_l)
                        self.A_preassembled,
                        coefficients_rom
                    ),
                    solutions
                ),
                axis=(0, 1)  # sum over the geometric dimensions of the domain.
            )
            A_kl = np.einsum(
                "...jk,dk->...jd",  # right multiplication (grad u_k)
                np.einsum(
                    "...jk,dj->...dk",  # left multiplication (grad u_l)
                    self.A_preassembled,
                    coefficients_rom
                ),
                coefficients_rom
            )

            def project_galerkin(B_k):
                return galerkin(a=np.ones(np.shape(A_kl)[:2]), B_total=B_k, A_preassembled=A_kl, method=self.method)

            c_i = np.array(list(self.mapfunction(project_galerkin, np.transpose(B_km))))
            return np.einsum("id,dj->ij", c_i, coefficients_rom)

    def evaluate_solutions(self, points: np.ndarray, solutions: List[np.ndarray]) -> np.ndarray:
        raise Exception("Not implemented.")


class SolutionsManagerFEM(SolutionsManager):
    def __init__(self, blocks_geometry: Tuple[int, int], N: int, num_cores=1, method="lsq"):
        nrb, ncb = blocks_geometry  # number of blocks (in columns dim) (in rows dim)
        self.N = N  # number of columns in each subsquare
        self.x_domain = (-ncb / 2.0, ncb / 2.0)
        self.y_domain = (-nrb / 2.0, nrb / 2.0)

        # number of inner vertices = dimension of the basis
        self.nc_inner_vertices = ncb * self.N - 1
        self.nr_inner_vertices = nrb * self.N - 1
        dim = (self.nc_inner_vertices) * (self.nr_inner_vertices)
        self.nc_cells = ncb * self.N + 1
        self.nr_cells = nrb * self.N + 1

        nb_vertices = self.nc_cells * self.nr_cells
        inner_vertices = np.zeros((self.nr_cells, self.nc_cells))
        inner_vertices[1:nrb * self.N, 1:ncb * self.N] = np.ones((self.nr_inner_vertices, self.nc_inner_vertices))
        inner_vertices = inner_vertices.reshape(nb_vertices)
        inner_vertices = np.array(inner_vertices, dtype=bool)

        # self.points_c = np.linspace(-ncb / 2, ncb / 2, self.nc_cells) ** 2 * np.sign(np.linspace(-1, 1, self.nc_cells))
        # self.points_r = np.linspace(-nrb / 2, nrb / 2, self.nr_cells) ** 2 * np.sign(np.linspace(-1, 1, self.nr_cells))

        self.points_c = np.linspace(*self.x_domain, self.nc_cells)
        self.points_r = np.linspace(*self.y_domain, self.nr_cells)
        width = height = 1 / self.N
        area = width * height
        # triangles partition with diagonal direction south-west to north-east, that's why the 2 in nb_tri_cells, each
        # square is divided in 2 sub triangles.
        nb_tri_cells = 2 * (ncb * self.N) * (nrb * self.N)

        # Source term
        B_total = np.zeros((self.nr_cells, self.nc_cells))
        for i in range(nrb * self.N):
            for j in range(ncb * self.N):
                # area = (self.points_c[i + 1] - self.points_c[i]) * (self.points_r[j + 1] - self.points_r[j])
                B_total[i, j] += area / 6
                B_total[i + 1, j] += area / 3
                B_total[i, j + 1] += area / 3
                B_total[i + 1, j + 1] += area / 6
        B_total = B_total[1:-1, 1:-1].reshape(dim)

        def A(a):
            res = np.zeros((nb_vertices, nb_vertices))
            for cell in range(nb_tri_cells):
                column = (cell // 2) % (ncb * self.N)
                line = cell // (2 * ncb * self.N)
                acell = a[line // self.N, column // self.N]
                if cell % 2 == 0:
                    pos = self.nc_cells * line + column
                    # grad \phi_i grad \phi_i
                    res[pos, pos] += acell * 2
                    res[pos + 1, pos + 1] += acell
                    res[pos + self.nc_cells, pos + self.nc_cells] += acell
                    # grad \phi_i grad \phi_j and the symmetric
                    res[pos, pos + 1] -= acell
                    res[pos + 1, pos] -= acell
                    res[pos, pos + self.nc_cells] -= acell
                    res[pos + self.nc_cells, pos] -= acell
                else:
                    # grad \phi_i grad \phi_i
                    pos = self.nc_cells * (line + 1) + column + 1
                    res[pos, pos] += acell * 2
                    res[pos - 1, pos - 1] += acell
                    res[pos - self.nc_cells, pos - self.nc_cells] += acell
                    # grad \phi_i grad \phi_j and the symmetric
                    res[pos, pos - 1] -= acell
                    res[pos - 1, pos] -= acell
                    res[pos, pos - self.nc_cells] -= acell
                    res[pos - self.nc_cells, pos] -= acell
            return res[inner_vertices][:, inner_vertices] / 2

        A_preassembled = np.array(list(map(A, np.eye(nrb * ncb).reshape((nrb * ncb, nrb, ncb))))).reshape(
            (nrb, ncb, dim, dim))
        super().__init__(A_preassembled, B_total, num_cores=num_cores, method=method)

    def evaluate_solutions(self, points: np.ndarray, solutions: List[np.ndarray]) -> np.ndarray:
        """

        :param points: (m, 2) array of points coordinates between [0, 1] where we want to measure.
        :param solutions: List of n solutions (coefficients) to use to evaluate in the points.
        :return: (n, m) array with the evaluations of the n solutions in the m points.
        """
        evaluations = []
        for solution in solutions:
            evaluations.append([])
            for x, y in points:
                val = np.zeros((self.nr_cells, self.nc_cells))
                val[1:-1, 1:-1] = np.reshape(solution, (self.nr_inner_vertices, self.nc_inner_vertices))
                val = val.T
                px = np.searchsorted(self.points_c, x) - 1
                py = np.searchsorted(self.points_r, y) - 1
                qx = (x - self.points_c[px]) / (self.points_c[px + 1] - self.points_c[px])
                qy = (y - self.points_r[py]) / (self.points_r[py + 1] - self.points_r[py])
                if qx + qy < 1:
                    evaluations[-1].append((1 - qx - qy) * val[px, py] + qx * val[px + 1, py] + qy * val[px, py + 1])
                else:
                    evaluations[-1].append(
                        (qx + qy - 1) * val[px + 1, py + 1] + (1 - qx) * val[px, py + 1] + (1 - qy) * val[px + 1, py])
        return np.array(evaluations)


class SolutionsManagerPolynomial(SolutionsManager):
    def __init__(self, lagrange_polynomials_degree):
        self.lagrange_polynomials_degree = lagrange_polynomials_degree
        self.quarter_dim, self.dim_1d, vspace_dim, self.base_lagrange, self.P = init_polynomial_variables(
            lagrange_polynomials_degree)

        # create A pre assembled matrix without a coefficient
        self.A_quarter = np.zeros((self.quarter_dim, self.quarter_dim, 2, 2))
        for i in range(self.quarter_dim):
            for j in range(self.quarter_dim):
                int_x_dx_phi_i_dx_phi_j = np.polyval(
                    np.polyint(np.polyder(self.base_lagrange[i // self.lagrange_polynomials_degree]) * np.polyder(
                        self.base_lagrange[j // self.lagrange_polynomials_degree])), 1)
                int_y_phi_i_phi_j = np.polyval(np.polyint(
                    self.base_lagrange[i % self.lagrange_polynomials_degree] * self.base_lagrange[
                        j % self.lagrange_polynomials_degree]), 1)
                self.A_quarter[i, j, 0, 0] = int_x_dx_phi_i_dx_phi_j * int_y_phi_i_phi_j

                int_x_phi_i_phi_j = np.polyval(np.polyint(
                    self.base_lagrange[i // self.lagrange_polynomials_degree] * self.base_lagrange[
                        j // self.lagrange_polynomials_degree]),
                    1)
                int_y_dy_phi_i_dy_phi_j = np.polyval(
                    np.polyint(np.polyder(self.base_lagrange[i % self.lagrange_polynomials_degree]) * np.polyder(
                        self.base_lagrange[j % self.lagrange_polynomials_degree])), 1)
                self.A_quarter[i, j, 1, 1] = int_x_phi_i_phi_j * int_y_dy_phi_i_dy_phi_j

                int_x_phi_i_dx_phi_j = np.polyval(
                    np.polyint(self.base_lagrange[i // self.lagrange_polynomials_degree] * np.polyder(
                        self.base_lagrange[j // self.lagrange_polynomials_degree])),
                    1)
                int_y_dy_phi_i_phi_j = np.polyval(
                    np.polyint(
                        np.polyder(self.base_lagrange[i % self.lagrange_polynomials_degree]) * self.base_lagrange[
                            j % self.lagrange_polynomials_degree]),
                    1)
                self.A_quarter[i, j, 0, 1] = int_x_phi_i_dx_phi_j * int_y_dy_phi_i_phi_j

                int_x_dx_phi_i_phi_j = np.polyval(
                    np.polyint(
                        np.polyder(self.base_lagrange[i // self.lagrange_polynomials_degree]) * self.base_lagrange[
                            j // self.lagrange_polynomials_degree]),
                    1)
                int_y_phi_i_dy_phi_j = np.polyval(
                    np.polyint(self.base_lagrange[i % self.lagrange_polynomials_degree] * np.polyder(
                        self.base_lagrange[j % self.lagrange_polynomials_degree])),
                    1)
                self.A_quarter[i, j, 1, 0] = int_x_dx_phi_i_phi_j * int_y_phi_i_dy_phi_j

        A_preassembled = np.zeros((4, vspace_dim, vspace_dim, 2, 2))
        for quarter in range(4):
            A_preassembled[quarter] = np.tensordot(self.P[quarter],
                                                   np.tensordot(self.P[quarter], self.A_quarter,
                                                                axes=((0,), (0,))),
                                                   axes=((0,), (1,)))

        # create B matrix
        B_quarter = np.zeros(self.quarter_dim)
        for i in range(self.quarter_dim):
            B_quarter[i] = np.polyval(np.polyint(self.base_lagrange[i // self.lagrange_polynomials_degree]),
                                      1) * np.polyval(
                np.polyint(self.base_lagrange[i % self.lagrange_polynomials_degree]), 1)

        B_total = np.zeros(vspace_dim)
        for quarter in range(4):
            B_total += np.dot(self.P[quarter].T, B_quarter)

        super().__init__(np.einsum("abcdd->abc", A_preassembled), B_total)

    def evaluate_solutions(self, points: np.ndarray, solutions: List[np.ndarray]) -> np.ndarray:
        """

        :param points: (m, 2) array of points coordinates between [0, 1] where we want to measure.
        :param solutions: List of n solutions (coefficients) to use to evaluate in the points.
        :return: (n, m) array with the evaluations of the n solutions in the m points.
        """
        M = len(points)
        square_ix = np.sign(np.array(points // 0.5, dtype=int))  # sign to force 0 or 1 when given a point in the border
        point_in_square = np.abs(2 * np.array(points) - 1)[:, [1, 0]]  # point in the subdomain
        square_ix[:, 1] *= 2
        square_ix = np.sum(square_ix, axis=1)  # which square subdomain

        x_eval_poly_lagrange = np.array(
            [np.polyval(polynom_unidim, point_in_square[:, 0]) for polynom_unidim in self.base_lagrange])
        y_eval_poly_lagrange = np.array(
            [np.polyval(polynom_unidim, point_in_square[:, 1]) for polynom_unidim in self.base_lagrange])

        eval_point_quarter = np.zeros((self.quarter_dim, M))
        for i in range(self.quarter_dim):
            eval_point_quarter[i, :] = x_eval_poly_lagrange[i // self.lagrange_polynomials_degree, :] * \
                                       y_eval_poly_lagrange[i % self.lagrange_polynomials_degree, :]

        eval_points = np.zeros((self.vspace_dim, M))
        for j in range(M):
            eval_points[:, j] = self.P[square_ix[j], :, :].T @ eval_point_quarter[:, j]

        return np.array([eval_points.T @ solution for solution in solutions])


def init_polynomial_variables(lagrange_polynomials_degree):
    lagrange_polynomials_degree = lagrange_polynomials_degree
    quarter_dim = lagrange_polynomials_degree ** 2
    dim_1d = 2 * lagrange_polynomials_degree - 1
    vspace_dim = dim_1d ** 2
    center = lagrange_polynomials_degree * dim_1d - lagrange_polynomials_degree

    points = (1 + np.sin(np.linspace(-np.pi / 2, np.pi / 2, lagrange_polynomials_degree + 1))) / 2
    base_lagrange = [lagrange(points, line) for line in np.eye(lagrange_polynomials_degree + 1)]

    # Create P conversion matrix
    P = np.zeros((4, quarter_dim, vspace_dim))
    for i in range(quarter_dim):
        P[0, i, center - (i % lagrange_polynomials_degree) - dim_1d * (i // lagrange_polynomials_degree)] = 1
        P[1, i, center + (i % lagrange_polynomials_degree) - dim_1d * (i // lagrange_polynomials_degree)] = 1
        P[2, i, center - (i % lagrange_polynomials_degree) + dim_1d * (i // lagrange_polynomials_degree)] = 1
        P[3, i, center + (i % lagrange_polynomials_degree) + dim_1d * (i // lagrange_polynomials_degree)] = 1

    return quarter_dim, dim_1d, vspace_dim, base_lagrange, P

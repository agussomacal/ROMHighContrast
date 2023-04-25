from collections import namedtuple
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from PerplexityLab.visualization import perplex_plot
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor

from src import config
from PerplexityLab.DataManager import DataManager, JOBLIB
from PerplexityLab.LabPipeline import LabPipeline

from src.lib.SolutionsManagers import SolutionsManagerFEM

ZERO = 1e-15
Bounds = namedtuple("Bounds", "lower upper")
MWhere = namedtuple("MWhere", "m start")


# @dataclass(unsafe_hash=True)
class VnFamily:
    def __init__(self, blocks_geometry: Tuple[int, int], a_bounds: Tuple[Bounds, ...], mesh: int):
        self.blocks_geometry = blocks_geometry
        self.a_bounds = a_bounds
        self.mesh = mesh

    def __eq__(self, other):
        return other.blocks_geometry == self.blocks_geometry and \
               other.a_bounds == self.a_bounds and \
               other.mesh == self.mesh

    def __hash__(self):
        return hash(self.__repr__())

    @property
    def dim(self):
        return sum([b.lower < b.upper for b in self.a_bounds])

    def __repr__(self):
        vals = [f"[{b.lower}, {b.upper}]" for b in self.a_bounds] + [f"mesh: {self.mesh}"]
        return "; ".join(vals)


def vn_family_sampler(n_train, n_test, geometry, lower_bounds, upper_bounds, mesh):
    a = [np.reshape(a_coefs, geometry)
         for a_coefs in zip(*[np.random.uniform(lower_bounds, upper_bounds, n_train + n_test)
                              for _ in range(np.prod(geometry))])]
    sm = SolutionsManagerFEM(blocks_geometry=geometry, N=mesh, num_cores=1, method="lsq")
    solutions = sm.generate_solutions(a)
    return {"solution_manager": sm, "a": a, "solutions": solutions}


def do_pca(n_train, solutions):
    pca = PCA(n_components=np.min((np.shape(solutions)[1], n_train))).fit(solutions[:n_train])
    pca_projections = pca.transform(solutions)
    return {"pca_projections": pca_projections}


def get_known_unknown_indexes(mwhere, pca_projections, learn_higher_modes_only):
    indexes = np.arange(np.shape(pca_projections)[1], dtype=int)
    known_indexes = indexes[mwhere.start:mwhere.start + mwhere.m]
    unknown_indexes = indexes[mwhere.start + mwhere.m:]
    if not learn_higher_modes_only:
        unknown_indexes = np.append(indexes[:mwhere.start], unknown_indexes)
    return known_indexes, unknown_indexes


def learn_eigenvalues(model: Pipeline):
    def decorated_function(n_train, n_test, pca_projections, mwhere: MWhere, learn_higher_modes_only=True):
        known_indexes, unknown_indexes = get_known_unknown_indexes(mwhere, pca_projections, learn_higher_modes_only)
        model.fit(pca_projections[:n_train, known_indexes], pca_projections[:n_train, unknown_indexes])
        predictions = model.predict(pca_projections[-n_test:, known_indexes])
        error = pca_projections[-n_test:, unknown_indexes] - predictions
        return {
            "error": error
        }

    decorated_function.__name__ = " ".join([s[0] for s in model.steps])
    return decorated_function


class NullModel(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        return self

    def predict(self, X):
        return 0


@perplex_plot
def k_plot(fig, ax, error, experiments, mwhere, learn_higher_modes_only, n_train, pca_projections, label_var="experiments",
           add_mwhere=False, color_dict=None):
    n_train, error, mwhere, experiments, learn_higher_modes_only, pca_projections = tuple(
        zip(*[(nt, e, m, ex, lhmo, pcap) for nt, e, m, ex, lhmo, pcap in
              zip(n_train, error, mwhere, experiments, learn_higher_modes_only, pca_projections) if
              e is not None and ex is not None]))

    mse = list(map(lambda e: np.sqrt(np.mean(np.array(e) ** 2, axis=0)).squeeze(), error))

    for i, (exp_i, y_i, ms, lhmo, ntr, pcap) in enumerate(zip(experiments, mse, mwhere, learn_higher_modes_only, n_train, pca_projections)):
        K_MAX = np.shape(pcap)[1]
        k_full = np.append([0], np.repeat(np.arange(1, K_MAX + 1, dtype=float), 2) * np.array([-1, 1] * K_MAX))
        k_full[k_full > 0] = np.log10(k_full[k_full > 0])
        k_full[k_full < 0] = -np.log10(-k_full[k_full < 0])
        known_indexes, unknown_indexes = get_known_unknown_indexes(ms, pcap, lhmo)
        k = k_full[unknown_indexes]
        # TODO: do it without an if
        if label_var == "experiments":
            label_i = exp_i
        elif label_var == "n_train":
            label_i = ntr
        else:
            raise Exception(f"label_var {label_var} not implemented.")

        if isinstance(color_dict, dict) and label_i in color_dict.keys():
            c = color_dict[label_i]
        else:
            c = sns.color_palette("colorblind")[i]
        m = "o"
        ax.plot(k[(y_i > ZERO) & (k < 0)], y_i[(y_i > ZERO) & (k < 0)], "--", marker=m, c=c)
        ax.plot(k[(y_i > ZERO) & (k > 0)], y_i[(y_i > ZERO) & (k > 0)], "--", marker=m,
                label=f"{label_i}{f': start={ms.start}, m={ms.m}' if add_mwhere else ''}", c=c)
    # k = np.sort(np.unique(np.ravel(k_full)))
    # ax.plot(k[k < 0], 1.0 / 10 ** (-k[k < 0]), ":k")
    # ax.plot(k[k > 0], 1.0 / 10 ** (k[k > 0]), ":k", label=r"$k^{-1}$")
    ticks = ax.get_xticks()
    ax.set_xticks(ticks, [fr"$10^{{{abs(int(t))}}}$" for t in ticks])
    ax.legend(loc='upper right')
    ax.set_yscale("log")
    ax.set_xlabel(r"$\alpha_k$" + "\t\t\t\t   " + r"$\beta_k$")
    ax.set_ylabel("MSE")


if __name__ == "__main__":
    name = f"FittingEigenvalues"
    data_manager = DataManager(
        path=config.results_path,
        name=name,
        format=JOBLIB,
        # country_alpha_code="FR",
        trackCO2=True
    )

    # Parameters for experiment
    geometry = [(2, 2), (4, 4)]
    lower_bounds = [1]
    upper_bounds = [100]
    mesh = [5]
    mwhere = [MWhere(start=0, m=3)]
    models = [
        Pipeline([("Null", NullModel())]),
        # Pipeline([("LR", LinearRegression())]),
        # Pipeline([("Quadratic", PolynomialFeatures(degree=2)), ("LR", LinearRegression())]),
        # Pipeline([("Degree 4", PolynomialFeatures(degree=4)), ("LR", LinearRegression())]),
        Pipeline([("Tree", DecisionTreeRegressor())]),
        # Pipeline([("RF", RandomForestRegressor(n_estimators=10))]),
        # Pipeline([("NN", FNNModel(hidden_layer_sizes=(20, 20,), activation="sigmoid"))]))
    ]

    # Experiment
    lab = LabPipeline()
    lab.define_new_block_of_functions("manifold_sampling", vn_family_sampler)
    lab.define_new_block_of_functions("eigen", do_pca)
    lab.define_new_block_of_functions(
        "experiments",
        *list(map(learn_eigenvalues, models)),
    )
    lab.execute(
        datamanager=data_manager,
        num_cores=15,
        forget=False,
        recalculate=False,
        save_on_iteration=None,
        geometry=geometry,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        mesh=mesh,
        n_test=[100],
        n_train=[1000, 10000],
        mwhere=mwhere,
        learn_higher_modes_only=[True],
    )

    # Plots
    palette = sns.color_palette("colorblind")
    k_plot(
        data_manager,
        folder=data_manager.path,
        plot_by=["geometry", "n_train", "m"],
        m=lambda mwhere: mwhere.m,
        mwhere=mwhere,
        axes_by="m",
        add_mwhere=False,
        color_dict={"RF": palette[0], "Tree": palette[2], "LR": palette[4], "Null": palette[5],
                    "Quadratic LR": palette[1], "Degree 4 LR": palette[3]},
    )

    print(f"CO2 emissions: {data_manager.CO2kg:.4f}kg")
    print(f"Power consumption: {data_manager.electricity_consumption_kWh:.4f}kWh")

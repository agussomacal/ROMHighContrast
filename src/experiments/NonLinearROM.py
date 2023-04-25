from collections import namedtuple

import numpy as np
import seaborn as sns
from PerplexityLab.DataManager import DataManager, JOBLIB
from PerplexityLab.LabPipeline import LabPipeline
from PerplexityLab.visualization import perplex_plot, generic_plot
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor

from src import config
from src.lib.SolutionsManagers import SolutionsManagerFEM

ZERO = 1e-15
Bounds = namedtuple("Bounds", "lower upper")
MWhere = namedtuple("MWhere", "m start")


def vn_family_sampler(n_max, geometry, lower_bounds, upper_bounds, mesh):
    np.random.seed(42)
    a = [np.reshape(a_coefs, geometry)
         for a_coefs in zip(*[np.random.uniform(lower_bounds, upper_bounds, n_max)
                              for _ in range(np.prod(geometry))])]
    sm = SolutionsManagerFEM(blocks_geometry=geometry, N=mesh, num_cores=1, method="lsq")
    solutions = sm.generate_solutions(a)
    return {"solution_manager": sm, "a": a, "solutions": solutions}


def do_pca(solutions):
    # This option is not reallistic but more similar to the paper
    # https://github.com/agussomacal/NonLinearRBA4PDEs
    pca = PCA(n_components=np.min((np.shape(solutions)))).fit(solutions)
    pca_projections = pca.transform(solutions)
    return {"pca_projections": pca_projections,
            "explained_variance": pca.explained_variance_,
            "singular_values": pca.singular_values_}


def get_known_unknown_indexes(mwhere, pca_projections, learn_higher_modes_only, only_j=None):
    indexes = np.arange(np.shape(pca_projections)[1], dtype=int)
    known_indexes = indexes[mwhere.start:mwhere.start + mwhere.m]
    only_j = len(indexes) if only_j is None else only_j + mwhere.start + mwhere.m
    unknown_indexes = indexes[mwhere.start + mwhere.m:only_j]
    if not learn_higher_modes_only:
        unknown_indexes = np.append(indexes[:mwhere.start], unknown_indexes)
    return known_indexes, unknown_indexes


def learn_eigenvalues(model: Pipeline):
    def decorated_function(n_train, n_test, pca_projections, mwhere: MWhere, only_j, learn_higher_modes_only=True):
        known_indexes, unknown_indexes = get_known_unknown_indexes(mwhere, pca_projections, learn_higher_modes_only,
                                                                   only_j)
        # always test against the same group of test solutions
        model.fit(pca_projections[n_test:n_test + n_train, known_indexes],
                  pca_projections[n_test:n_test + n_train, unknown_indexes])
        print(np.shape(pca_projections), n_test, known_indexes)
        predictions = model.predict(pca_projections[:n_test, known_indexes])
        print(model, np.shape(predictions))
        error = pca_projections[:n_test, unknown_indexes] - predictions.reshape((-1, len(unknown_indexes)))
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
def k_plot(fig, ax, error, experiments, mwhere, learn_higher_modes_only, n_train, pca_projections, only_j,
            singular_values, label_var="experiments", add_mwhere=False, color_dict=None):
    mse = list(map(lambda e: np.sqrt(np.mean(np.array(e) ** 2, axis=0)).squeeze(), error))

    for i, (exp_i, y_i, ms, lhmo, ntr, pcap, oj) in enumerate(
            zip(experiments, mse, mwhere, learn_higher_modes_only, n_train, pca_projections, only_j)):
        known_indexes, unknown_indexes = get_known_unknown_indexes(ms, pcap, lhmo, oj)
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
        ax.plot(unknown_indexes, y_i, "--", marker=m,
                label=f"{label_i}{f': start={ms.start}, m={ms.m}' if add_mwhere else ''}", c=c)

    ax.plot(np.sort(np.unique(singular_values))[::-1], ":k", label="singular_values", alpha=0.5)
    ticks = ax.get_xticks()
    ax.set_xticks(ticks, [fr"$10^{{{abs(int(t))}}}$" for t in ticks])
    ax.legend(loc='upper right')
    ax.set_yscale("log")
    # ax.set_xscale("log")
    ax.set_ylabel("MSE")


if __name__ == "__main__":
    name = f"FittingEigenvaluesMplus1"
    data_manager = DataManager(
        path=config.results_path,
        name=name,
        format=JOBLIB,
        country_alpha_code="FR",
        trackCO2=True
    )

    # Parameters for experiment
    # geometry = [(2, 2), (4, 4)]
    geometry = [(2, 2)]
    lower_bounds = [1]
    upper_bounds = [100]  # , 1e5
    mesh = [5]
    mwhere = [MWhere(start=0, m=4)]  #
    models = [
        # Pipeline([("Null", NullModel())]),
        Pipeline([("LR", LinearRegression())]),
        Pipeline([("Quadratic", PolynomialFeatures(degree=2)), ("LR", LinearRegression())]),
        Pipeline([("Degree 4", PolynomialFeatures(degree=4)), ("LR", LinearRegression())]),
        Pipeline([("Tree", DecisionTreeRegressor())]),
        Pipeline([("RF", RandomForestRegressor(n_estimators=10))]),
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
        n_max=[25000],
        mwhere=mwhere,
        learn_higher_modes_only=[True],
        only_j=[1, 20],
    )

    data_manager.load()
    generic_plot(
        data_manager,
        x="n_train",
        label="experiments",
        # y="error",
        y="mse",
        plot_func=sns.barplot,
        # plot_func=sns.boxplot,
        mse=lambda error: np.sqrt(np.mean(np.array(error) ** 2)),
        plot_by=["geometry", "upper_bounds"],
        m=lambda mwhere: mwhere.m,
        axes_by=["m", "only_j"],
    )

    # Plots
    palette = sns.color_palette("colorblind")
    k_plot(
        data_manager,
        folder=data_manager.path,
        plot_by=["geometry", "upper_bounds"],
        m=lambda mwhere: mwhere.m,
        mwhere=mwhere,
        axes_by=["m", "n_train", "only_j"],
        add_mwhere=False,
        color_dict={"RF": palette[0], "Tree": palette[2], "LR": palette[4], "Null": palette[5],
                    "Quadratic LR": palette[1], "Degree 4 LR": palette[3]},
    )

    print(f"CO2 emissions: {data_manager.CO2kg:.4f}kg")
    print(f"Power consumption: {data_manager.electricity_consumption_kWh:.4f}kWh")

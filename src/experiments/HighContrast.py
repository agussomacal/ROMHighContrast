import inspect
import itertools
import os
from collections import namedtuple
from multiprocessing import Pool
from pathlib import Path
from time import time
from typing import Callable

import joblib
import dill
import numpy as np
from matplotlib import pylab as plt, cm

from lib.ReducedBasis import ReducedBasisGreedy, INFINIT_A, ReducedBasisRandom, GREEDY_FOR_GALERKIN, GREEDY_FOR_H10
from lib.SolutionsManagers import SolutionsManager
from src.lib.VizUtils import plot_solutions_together, save_fig

MachinePrecision = 1e-13
FIGSIZE = (8, 8)

plt.rcParams.update({'font.size': 14})
from tqdm import tqdm

from src.config import results_path
from lib.SolutionsManagers import SolutionsManagerFEM

TypeOfProblems = namedtuple("TypeOfProblems",
                            "forward_modeling projection state_estimation parameter_estimation_inverse parameter_estimation_linear")
RBErrorDataType = namedtuple("RBErrorDataType", "ReducedBasisName ReducedBasis a2test errors")


def get_not_default_args_names(f: Callable):
    return [k for k, v in inspect.signature(f).parameters.items() if v.default is inspect.Parameter.empty]


reduced_basis_builders = [
    ReducedBasisRandom(),
    ReducedBasisRandom(False),
    ReducedBasisGreedy(greedy_for=GREEDY_FOR_H10),
    ReducedBasisGreedy(greedy_for=GREEDY_FOR_GALERKIN),
]

color_dict = {
    reduced_basis_builders[0].name: "firebrick",
    reduced_basis_builders[1].name: "darkgoldenrod",
    reduced_basis_builders[2].name: "forestgreen",
    reduced_basis_builders[3].name: "royalblue",
}

marker_dict = {
    reduced_basis_builders[0].name: ".",
    reduced_basis_builders[1].name: ".",
    reduced_basis_builders[2].name: ".",
    reduced_basis_builders[3].name: ".",
}


def get_full_a(a_per_block, sm, high_contrast_blocks):
    a = np.ones(((len(a_per_block),) + sm.blocks_geometry))
    for a_vec, hcb_same in zip(a_per_block.T, high_contrast_blocks):
        for ix in hcb_same:
            a[:, ix[0], ix[1]] = a_vec
    return a


def calculate_time(func: Callable, verbose=True):
    def new_func(**kwargs):
        if verbose:
            print(f"calculating {func.__name__}")
        t0 = time()
        res = func(**kwargs)
        t = time() - t0
        if verbose:
            print(f"time spent: {t}")
        return t, res

    return new_func


def calculate_relative_error(sm: SolutionsManager, solutions, approximate_solutions):
    return sm.H10norm(approximate_solutions - solutions) / sm.H10norm(solutions)


def calculate_parameter_estimation_error(difference):
    np.sum(np.sqrt(difference ** 2), axis=(-2, -1))


def get_folder_from_params(name):
    return results_path.joinpath(f"HighContrast_{name}")


def get_data(experiment_path):
    data_path = f"{experiment_path}/data.compressed"
    data = joblib.load(data_path) if os.path.exists(data_path) else dict()
    return data, data_path


def get_a2test_and_train(blocks_geometry, high_contrast_blocks, mesh_discretization_per_dim, diff_coef_refinement,
                         max_num_samples_offline, seed, num_cores=1, method="lsq"):
    sm = SolutionsManagerFEM(blocks_geometry, N=mesh_discretization_per_dim, num_cores=num_cores, method=method)
    a_high_contrast = np.transpose(list(map(np.ravel, np.meshgrid(
        *[1 / np.linspace(1 / INFINIT_A, 1, num=min((diff_coef_refinement * int(np.log2(INFINIT_A)),
                                                     int(np.ceil(
                                                         max_num_samples_offline ** (1 / len(high_contrast_blocks)))))),
                          endpoint=False)] * len(high_contrast_blocks)))))
    np.random.seed(seed)
    a_inf_high_contrast = np.transpose(list(map(np.ravel, np.meshgrid(*[[INFINIT_A, 1]] * len(high_contrast_blocks)))))
    if len(a_high_contrast) > max_num_samples_offline - len(a_inf_high_contrast):
        a_high_contrast = a_high_contrast[
            np.random.choice(len(a_high_contrast), size=max((0, max_num_samples_offline - len(a_inf_high_contrast))),
                             replace=False)]
    a_high_contrast = np.vstack((a_inf_high_contrast, a_high_contrast))
    a = get_full_a(a_high_contrast, sm, high_contrast_blocks)
    return sm, a, a_high_contrast


def experiment(name, reduced_basis_builders=[ReducedBasisGreedy()],
               mesh_discretization_per_dim=6,
               diff_coef_refinement: int = 30,
               vn_max_dim: int = 20, num_measurements: int = 50,
               blocks_geometry=(4, 4),
               high_contrast_blocks=[[(1, 1), (1, 2), (2, 1), (2, 2)]], vn_max_dim2do_stats: int = None,
               recalculate=False, num_cores=1, max_num_samples_offline=10000, seed=42, recalculate_basis=False,
               method="lsqsparse", verbose=True):
    vn_max_dim2do_stats = vn_max_dim if vn_max_dim2do_stats is None else vn_max_dim2do_stats

    # --------- paths and data ---------- #
    experiment_path = get_folder_from_params(name)
    experiment_path.mkdir(parents=True, exist_ok=True)
    data, data_path = get_data(experiment_path)

    if verbose:
        print("\n\n========== ========== =========== ==========")
        print(experiment_path)

    # --------- true solutions calculation/loading ---------- #
    sm, a, a_high_contrast = get_a2test_and_train(
        blocks_geometry, high_contrast_blocks, mesh_discretization_per_dim, diff_coef_refinement,
        max_num_samples_offline, seed, num_cores, method
    )
    if verbose:
        print("Solutions to calculate: ", len(a_high_contrast))
    if recalculate or "solutions" not in data.keys():
        if verbose:
            print("Pre-computing solutions")
        data["time2calculate_solutions"], data["solutions"] = calculate_time(sm.generate_solutions, verbose)(a2try=a)
        data["time2calculate_h1norm"], data["solutions_H1norm"] = calculate_time(sm.H10norm, verbose)(
            solutions=data["solutions"])
        joblib.dump(data, data_path)
    if verbose:
        print(f"time to calculate {len(a)} solutions was {data['time2calculate_solutions']}.")
        print(f"V space of solutions dimension {np.shape(data['solutions'])[1]}.")

    measurement_points = np.random.uniform(size=(num_measurements, 2))
    measurements = sm.evaluate_solutions(measurement_points, data["solutions"])

    # --------- create reduced basis space ---------- #
    for reduced_basis_builder in reduced_basis_builders:
        if reduced_basis_builder.name not in data.keys() or data[reduced_basis_builder.name][
            "basis"].dim < vn_max_dim or recalculate_basis:
            if verbose:
                print(f"Creating full reduced basis {reduced_basis_builder.name}")
            data[reduced_basis_builder.name] = {"errors": {}, "times": {}}
            data[reduced_basis_builder.name]["time2build"], data[reduced_basis_builder.name]["basis"] = \
                calculate_time(reduced_basis_builder.build, verbose)(n=vn_max_dim, sm=sm,
                                                                     solutions2train=data["solutions"],
                                                                     a2train=a, optim_method="lsq",
                                                                     solutions2train_h1norm=data["solutions_H1norm"])
            joblib.dump(data, data_path)
        else:
            data[reduced_basis_builder.name]["basis"].marker = reduced_basis_builder.marker
    reduced_basis_2show = [rb.name for rb in reduced_basis_builders]

    # --------- Calculate errors and statistics ---------- #
    n2try = np.arange(1, vn_max_dim + 1)
    for n in tqdm(n2try, desc="Pre-calculating statistics."):
        if verbose:
            print(f"dim(Vn)={n}")
        for rb_name in reduced_basis_2show:
            if n <= vn_max_dim2do_stats and (recalculate or n not in data[rb_name]["errors"].keys()):
                rb = data[rb_name]["basis"][:n]
                rb.orthonormalize()

                se_time, (c, se_approx_solutions) = calculate_time(rb.state_estimation, verbose)(
                    sm=sm, measurement_points=measurement_points, measurements=measurements, return_coefs=True)
                fm_time, fm_approx_solutions = calculate_time(rb.forward_modeling, verbose)(sm=sm, a=a)
                pj_time, pj_approx_solutions = calculate_time(rb.projection, verbose)(sm=sm,
                                                                                      true_solutions=data["solutions"])
                inv_time, inv_parameters = calculate_time(rb.parameter_estimation_inverse, verbose)(c=c)
                lin_time, lin_parameters = calculate_time(rb.parameter_estimation_linear, verbose)(c=c)

                fm_error = calculate_time(sm.H10norm, verbose)(solutions=fm_approx_solutions - data["solutions"])[1]
                pj_error = calculate_time(sm.H10norm, verbose)(solutions=pj_approx_solutions - data["solutions"])[1]
                se_error = calculate_time(sm.H10norm, verbose)(solutions=se_approx_solutions - data["solutions"])[1]

                data[rb_name]["errors"][n] = TypeOfProblems(
                    forward_modeling=fm_error / data["solutions_H1norm"],
                    projection=pj_error / data["solutions_H1norm"],
                    state_estimation=se_error / data["solutions_H1norm"],
                    parameter_estimation_inverse=np.abs(1 - np.array(rb.parameter_estimation_inverse(c)) / a),
                    parameter_estimation_linear=np.abs(1 - np.array(rb.parameter_estimation_linear(c)) / a)
                )

                data[rb_name]["times"][n] = TypeOfProblems(
                    forward_modeling=fm_time,
                    projection=pj_time,
                    state_estimation=se_time,
                    parameter_estimation_inverse=inv_time,
                    parameter_estimation_linear=lin_time
                )

                joblib.dump(data, data_path)
    return sm, data


type_of_problem_dict = {
    "forward_modeling": "galerkin projection",
    "projection": r"$H_0^1$ projection",
    "state_estimation": "state_estimation",
    "parameter_estimation_inverse": "parameter_estimation_inverse",
    "parameter_estimation_linear": "parameter_estimation_linear"
}


def plot_rates_of_convergence(ax, data, reduced_basis_2show, type_of_problems, color=None, linestyle="solid",
                              marker='.'):
    for i, type_of_problem in enumerate(type_of_problems if isinstance(type_of_problems, list) else [type_of_problems]):
        for j, rb_name in enumerate(reduced_basis_2show):
            rb_stats = data[rb_name]["errors"]
            calculated_ns = sorted(rb_stats.keys())
            linf = [np.max(rb_stats[n][TypeOfProblems._fields.index(type_of_problem)]) for n in calculated_ns]
            ax.plot(calculated_ns, linf,
                    label=f"{rb_name} {': ' + type_of_problem_dict[type_of_problem] if isinstance(type_of_problems, list) else ''}",
                    c=color(data[rb_name]["basis"].name, type_of_problem)
                    if isinstance(color, Callable) else cm.Set1(i * len(reduced_basis_2show) + j),
                    linestyle=linestyle(data[rb_name]["basis"].name, type_of_problem)
                    if isinstance(linestyle, Callable) else linestyle,
                    marker=marker(data[rb_name]["basis"].name, type_of_problem)
                    if isinstance(marker, Callable) else marker)
    ax.set_xlabel(r"$\mathrm{dim}(V_n)$")
    ax.set_ylabel(r"maximal $H^1_0$ error")
    ax.set_yscale("log")
    # handles, labels = fig.gca().get_legend_handles_labels()
    # order = np.argsort(labels)
    # ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
    ax.legend()


def plot_results(name, reduced_basis_builders, a2show, high_contrast_blocks, blocks_geometry,
                 mesh_discretization_per_dim, diff_coef_refinement, max_num_samples_offline, seed, num_cores, method,
                 **kwargs):
    # --------- paths and data ---------- #
    experiment_path = get_folder_from_params(name)
    experiment_path.mkdir(parents=True, exist_ok=True)
    data, data_path = get_data(experiment_path)

    sm, a, a_high_contrast = get_a2test_and_train(
        blocks_geometry, high_contrast_blocks, mesh_discretization_per_dim, diff_coef_refinement,
        max_num_samples_offline, seed, num_cores, method
    )

    reduced_basis_2show = [rb.name for rb in reduced_basis_builders]

    # --------- ---------- ---------- ---------- #
    # ------------ Plotting results ------------ #
    print("Plotting error paths by reduced basis...")
    error_path_path = Path.joinpath(experiment_path, 'ErrorPath')
    error_path_path.mkdir(parents=True, exist_ok=True)
    for i, type_of_problem in enumerate(TypeOfProblems._fields):
        for rb_name in reduced_basis_2show:
            fig, ax = plt.subplots(ncols=1, figsize=FIGSIZE)
            fig.suptitle(f"{type_of_problem.replace('_', ' ')}")
            ax.set_title(f"Reduced basis: {rb_name}")
            rb_stats = data[rb_name]["errors"]
            ahc = 1 / np.max(a_high_contrast, axis=-1)
            order = np.argsort(ahc)
            for n in sorted(rb_stats.keys()):
                error = rb_stats[n][i].max(axis=(-1, -2)) if "parameter_estimation" in type_of_problem else rb_stats[n][
                    i]
                ax.plot(ahc[order], error[order], label=n,
                        marker=None,
                        c=cm.get_cmap('viridis')((n - max(rb_stats.keys())) / max(rb_stats.keys())),  # Spectral
                        # linestyle=plotting_styles4error_paths[N % len(plotting_styles4error_paths)])
                        )
            # ax.axvline(1 / a2show, linestyle='-.', c='k')
            # ax.set_ylim((None, 1))
            ax.set_xlabel(r"$1/y_1$")
            ax.set_ylabel(r"$H^1_0$ error")
            ax.set_yscale("log")
            ax.legend(bbox_to_anchor=(1.01, 0.5), loc="center left")
            plt.savefig(f"{error_path_path}/{name}_{type_of_problem}_error_path_{rb_name}.png")
            plt.close()

    # -------- particular solutions ---------
    print("Plotting particular solutions...")
    if a2show is not None:
        a2show_full = get_full_a(a2show, sm, high_contrast_blocks)
        true_solution_coefs = sm.generate_solutions(a2show_full)
        plot_solutions_together(sm, diffusion_coefficients=None, solutions=true_solution_coefs,
                                num_points_per_dim_to_plot=100,
                                contour_levels=7, axes_xy_proportions=(6, 6))
        plt.savefig(f"{experiment_path}/{name}_TrueSolution.png")
        plt.close()

    print("Plotting rates of convergence...")
    error_rates_path = Path.joinpath(experiment_path, 'ErrorRates')
    error_rates_path.mkdir(parents=True, exist_ok=True)
    for i, type_of_problem in enumerate(TypeOfProblems._fields):
        with save_fig(pathplot=f"{error_rates_path}/{name}_{type_of_problem}_error_rates.png",
                      axes_xy_proportions=FIGSIZE, dpi=None) as ax:
            plot_rates_of_convergence(
                ax, data, reduced_basis_2show, type_of_problem,
                color=lambda rbn, top: color_dict[data[rbn]["basis"].name],
                linestyle=lambda rbn, top: "solid",
                marker=lambda rbn, top: '.',
            )

    # print("Plotting rates of convergence together...")
    # projection_name_dict = {"projection": r"$H^1_0$ projection", "forward_modeling": "Galerkin projection"}
    # projection_linestyle_dict = {"projection": "solid", "forward_modeling": "dashed"}
    # fig, ax = plt.subplots(ncols=1, figsize=FIGSIZE)
    # # ax.set_title("Convergence of error")
    # for type_of_problem in ["projection", "forward_modeling"]:
    #     for rb_name in reduced_basis_2show:
    #         rb_stats = data[rb_name]["errors"]
    #         calculated_ns = sorted(rb_stats.keys())
    #         linf = [np.max(rb_stats[n][TypeOfProblems._fields.index(type_of_problem)]) for n in calculated_ns]
    #         ax.plot(calculated_ns, linf, label=f"{rb_name} {projection_name_dict[type_of_problem]}",
    #                 c=color_dict[data[rb_name]["basis"].name],
    #                 linestyle=projection_linestyle_dict[type_of_problem],
    #                 marker=data[rb_name]["basis"].marker)
    # ax.set_xlabel(r"$\mathrm{dim}(V_n)$")
    # ax.set_ylabel(r"maximal $H^1_0$ error")
    # ax.set_yscale("log")
    # handles, labels = fig.gca().get_legend_handles_labels()
    # order = np.argsort(labels)
    # ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
    # plt.savefig(f"{results_path}/{name}_comparative_error_rates.png")
    # plt.close()
    # print("Plotting finished.")


def gather_experiments(names, high_contrast_blocks_list, reduced_basis_builder=ReducedBasisGreedy(), name="",
                       type_of_problems=None, folder_name="HighContrastDimensionality", **kwargs):
    PROJECTION = "forward_modeling"
    experiment_path = results_path.joinpath(folder_name + name)
    experiment_path.mkdir(exist_ok=True, parents=True)
    print("Plotting rates of convergence...")
    for i, type_of_problem in enumerate(TypeOfProblems._fields):
        if type_of_problems is not None and type_of_problem not in type_of_problems:
            continue

        fig, ax = plt.subplots(ncols=1, figsize=FIGSIZE)
        if type_of_problem == PROJECTION:
            fig_log, ax_log = plt.subplots(ncols=1, figsize=FIGSIZE)
            # ax_log.set_title(f"{reduced_basis_builder.name}: {type_of_problem}")
        # ax.set_title(f"{reduced_basis_builder.name}: {type_of_problem}")
        for j, (name, high_contrast_blocks) in enumerate(zip(names, high_contrast_blocks_list)):
            sub_experiment_path = get_folder_from_params(name)
            data, data_path = get_data(sub_experiment_path)
            rb_stats = data[reduced_basis_builder.name]["errors"]
            calculated_ns = np.array(list(sorted(rb_stats.keys())))
            linf = np.array([np.max(rb_stats[n][i]) for n in calculated_ns])

            linf_log = -np.log(linf)
            calculated_ns_log = np.log(calculated_ns)

            label = f"d: {len(high_contrast_blocks)}"
            c = cm.Set1(j)
            if type_of_problem == PROJECTION:
                rate, origin = np.ravel(np.linalg.lstsq(
                    np.vstack([calculated_ns[linf > MachinePrecision], np.ones(np.sum(linf > MachinePrecision))]).T,
                    np.log(linf[linf > MachinePrecision]).reshape((-1, 1)), rcond=None)[0])
                ax.plot(calculated_ns[linf > MachinePrecision],
                        np.exp(rate * calculated_ns[linf > MachinePrecision] + origin), ":", c=c, alpha=0.7)

                rate_log, origin_log = np.ravel(np.linalg.lstsq(
                    np.vstack([calculated_ns_log[linf > MachinePrecision], np.ones(np.sum(linf > MachinePrecision))]).T,
                    np.log(linf_log[linf > MachinePrecision]).reshape((-1, 1)), rcond=None)[0])
                ax_log.plot(calculated_ns[linf > MachinePrecision],
                            np.exp(rate_log * calculated_ns_log[linf > MachinePrecision] + origin_log), ":", c=c,
                            alpha=0.7)

                label_log = label + f" {rate_log:.2f}"
                label = label + f" {rate:.2f}"
                ax_log.plot(calculated_ns, linf_log, label=label_log, c=c, linestyle="--", marker=".")
                ax_log.set_xscale("log")
                ax_log.set_yscale("log")
                ax_log.set_xlabel(r"$\mathrm{dim}(V_n)$")
                ax_log.set_ylabel(r"log(maximal $H^1_0$ error)")

            ax.plot(calculated_ns, linf, label=label, c=c, linestyle="--", marker=".")

        # ax.plot(calculated_ns, np.exp(-calculated_ns ** (1 / 2)), ".-k", alpha=0.5, label=r"e")

        ax.set_xlabel(r"$\mathrm{dim}(V_n)$")
        ax.set_ylabel(r"maximal $H^1_0$ error")
        ax.set_yscale("log")
        ax.legend()

        if type_of_problem == PROJECTION:
            ax_log.legend()
            fig_log.savefig(f"{experiment_path}/{name}_{type_of_problem}_error_rates_loglog.png")
        fig.savefig(f"{experiment_path}/{name}_{type_of_problem}_error_rates_log.png")
        plt.close("all")

    print("Plotting finished.")


def paper_plots(names, high_contrast_blocks_list, reduced_basis_builders):
    experiment_path = results_path.joinpath("paper_plots")
    experiment_path.mkdir(exist_ok=True, parents=True)

    print("Plotting rates of convergence...")
    for name, high_contrast_blocks in zip(names, high_contrast_blocks_list):
        sub_experiment_path = get_folder_from_params(name)
        data, data_path = get_data(sub_experiment_path)
        if len(high_contrast_blocks) == 1:
            # all reduced basis
            reduced_basis_2show = [rb.name for rb in reduced_basis_builders]
            for type_of_problem in ["forward_modeling", "projection"]:
                with save_fig(pathplot=f"{experiment_path}/{name}_{type_of_problem}_error_rates.png",
                              axes_xy_proportions=FIGSIZE, dpi=None) as ax:
                    plot_rates_of_convergence(
                        ax, data, reduced_basis_2show=reduced_basis_2show,
                        type_of_problems=type_of_problem,
                        color=lambda rbn, top: color_dict[data[rbn]["basis"].name],
                        linestyle="solid",
                        marker='.',
                    )
        elif len(high_contrast_blocks) == 2:
            # only greedy algorithms reduced basis
            reduced_basis_2show = [rb.name for rb in reduced_basis_builders if "Greedy" in rb.name]
            with save_fig(pathplot=f"{experiment_path}/{name}_{type_of_problem}_error_rates.png",
                          axes_xy_proportions=FIGSIZE, dpi=None) as ax:
                plot_rates_of_convergence(
                    ax, data, reduced_basis_2show=reduced_basis_2show,
                    type_of_problems=["forward_modeling", "projection"],
                    linestyle=lambda rbn, top: "solid" if top == "projection" else "dashed",
                    marker=lambda rbn, top: "." if top == "projection" else "*"
                )

    print("Plotting rates of convergence...")
    reduced_basis_builder = ReducedBasisGreedy(GREEDY_FOR_GALERKIN)
    common_name = os.path.commonprefix(names)
    type_of_problem = "forward_modeling"
    with save_fig(pathplot=f"{experiment_path}/{common_name}_{type_of_problem}_dimensional_deterioration.png",
                  axes_xy_proportions=FIGSIZE, dpi=None) as ax:
        for j, (name, high_contrast_blocks) in enumerate(zip(names, high_contrast_blocks_list)):
            sub_experiment_path = get_folder_from_params(name)
            data, data_path = get_data(sub_experiment_path)
            rb_stats = data[reduced_basis_builder.name]["errors"]
            calculated_ns = np.array(list(sorted(rb_stats.keys())))
            linf = np.array([np.max(rb_stats[n][TypeOfProblems._fields.index(type_of_problem)]) for n in calculated_ns])

            label = f"d: {len(high_contrast_blocks)}"
            c = cm.Set1(j)
            rate, origin = np.ravel(np.linalg.lstsq(
                np.vstack([calculated_ns[linf > MachinePrecision], np.ones(np.sum(linf > MachinePrecision))]).T,
                np.log(linf[linf > MachinePrecision]).reshape((-1, 1)), rcond=None)[0])
            ax.plot(calculated_ns[linf > MachinePrecision],
                    np.exp(rate * calculated_ns[linf > MachinePrecision] + origin), ":", c=c, alpha=0.7)
            label = label + f" {rate:.2f}"
            ax.plot(calculated_ns, linf, label=label, c=c, linestyle="--", marker=".")
        ax.set_xlabel(r"$\mathrm{dim}(V_n)$")
        ax.set_ylabel(r"maximal $H^1_0$ error")
        ax.set_yscale("log")
        ax.legend()


if __name__ == "__main__":
    general_params = {
        "reduced_basis_builders": reduced_basis_builders,
        "mesh_discretization_per_dim": 20,  # 20
        "diff_coef_refinement": 10,  # 30
        "num_measurements": 50,
        "num_cores": 1,
        "max_num_samples_offline": 1000,
        "seed": 42,
        "vn_max_dim": 15,
        "vn_max_dim2do_stats": None,  # 6 None,
        "recalculate": False,
        "recalculate_basis": False,
        "blocks_geometry": (4, 4),
        "method": "lsqsparse",
        "verbose": True
    }

    # lsq (no sparse) 1 core: 113.14256739616394
    # lsq (no sparse) 15 cores: 221.70560717582703
    # lsqsparse 1 core: 37.1559112071991
    # lsqsparse 15 cores: 124.76248121261597
    high_contrast_blocks = [[(0, 1)], [(1, 3)], [(2, 1), (2, 2), (2, 3)]]
    complement = set(itertools.product(range(4), range(4)))
    for e in high_contrast_blocks:
        complement = complement.difference(set(e))
    high_contrast_blocks.append(list(complement))
    names = [f"{general_params['mesh_discretization_per_dim']}_GeomAssumptionsD{i + 1}" for i in
             range(len(high_contrast_blocks))]
    high_contrast_blocks_list = [high_contrast_blocks[:i + 1] for i in range(len(high_contrast_blocks))]


    def par_func(x):
        experiment(name=x[0], high_contrast_blocks=x[1], **general_params)
        plot_results(name=x[0], high_contrast_blocks=x[1], a2show=np.array([INFINIT_A] * len(x[1])), **general_params)


    # list(Pool(4).map(par_func, zip(names, high_contrast_blocks_list)))
    # gather_experiments(names=names, high_contrast_blocks_list=high_contrast_blocks_list,
    #                    reduced_basis_builder=ReducedBasisGreedy(greedy_for=GREEDY_FOR_GALERKIN),
    #                    name=f"Geom_{general_params['mesh_discretization_per_dim']}", **general_params)
    paper_plots(names, high_contrast_blocks_list, reduced_basis_builders)

    high_contrast_blocks = [
        [(0, 0), (1, 1), (2, 2), (3, 3)],
        [(0, 2), (1, 3), (2, 0), (3, 1)],
        [(1, 0), (0, 1), (3, 2), (2, 3)],
        [(0, 3), (1, 2), (2, 1), (3, 0)]
    ]
    names = [f"{general_params['mesh_discretization_per_dim']}_NotGeomAssumptionsD{i + 1}" for i in
             range(len(high_contrast_blocks))]
    high_contrast_blocks_list = [high_contrast_blocks[:i + 1] for i in range(len(high_contrast_blocks))]

    list(Pool(4).map(par_func, zip(names, high_contrast_blocks_list)))
    # gather_experiments(names=names, high_contrast_blocks_list=high_contrast_blocks_list,
    #                    reduced_basis_builder=ReducedBasisGreedy(greedy_for=GREEDY_FOR_GALERKIN),
    #                    name=f"NotGeom_{general_params['mesh_discretization_per_dim']}", **general_params)
    paper_plots(names, high_contrast_blocks_list, reduced_basis_builders)
    # 6241

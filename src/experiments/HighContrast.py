import os
from collections import namedtuple
from pathlib import Path
from time import time
from typing import Callable

import joblib
import dill
import numpy as np
from matplotlib import pylab as plt, cm

from lib.ReducedBasis import ReducedBasisGreedy, INFINIT_A, ReducedBasisRandom, GREEDY_FOR_GALERKIN, GREEDY_FOR_H10
from lib.SolutionsManagers import SolutionsManager
from src.lib.VizUtils import plot_solutions_together

plt.rcParams.update({'font.size': 14})
from tqdm import tqdm

from src.config import results_path
from lib.SolutionsManagers import SolutionsManagerFEM

TypeOfProblems = namedtuple("TypeOfProblems",
                            "forward_modeling projection state_estimation parameter_estimation_inverse parameter_estimation_linear")
RBErrorDataType = namedtuple("RBErrorDataType", "ReducedBasisName ReducedBasis a2test errors")
SOLVER_POLYNOMIAL = "polynomials"
SOLVER_FEM = "fem"

plotting_styles4error_paths = [
    "-",
    "dotted",
    "-.",
    "--",
]


def get_full_a(a_per_block, sm, high_contrast_blocks):
    a = np.ones(((len(a_per_block),) + sm.blocks_geometry))
    for a_vec, hcb_same in zip(a_per_block.T, high_contrast_blocks):
        for ix in hcb_same:
            a[:, ix[0], ix[1]] = a_vec
    return a


def calculate_time(func: Callable):
    def new_func(**kwargs):
        print(f"calculating {func.__name__}")
        t0 = time()
        res = func(**kwargs)
        t = time() - t0
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


def experiment(name, reduced_basis_builders=[ReducedBasisGreedy()], greedy_for="projection",
               mesh_discretization_per_dim=6,
               diff_coef_refinement: int = 30, vn_max_dim: int = 20, num_measurements: int = 50,
               a2show=None, blocks_geometry=(4, 4),
               high_contrast_blocks=[[(1, 1), (1, 2), (2, 1), (2, 2)]],
               recalculate=False, num_cores=1, max_num_samples_offline=10000, seed=42):
    # --------- paths and data ---------- #
    experiment_path = get_folder_from_params(name)
    experiment_path.mkdir(parents=True, exist_ok=True)
    data, data_path = get_data(experiment_path)

    print("\n\n========== ========== =========== ==========")
    print(experiment_path)

    # --------- true solutions calculation/loading ---------- #
    sm = SolutionsManagerFEM(blocks_geometry, N=mesh_discretization_per_dim, num_cores=num_cores)
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

    print("Solutions to calculate: ", len(a_high_contrast))
    a = get_full_a(a_high_contrast, sm, high_contrast_blocks)

    if "solutions" not in data.keys():
        print("Pre-computing solutions")
        data["time2calculate_solutions"], data["solutions"] = calculate_time(sm.generate_solutions)(a2try=a)
        data["time2calculate_h1norm"], data["solutions_H1norm"] = calculate_time(sm.H10norm)(
            solutions=data["solutions"])
        joblib.dump(data, data_path)
    print(f"time to calculate {len(a)} solutions was {data['time2calculate_solutions']}.")
    print(f"V space of solutions dimension {np.shape(data['solutions'])[1]}.")

    measurement_points = np.random.uniform(size=(num_measurements, 2))
    measurements = sm.evaluate_solutions(measurement_points, data["solutions"])

    # --------- create reduced basis space ---------- #
    for reduced_basis_builder in reduced_basis_builders:
        if reduced_basis_builder.name not in data.keys() or data[reduced_basis_builder.name]["basis"].dim < vn_max_dim:
            print(f"Creating full reduced basis {reduced_basis_builder.name}")
            data[reduced_basis_builder.name] = {"errors": {}, "times": {}}
            data[reduced_basis_builder.name]["time2build"], data[reduced_basis_builder.name]["basis"] = \
                calculate_time(reduced_basis_builder.build)(n=vn_max_dim, sm=sm, solutions2train=data["solutions"],
                                                            a2train=a, optim_method="lsq", greedy_for=greedy_for,
                                                            solutions2train_h1norm=data["solutions_H1norm"])
            joblib.dump(data, data_path)
        else:
            data[reduced_basis_builder.name]["basis"].marker = reduced_basis_builder.marker
    reduced_basis_2show = [rb.name for rb in reduced_basis_builders]

    # --------- Calculate errors and statistics ---------- #
    n2try = np.arange(1, vn_max_dim + 1)
    for n in tqdm(n2try, desc="Pre-calculating statistics."):
        print(f"dim(Vn)={n}")
        for rb_name in reduced_basis_2show:
            if recalculate or n not in data[rb_name]["errors"].keys():
                rb = data[rb_name]["basis"][:n]
                rb.orthonormalize()

                se_time, (c, se_approx_solutions) = calculate_time(rb.state_estimation)(
                    sm=sm, measurement_points=measurement_points, measurements=measurements, return_coefs=True)
                fm_time, fm_approx_solutions = calculate_time(rb.forward_modeling)(sm=sm, a=a)
                pj_time, pj_approx_solutions = calculate_time(rb.projection)(sm=sm, true_solutions=data["solutions"])
                inv_time, inv_parameters = calculate_time(rb.parameter_estimation_inverse)(c=c)
                lin_time, lin_parameters = calculate_time(rb.parameter_estimation_linear)(c=c)

                fm_error = calculate_time(sm.H10norm)(solutions=fm_approx_solutions - data["solutions"])[1]
                pj_error = calculate_time(sm.H10norm)(solutions=pj_approx_solutions - data["solutions"])[1]
                se_error = calculate_time(sm.H10norm)(solutions=se_approx_solutions - data["solutions"])[1]

                data[rb_name]["errors"][n] = TypeOfProblems(
                    forward_modeling=fm_error / data["solutions_H1norm"],
                    projection=pj_error / data["solutions_H1norm"],
                    state_estimation=se_error / data["solutions_H1norm"],
                    parameter_estimation_inverse=np.abs(1 / a - 1 / np.array(rb.parameter_estimation_inverse(c))),
                    parameter_estimation_linear=np.abs(1 / a - 1 / np.array(rb.parameter_estimation_linear(c)))
                )

                data[rb_name]["times"][n] = TypeOfProblems(
                    forward_modeling=fm_time,
                    projection=pj_time,
                    state_estimation=se_time,
                    parameter_estimation_inverse=inv_time,
                    parameter_estimation_linear=lin_time
                )

                joblib.dump(data, data_path)

    # [k for k in data.keys() if k not in ["solutions", "time2calculate_solutions"]]

    # --------- ---------- ---------- ---------- #
    # ------------ Plotting results ------------ #
    print("Plotting error paths by reduced basis...")
    error_path_path = Path.joinpath(experiment_path, 'ErrorPath')
    error_path_path.mkdir(parents=True, exist_ok=True)
    for i, type_of_problem in enumerate(TypeOfProblems._fields):
        for rb_name in reduced_basis_2show:
            fig, ax = plt.subplots(ncols=1, figsize=(12, 6))
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
                        c=cm.get_cmap('viridis')((n - n2try[0]) / len(n2try)),  # Spectral
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
        fig, ax = plt.subplots(ncols=1, figsize=(12, 6))
        ax.set_title(type_of_problem)
        for rb_name in reduced_basis_2show:
            rb_stats = data[rb_name]["errors"]
            calculated_ns = sorted(rb_stats.keys())
            linf = [np.max(rb_stats[n][i]) for n in calculated_ns]
            ax.plot(calculated_ns, linf, label=rb_name, c=data[rb_name]["basis"].color,
                    linestyle=data[rb_name]["basis"].linestyle, marker=data[rb_name]["basis"].marker)
        ax.set_xlabel(r"$\mathrm{dim}(V_n)$")
        ax.set_ylabel(r"maximal $H^1_0$ error")
        ax.set_yscale("log")
        handles, labels = fig.gca().get_legend_handles_labels()
        order = np.argsort(labels)
        ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
        plt.savefig(f"{error_rates_path}/{name}_{type_of_problem}_error_rates.png")
        plt.close()

    print("Plotting rates of convergence together...")
    projection_name_dict = {"projection": r"$H^1_0$ projection", "forward_modeling": "Galerkin projection"}
    projection_linestyle_dict = {"projection": "solid", "forward_modeling": "dashed"}
    fig, ax = plt.subplots(ncols=1, figsize=(12, 6))
    ax.set_title("Convergence of error")
    for type_of_problem in ["projection", "forward_modeling"]:
        for rb_name in reduced_basis_2show:
            rb_stats = data[rb_name]["errors"]
            calculated_ns = sorted(rb_stats.keys())
            linf = [np.max(rb_stats[n][TypeOfProblems._fields.index(type_of_problem)]) for n in calculated_ns]
            ax.plot(calculated_ns, linf, label=f"{rb_name} {projection_name_dict[type_of_problem]}",
                    c=data[rb_name]["basis"].color,
                    linestyle=projection_linestyle_dict[type_of_problem], marker=data[rb_name]["basis"].marker)
    ax.set_xlabel(r"$\mathrm{dim}(V_n)$")
    ax.set_ylabel(r"maximal $H^1_0$ error")
    ax.set_yscale("log")
    handles, labels = fig.gca().get_legend_handles_labels()
    order = np.argsort(labels)
    ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
    plt.savefig(f"{results_path}/{name}_comparative_error_rates.png")
    plt.close()
    print("Plotting finished.")


def gather_experiments(names, high_contrast_blocks_list, reduced_basis_builder=ReducedBasisGreedy, **kwargs):
    experiment_path = results_path.joinpath("HighContrastDimensionality")
    experiment_path.mkdir(exist_ok=True, parents=True)
    print("Plotting rates of convergence...")
    for i, type_of_problem in enumerate(TypeOfProblems._fields):
        fig, ax = plt.subplots(ncols=1, figsize=(12, 6))
        if type_of_problem == "projection":
            fig_log, ax_log = plt.subplots(ncols=1, figsize=(12, 6))
            ax_log.set_title(f"{reduced_basis_builder.name}: {type_of_problem}")
        ax.set_title(f"{reduced_basis_builder.name}: {type_of_problem}")
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
            if type_of_problem == "projection":
                rate, origin = np.ravel(np.linalg.lstsq(
                    np.vstack([calculated_ns, np.ones(len(linf))]).T,
                    np.log(linf).reshape((-1, 1)), rcond=None)[0])
                ax.plot(calculated_ns, np.exp(rate * calculated_ns + origin), ":", c=c, alpha=0.7)

                rate_log, origin_log = np.ravel(np.linalg.lstsq(
                    np.vstack([calculated_ns_log, np.ones(len(linf))]).T,
                    np.log(linf_log).reshape((-1, 1)), rcond=None)[0])
                ax_log.plot(calculated_ns, np.exp(rate_log * calculated_ns_log + origin_log), ":", c=c, alpha=0.7)

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

        if type_of_problem == "projection":
            ax_log.legend()
            fig_log.savefig(f"{experiment_path}/{type_of_problem}_error_rates_loglog.png")
        fig.savefig(f"{experiment_path}/{type_of_problem}_error_rates_log.png")
        plt.close("all")

    print("Plotting finished.")


if __name__ == "__main__":
    general_params = {
        "reduced_basis_builders": [ReducedBasisGreedy(greedy_for=GREEDY_FOR_H10),
                                   ReducedBasisGreedy(greedy_for=GREEDY_FOR_GALERKIN),
                                   ReducedBasisRandom()],
        "mesh_discretization_per_dim": 10,
        "diff_coef_refinement": 10,  # 30
        "num_measurements": 50,
        "num_cores": 15,
        "max_num_samples_offline": 1000,
        "seed": 42,
        "recalculate": False
    }

    # ---------- 2x2 block geometry ---------- #
    # general_params["vn_max_dim"] = 16
    # experiment(
    #     name="single2x2",
    #     a2show=np.array([INFINIT_A]),
    #     blocks_geometry=(2, 2),
    #     high_contrast_blocks=[[(0, 0)]],
    #     **general_params
    # )
    # experiment(
    #     name="opposite2x2",
    #     a2show=np.array([INFINIT_A]),
    #     blocks_geometry=(2, 2),
    #     high_contrast_blocks=[[(0, 0), (1, 1)]],
    #     **general_params
    # )

    # # ---------- 3x3 block geometry ---------- #
    general_params["vn_max_dim"] = 40
    experiment(
        name="opposite4x4",
        a2show=np.array([INFINIT_A]),
        blocks_geometry=(4, 4),
        high_contrast_blocks=[[(0, 1), (1, 2)]],
        **general_params
    )
    experiment(
        name="checkerboard",
        a2show=np.array([INFINIT_A]),
        blocks_geometry=(4, 4),
        high_contrast_blocks=[[(0, 0), (0, 2), (1, 1), (1, 3), (2, 0), (2, 2), (3, 1), (3, 3)]],
        **general_params
    )

    high_contrast_blocks = [(1, 1), (0, 3), (3, 1), (2, 3)]
    names = [f"d{i + 1}" for i in range(len(high_contrast_blocks))]
    high_contrast_blocks = list(map(lambda x: [x], high_contrast_blocks))
    high_contrast_blocks_list = [high_contrast_blocks[:i + 1] for i in range(len(high_contrast_blocks))]
    for name, hcb in zip(names, high_contrast_blocks_list):
        experiment(
            name=name,
            a2show=np.array([INFINIT_A]),
            blocks_geometry=(4, 4),
            high_contrast_blocks=hcb,
            **general_params
        )

    gather_experiments(names=names, high_contrast_blocks_list=high_contrast_blocks_list, **general_params)
    # np.array([-2.29, -0.73, -0.4, -0.3])*np.arange(1,5) #-> array([-2.29, -1.46, -1.2 , -1.2 ])

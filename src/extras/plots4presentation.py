import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm
from matplotlib.collections import LineCollection
import seaborn as sns

from src import config
from src.lib.SolutionsManagers import SolutionsManagerFEM
from src.lib.VizUtils import plot_solutions_together

# ============= ============= ============= ============= #
# Parameters
# ============= ============= ============= ============= #
presentation_path = Path.joinpath(config.results_path, 'Presentation')
presentation_path.mkdir(parents=True, exist_ok=True)

num_snapshots = 10
num_snapshots_optim = 100
num_points_per_dim_to_plot = 100  # size of the plotted images
axes_xy_proportions = (4, 4)
number_of_measures = 50

contour_levels = 0
# cmap = matplotlib.colormaps['coolwarm'].resampled(contour_levels)
cmap = "coolwarm"
cmapf = matplotlib.colormaps.get_cmap(cmap)

# object that knows how to create the solutions to the PDE + projections etc
sm = SolutionsManagerFEM(blocks_geometry=(2, 2), N=15, num_cores=1)
print("The space V has dimension {}".format(sm.vspace_dim))


# ============= ============= ============= ============= #
# Useful functions
# ============= ============= ============= ============= #
def save_fig_without_white(filename):
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(filename, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()


def measurements_sampling_method_uniform(number_of_measures, xlim, ylim, seed=42, **kwargs) -> np.ndarray:
    np.random.seed(seed)
    return np.array([(np.random.uniform(*xlim), np.random.uniform(*ylim)) for _ in range(number_of_measures)])


# ============= ============= ============= ============= #
# Multiple samples
# ============= ============= ============= ============= #
T = np.linspace(0, 2 * np.pi, num_snapshots)
parameters_dict = list()
measurements = list()
for i, t in enumerate(T):
    # define the matrix of diffusion coefficients \theta for each solution u to be computed.
    y = np.array(5 + 4 * np.array([[
        [np.sin(t), np.sin(2 * t)],
        [np.sin(3 * t), np.sin(4 * t)],
    ]]), dtype=int)
    parameters_dict.append(y[0].tolist())
    # generate as many solutions as diffusion coefficients matrices given
    u = sm.generate_solutions(a2try=y[:, ::-1])
    print("Solutions shape: ", np.shape(u))

    plot_solutions_together(
        sm,
        diffusion_coefficients=np.array(y),
        solutions=u,
        num_points_per_dim_to_plot=num_points_per_dim_to_plot,
        contour_levels=contour_levels,
        axes_xy_proportions=axes_xy_proportions,
        titles=False,
        colorbar=False,
        measurement_points=None,
        cmap=cmap, add_grid=True)
    save_fig_without_white(f"{presentation_path}/solutions_{i}.png")

# ============= ============= ============= ============= #
# Optimization
# ============= ============= ============= ============= #
T = np.linspace(2 * np.pi / 4, np.pi * 3 / 4, num_snapshots_optim)
parameters_optim_dict = list()
u = list()
for i, t in enumerate(T):
    # define the matrix of diffusion coefficients \theta for each solution u to be computed.
    y = np.array(5 + 4 * np.array([[
        [np.sin(t), np.sin(2 * t)],
        [np.sin(3 * t), np.sin(4 * t)],
    ]]))
    parameters_optim_dict.append(y[0].tolist())
    # generate as many solutions as diffusion coefficients matrices given
    u = sm.generate_solutions(a2try=y[:, ::-1])

    plot_solutions_together(
        sm,
        diffusion_coefficients=np.array(y),
        solutions=u,
        num_points_per_dim_to_plot=num_points_per_dim_to_plot,
        contour_levels=contour_levels,
        axes_xy_proportions=axes_xy_proportions,
        titles=False,
        colorbar=False,
        measurement_points=None,
        cmap=cmap, add_grid=True)
    save_fig_without_white(f"{presentation_path}/solutions_optim_{i}.png")

# ============= ============= ============= ============= #
# Inverse problem pointwise increase
# ============= ============= ============= ============= #
points = measurements_sampling_method_uniform(number_of_measures, xlim=sm.x_domain, ylim=sm.y_domain, seed=42)
measurements = sm.evaluate_solutions(
    np.concatenate((points[:, 0].reshape((-1, 1)), points[:, 1].reshape((-1, 1))), axis=1),
    solutions=[u]).ravel().tolist()

# uses the las t y,u from the optimization
for i in range(number_of_measures):
    plot_solutions_together(
        sm,
        diffusion_coefficients=np.array(y),
        solutions=u,
        num_points_per_dim_to_plot=num_points_per_dim_to_plot,
        contour_levels=contour_levels,
        axes_xy_proportions=axes_xy_proportions,
        titles=False,
        colorbar=False,
        measurement_points=points[:i + 1, :], measurements_color="white",
        cmap=cmap, add_grid=True)
    save_fig_without_white(f"{presentation_path}/solutions_point_measurements_increase_{i}.png")

    norm = matplotlib.colors.Normalize(vmin=0, vmax=np.max(measurements))
    sns.barplot(data=pd.DataFrame(np.transpose([measurements[:i + 1] + [0] * (number_of_measures - i - 1),
                                                list(range(1, number_of_measures + 1))]),
                                  columns=["measurements", "sensors", ]),
                x="sensors", y="measurements", palette={j + 1: cmapf(norm(m)) for j, m in enumerate(measurements)})
    plt.ylim([0, 0.11])
    # plt.xlim([1, number_of_measures])
    plt.xlabel("")
    plt.ylabel("")
    # plt.xlabel(r"Sensors $i$")
    # plt.ylabel(r"Measurements $z_i = \ell_i(u)$")
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
    plt.box(False)
    plt.savefig(f"{presentation_path}/barplot_measurements_increase_{i}.png", bbox_inches='tight', pad_inches=0,
                transparent=True)
    plt.close()
    # save_fig_without_white(f"{presentation_path}/barplot_measurements_increase_{i}.jpeg")

# T = np.linspace(0, 2 * np.pi, num_snapshots)
# parameters_dict = list()
# measurements = list()
# for i, t in enumerate(T):
#     # define the matrix of diffusion coefficients \theta for each solution u to be computed.
#     y = np.array(5 + 4 * np.array([[
#         [np.sin(t), np.sin(2 * t)],
#         [np.sin(3 * t), np.sin(4 * t)],
#     ]]), dtype=int)
#     parameters_dict.append(y[0].tolist())
#     # generate as many solutions as diffusion coefficients matrices given
#     u = sm.generate_solutions(a2try=y[:, ::-1])
#     print("Solutions shape: ", np.shape(u))
#
#     plot_solutions_together(
#         sm,
#         diffusion_coefficients=np.array(y),
#         solutions=u,
#         num_points_per_dim_to_plot=num_points_per_dim_to_plot,
#         contour_levels=0,
#         axes_xy_proportions=axes_xy_proportions,
#         titles=False,
#         colorbar=False,
#         measurement_points=None,
#         cmap="coolwarm", add_grid=True)
#     save_fig_without_white(f"{presentation_path}/solutions_{i}.jpeg")
#
#     plot_solutions_together(
#         sm,
#         diffusion_coefficients=np.array(y),
#         solutions=u,
#         num_points_per_dim_to_plot=num_points_per_dim_to_plot,
#         contour_levels=0,
#         axes_xy_proportions=axes_xy_proportions,
#         titles=False,
#         colorbar=False,
#         measurement_points=points,
#         cmap="coolwarm", add_grid=True)
#     save_fig_without_white(f"{presentation_path}/solutions_points_{i}.svg")
#
#     measurements.append(
#         sm.evaluate_solutions(np.concatenate((points[:, 0].reshape((-1, 1)), points[:, 1].reshape((-1, 1))), axis=1),
#                               solutions=[u]).ravel().tolist())

fig, ax = plt.subplots(1, 1, figsize=axes_xy_proportions)
x, y = np.meshgrid(sm.points_r, sm.points_c)
plt.scatter(x, y, c="white", alpha=1, marker="o", s=25)
# plt.scatter(x, y, c="black", alpha=1, marker="o", s=10)
segs1 = np.stack((x, y), axis=2)
segs2 = segs1.transpose(1, 0, 2)
plt.gca().add_collection(LineCollection(segs1, edgecolors="white", linewidths=1.5, alpha=0.7))
plt.gca().add_collection(LineCollection(segs2, edgecolors="white", linewidths=1.5, alpha=0.7))
save_fig_without_white(f"{presentation_path}/grid.png")

with open(f"{presentation_path}/metadata.json", "w") as f:
    json.dump(
        {
            "parameters_dict": parameters_dict,
            "parameters_optim_dict": parameters_optim_dict,
            "num_snapshots_optim": num_snapshots_optim,
            "num_snapshots": num_snapshots,
            "measurements": measurements,
            "number_of_measures": number_of_measures,
            "Vdim": sm.vspace_dim
        },
        f
    )

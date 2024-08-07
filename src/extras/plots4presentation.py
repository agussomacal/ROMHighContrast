import json
from pathlib import Path
from typing import Union, Tuple, List

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm, ticker
from matplotlib.collections import LineCollection
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

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
measurements_color = "black"

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


def barplot_measurements(filename, measurements, max_measurements):
    number_of_measures = len(measurements)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=max_measurements)
    sns.barplot(data=pd.DataFrame(np.transpose([measurements, list(range(1, number_of_measures + 1))]),
                                  columns=["measurements", "sensors", ]),
                x="sensors", y="measurements", palette={j + 1: cmapf(norm(m)) for j, m in enumerate(measurements)})
    plt.ylim([0, 0.11])
    # plt.xlim([1, number_of_measures])
    plt.xlabel(r"")
    plt.ylabel(r"")
    # plt.xlabel(r"Sensors $i$")
    # plt.ylabel(r"Measurements $z_i = \ell_i(u)$")
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
    plt.box(False)
    # ax.xaxis.label.set_color('white')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()


def space_y(t, r=4):
    return 5 + r * np.array([
        [np.sin(t), np.sin(2 * t)],
        [np.sin(3 * t), np.sin(4 * t)],
    ])


def measurements_sampling_method_uniform(number_of_measures, xlim, ylim, seed=42, **kwargs) -> np.ndarray:
    np.random.seed(seed)
    return np.array([(np.random.uniform(*xlim), np.random.uniform(*ylim)) for _ in range(number_of_measures)])


# def measurements_sampling_method_grid(number_of_measures_per_dim, xlim, ylim, seed=42, **kwargs) -> np.ndarray:
#     np.random.seed(seed)
#     return np.array(
#         [(xlim[0] + i / (xlim[1] - xlim[0]), ylim[0] + j / (ylim[1] - ylim[0])) for i in range(number_of_measures_per_dim+1) for j
#          in range(number_of_measures_per_dim+1)])
# measurements_sampling_method_grid(2, sm.x_domain, sm.y_domain, seed=42)

def measurements_sampling_method_grid(number_of_measures, xlim, ylim) -> np.ndarray:
    n_per_dim = int(np.ceil(np.sqrt(number_of_measures)))
    x, y = np.meshgrid(*[np.linspace(*xlim, num=n_per_dim + 2)[1:-1],
                         np.linspace(*ylim, num=n_per_dim + 2)[1:-1]])
    measurement_points = np.concatenate([x.reshape((-1, 1)), y.reshape((-1, 1))], axis=1)
    return measurement_points


def calculate_averages_from_image(image, num_cells_per_dim: Union[int, Tuple[int, int]]):
    # Example of how to calculate the averages in a single pass:
    # np.arange(6 * 10).reshape((6, 10)).reshape((2, 3, 5, 2)).mean(-1).mean(-2)
    img_x, img_y = np.shape(image)
    ncx, ncy = (num_cells_per_dim, num_cells_per_dim) if isinstance(num_cells_per_dim, int) else num_cells_per_dim
    return image.reshape((ncx, img_x // ncx, ncy, img_y // ncy)).mean(-1).mean(-2)


def make_image_high_resolution(matrix, reconstruction_factor):
    resolution_factor = reconstruction_factor if isinstance(reconstruction_factor, (list, tuple, np.ndarray)) else (
        reconstruction_factor, reconstruction_factor)
    return np.repeat(np.repeat(matrix, resolution_factor[0], axis=0), resolution_factor[1], axis=1)


def reduced_basis_generator_pca(solutions_offline: List[np.ndarray], number_of_reduced_base_elements: int) -> List[
    np.ndarray]:
    transformer = StandardScaler(with_mean=True, with_std=True)  # to center by mean and scale by variance.
    # the mean and std is taken along the columns which means on each of the dimensions of the V space.
    basis = PCA(n_components=number_of_reduced_base_elements).fit(
        transformer.fit_transform(solutions_offline)).components_
    # we need to come back to the original space
    return transformer.inverse_transform(basis)


def state_estimation_fitting_method_least_squares(measurement_points, measurements, reduced_basis: List, **kwargs):
    # evaluate the reduced basis elements in the measurement points to get the matrix A: the rectangular matrix mxn
    # whose columns corresponds to the m evaluations of each element of the reduced basis in the measurement points.
    measurements_reduced_basis = sm.evaluate_solutions(measurement_points, reduced_basis)
    # solves the linear least squares problem Ax = b where b is the measurements which is the same as finding the
    # coordinates (coeficients) of the element in Vn space which is the proyection P_Vn u in l2.
    coefficients = np.linalg.lstsq(measurements_reduced_basis.T, measurements.T, rcond=-1)[0]
    # build the approximation
    approximate_solutions = coefficients.T @ np.array(reduced_basis)
    return approximate_solutions


# ============= ============= ============= ============= #
# FEM grid
# ============= ============= ============= ============= #
fig, ax = plt.subplots(1, 1, figsize=axes_xy_proportions)
x, y = np.meshgrid(sm.points_r, sm.points_c)
plt.scatter(x, y, c="white", alpha=1, marker="o", s=25)
# plt.scatter(x, y, c="black", alpha=1, marker="o", s=10)
segs1 = np.stack((x, y), axis=2)
segs2 = segs1.transpose(1, 0, 2)
plt.gca().add_collection(LineCollection(segs1, edgecolors="white", linewidths=1.5, alpha=0.7))
plt.gca().add_collection(LineCollection(segs2, edgecolors="white", linewidths=1.5, alpha=0.7))
save_fig_without_white(f"{presentation_path}/grid.png")

# ============= ============= ============= ============= #
# Limiting solutions
# ============= ============= ============= ============= #
sm_lm = SolutionsManagerFEM(blocks_geometry=(5, 5), N=10, num_cores=1)
T = np.logspace(0, 3, num_snapshots)
parameters_ls_dict = list()
infty_subdomains = [(0, 0), (1, 2), (3, 0), (4, 4)]
for i, t in tqdm(enumerate(T)):
    # define the matrix of diffusion coefficients \theta for each solution u to be computed.
    y = np.ones((1, 5, 5))
    for subdomain in infty_subdomains:
        y[(0,) + subdomain] = t
    y = np.array(y, dtype=int)
    parameters_ls_dict.append(y[0].tolist())
    # generate as many solutions as diffusion coefficients matrices given
    u = sm_lm.generate_solutions(a2try=y[:, ::-1])

    plot_solutions_together(
        sm_lm,
        diffusion_coefficients=np.array(y),
        solutions=u,
        num_points_per_dim_to_plot=num_points_per_dim_to_plot,
        contour_levels=contour_levels,
        axes_xy_proportions=axes_xy_proportions,
        titles=False,
        colorbar=False,
        measurement_points=None,
        cmap=cmap, add_grid=False)
    save_fig_without_white(f"{presentation_path}/solutions_lim_sol_{i}.png")

for subdomain in infty_subdomains:
    y = np.ones((1, 5, 5))
    y[(0,) + subdomain] = t
    y = np.array(y, dtype=int)
    parameters_ls_dict.append(y[0].tolist())
    # generate as many solutions as diffusion coefficients matrices given
    u = sm_lm.generate_solutions(a2try=y[:, ::-1])

    plot_solutions_together(
        sm_lm,
        diffusion_coefficients=np.array(y),
        solutions=u,
        num_points_per_dim_to_plot=num_points_per_dim_to_plot,
        contour_levels=contour_levels,
        axes_xy_proportions=axes_xy_proportions,
        titles=False,
        colorbar=False,
        measurement_points=None,
        cmap=cmap, add_grid=False)
    save_fig_without_white(f"{presentation_path}/solutions_lim_sol_{'_'.join(map(str, subdomain))}.png")

y = np.ones((1, 5, 5))
y = np.array(y, dtype=int)
parameters_ls_dict.append(y[0].tolist())
# generate as many solutions as diffusion coefficients matrices given
u = sm_lm.generate_solutions(a2try=y[:, ::-1])

plot_solutions_together(
    sm_lm,
    diffusion_coefficients=np.array(y),
    solutions=u,
    num_points_per_dim_to_plot=num_points_per_dim_to_plot,
    contour_levels=contour_levels,
    axes_xy_proportions=axes_xy_proportions,
    titles=False,
    colorbar=False,
    measurement_points=None,
    cmap=cmap, add_grid=False)
save_fig_without_white(f"{presentation_path}/solutions_lim_sol_11.png")

plot_solutions_together(
    sm_lm,
    diffusion_coefficients=np.array(y),
    solutions=u * 0,
    num_points_per_dim_to_plot=num_points_per_dim_to_plot,
    contour_levels=contour_levels,
    axes_xy_proportions=axes_xy_proportions,
    titles=False,
    colorbar=False,
    measurement_points=None,
    cmap=cmap, add_grid=False)
save_fig_without_white(f"{presentation_path}/solutions_lim_sol_infinf.png")

# ============= ============= ============= ============= #
# Not UEA assumption
# ============= ============= ============= ============= #
T = np.logspace(0, 3, num_snapshots)
parameters_inf_dict = list()
for i, t in enumerate(T):
    # define the matrix of diffusion coefficients \theta for each solution u to be computed.
    y = np.array([[[1, 1], [1, t]]], dtype=int)
    parameters_inf_dict.append(y[0].tolist())
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
    save_fig_without_white(f"{presentation_path}/solutions_inf_{i}.png")

# ============= ============= ============= ============= #
# Multiple samples
# ============= ============= ============= ============= #
T = np.linspace(0, 2 * np.pi, num_snapshots)
parameters_dict = list()
measurements = list()
for i, t in enumerate(T):
    # define the matrix of diffusion coefficients \theta for each solution u to be computed.
    y = np.array([space_y(t, 4)], dtype=int)
    parameters_dict.append(y[0].tolist())
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
    save_fig_without_white(f"{presentation_path}/solutions_{i}.png")

# ============= ============= ============= ============= #
# Optimization
# ============= ============= ============= ============= #
T = np.linspace(2 * np.pi / 4, np.pi * 3 / 4, num_snapshots_optim)
parameters_optim_dict = list()
u = list()
for i, t in enumerate(T):
    # define the matrix of diffusion coefficients \theta for each solution u to be computed.
    y = np.array([space_y(t, 4)])
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
points = np.array(sorted(points, key=lambda p: p[0]))
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
        measurement_points=points[:i + 1, :], measurements_color=measurements_color,
        cmap=cmap, add_grid=True)
    save_fig_without_white(f"{presentation_path}/solutions_point_measurements_increase_{i}.png")

    barplot_measurements(
        filename=f"{presentation_path}/barplot_measurements_increase_{i}.png",
        measurements=measurements[:i + 1] + [0] * (number_of_measures - i - 1),
        max_measurements=np.max(measurements))

# ============= ============= ============= ============= #
# Inverse problem averages
# ============= ============= ============= ============= #
num_cells_per_dim = 7
sub_sampling = 5  # reconstruction factor
n = sub_sampling * num_cells_per_dim
points_avg = measurements_sampling_method_grid(n ** 2, xlim=sm.x_domain, ylim=sm.y_domain)

T = np.linspace(0, 2 * np.pi, num_snapshots_optim)
parameters_avg_dict = list()
for i, t in enumerate(T):
    # define the matrix of diffusion coefficients \theta for each solution u to be computed.
    y = np.array([space_y(t, r=4)])
    parameters_avg_dict.append(y[0].tolist())
    # generate as many solutions as diffusion coefficients matrices given
    u = sm.generate_solutions(a2try=y[:, ::-1])

    measurements_avg = sm.evaluate_solutions(
        np.concatenate((points_avg[:, 0].reshape((-1, 1)), points_avg[:, 1].reshape((-1, 1))), axis=1),
        solutions=[u]).ravel().tolist()
    measurements_avg = calculate_averages_from_image(np.reshape(measurements_avg, (n, n)), num_cells_per_dim)
    uavg = make_image_high_resolution(measurements_avg, num_cells_per_dim)

    fig, ax = plt.subplots(1, 1, figsize=axes_xy_proportions)
    ax.imshow(uavg, origin='lower', cmap=cmap, extent=(-1, 1, -1, 1))
    ax.xaxis.set_major_locator(ticker.NullLocator())
    ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.vlines(np.linspace(*sm.x_domain, num=sm.blocks_geometry[1] + 1)[1:-1], ymin=sm.y_domain[0],
              ymax=sm.y_domain[1],
              linestyle="dashed", alpha=0.7, color="black")
    ax.hlines(np.linspace(*sm.y_domain, num=sm.blocks_geometry[0] + 1)[1:-1], xmin=sm.x_domain[0],
              xmax=sm.x_domain[1],
              linestyle="dashed", alpha=0.7, color="black")
    save_fig_without_white(f"{presentation_path}/solution_avg_{i}.png")

    if i == 25:  # TODO: harcoded i=25
        path2subcell = "/home/callum/Repos/SlidesMaker/DefenseSlides/Material/SubCell"
        np.save(f"{path2subcell}/solution_avg_25", measurements_avg)
        fig, ax = plt.subplots(1, 1, figsize=axes_xy_proportions)
        ax.imshow(uavg/np.max(u), origin='lower', cmap="viridis", extent=(-1, 1, -1, 1), vmin=-1, vmax=1)
        ax.xaxis.set_major_locator(ticker.NullLocator())
        ax.yaxis.set_major_locator(ticker.NullLocator())
        ax.vlines(np.linspace(*sm.x_domain, num=sm.blocks_geometry[1] + 1)[1:-1], ymin=sm.y_domain[0],
                  ymax=sm.y_domain[1],
                  linestyle="dashed", alpha=0.7, color="black")
        ax.hlines(np.linspace(*sm.y_domain, num=sm.blocks_geometry[0] + 1)[1:-1], xmin=sm.x_domain[0],
                  xmax=sm.x_domain[1],
                  linestyle="dashed", alpha=0.7, color="black")
        save_fig_without_white(f"{path2subcell}/solution_avg.png")

        plot_solutions_together(
            sm,
            diffusion_coefficients=np.array(y),
            solutions=u/np.max(u),
            num_points_per_dim_to_plot=num_points_per_dim_to_plot,
            contour_levels=contour_levels,
            axes_xy_proportions=axes_xy_proportions,
            titles=False,
            colorbar=False,
            vmin=-1, vmax=1,
            measurement_points=None, measurements_color=measurements_color,
            cmap=matplotlib.colormaps.get_cmap("viridis"), add_grid=True)
        save_fig_without_white(f"{path2subcell}/solution.png")
        break


    measurements_avg = measurements_avg.ravel().tolist() + [0]
    norm = matplotlib.colors.Normalize(vmin=0, vmax=np.max(measurements_avg))
    plt.figure()
    sns.barplot(data=pd.DataFrame(np.transpose([measurements_avg,
                                                list(range(1, num_cells_per_dim ** 2 + 2))]),
                                  columns=["measurements", "sensors", ]),
                x="sensors", y="measurements", palette={j + 1: cmapf(norm(m)) for j, m in enumerate(measurements_avg)})
    plt.ylim([0, 0.11])
    plt.xlabel("")
    plt.ylabel("")
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
    plt.box(False)
    plt.savefig(f"{presentation_path}/barplot_measurements_avg_{i}.png", bbox_inches='tight', pad_inches=0,
                transparent=True)
    plt.close()

# ============= ============= ============= ============= #
# State estimation
# ============= ============= ============= ============= #
solutions_offline = sm.generate_solutions(a2try=np.array(
    [space_y(t, r) for t, r in zip(np.random.uniform(0, np.pi * 2, size=100), np.random.uniform(0, 4.5, size=100))]))
rb = reduced_basis_generator_pca(solutions_offline, number_of_reduced_base_elements=2)
T = np.linspace(0, 2 * np.pi, num_snapshots_optim)
parameters_se_dict = list()
for i, t in enumerate(T):
    # define the matrix of diffusion coefficients \theta for each solution u to be computed.
    y = np.array([space_y(t, r=4)])
    parameters_se_dict.append(y[0].tolist())
    # generate as many solutions as diffusion coefficients matrices given
    u = sm.generate_solutions(a2try=y[:, ::-1])
    plot_solutions_together(
        sm,
        diffusion_coefficients=np.array(y),
        solutions=u,
        num_points_per_dim_to_plot=num_points_per_dim_to_plot,
        contour_levels=0,
        axes_xy_proportions=axes_xy_proportions,
        titles=False,
        colorbar=False, measurements_color=measurements_color,
        measurement_points=points,
        cmap="coolwarm", add_grid=True)
    save_fig_without_white(f"{presentation_path}/state_estimation_solutions_points_{i}.png")

    plot_solutions_together(
        sm,
        diffusion_coefficients=np.array(y),
        solutions=u,
        num_points_per_dim_to_plot=num_points_per_dim_to_plot,
        contour_levels=0,
        axes_xy_proportions=axes_xy_proportions,
        titles=False,
        colorbar=False, measurements_color=measurements_color,
        measurement_points=None,
        cmap="coolwarm", add_grid=True)
    save_fig_without_white(f"{presentation_path}/state_estimation_solutions_{i}.png")

    measurements = sm.evaluate_solutions(
        np.concatenate((points[:, 0].reshape((-1, 1)), points[:, 1].reshape((-1, 1))), axis=1),
        solutions=[u]).ravel().tolist()

    barplot_measurements(
        filename=f"{presentation_path}/state_estimation_barplot_measurements_{i}.png",
        measurements=measurements,
        max_measurements=np.max(measurements))

    uhat = state_estimation_fitting_method_least_squares(points, np.reshape(measurements, (1, -1)), reduced_basis=rb)
    plot_solutions_together(
        sm,
        diffusion_coefficients=np.array(y),
        solutions=uhat,
        num_points_per_dim_to_plot=num_points_per_dim_to_plot,
        contour_levels=0,
        axes_xy_proportions=axes_xy_proportions,
        titles=False,
        colorbar=False,
        measurement_points=None,
        cmap="coolwarm", add_grid=True)
    save_fig_without_white(f"{presentation_path}/state_estimation_{i}.png")

with open(f"{presentation_path}/metadata.json", "w") as f:
    json.dump(
        {
            "infty_subdomains": infty_subdomains,
            "parameters_ls_dict": parameters_ls_dict,
            "parameters_inf_dict": parameters_inf_dict,
            "parameters_dict": parameters_dict,
            "parameters_optim_dict": parameters_optim_dict,
            "parameters_avg_dict": parameters_avg_dict,
            "parameters_se_dict": parameters_se_dict,
            "num_snapshots_optim": num_snapshots_optim,
            "num_snapshots": num_snapshots,
            "number_of_measures": number_of_measures,
            "Vdim": sm.vspace_dim,
            "num_cells_per_dim": num_cells_per_dim,
            "sub_sampling": sub_sampling
        },
        f
    )

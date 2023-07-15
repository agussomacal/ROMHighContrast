import itertools
from contextlib import contextmanager

import matplotlib.pylab as plt
import numpy as np
from matplotlib import ticker

AXES_PROPORTIONS = (3, 3)


@contextmanager
def save_fig(pathplot, axes_xy_proportions=(4, 4), dpi=None):
    fig, ax = plt.subplots(figsize=axes_xy_proportions)
    yield ax
    plt.savefig(f"{pathplot}{'.png' if pathplot[-4:] not in ['.png', '.jpg', '.svg'] else ''}", dpi=dpi)
    plt.close()


def squared_subplots(N_subplots, axes_xy_proportions=(4, 4)):
    if N_subplots > 0:
        nrows = int(np.sqrt(N_subplots))
        ncols = int(np.ceil(N_subplots / nrows))
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True,
                               figsize=(axes_xy_proportions[0] * ncols, axes_xy_proportions[1] * nrows))
        if N_subplots == 1:
            ax = np.array(ax).reshape((1, 1))
        if len(ax.shape) == 1:
            ax = ax.reshape((1, -1))
        for i, j in itertools.product(np.arange(nrows), np.arange(ncols)):
            yield ax[i, j]


def plot_solution(ax, x, y, u_reshaped, sm, contour_levels=0, vmin=None, vmax=None, colorbar=True):
    if contour_levels:
        h = ax.contourf(x, y, u_reshaped, levels=contour_levels, origin='lower')
    else:
        h = ax.imshow(u_reshaped, vmin=vmin, vmax=vmax, origin='lower')
    if colorbar:
        plt.colorbar(h)
    ax.vlines(np.linspace(*sm.x_domain, num=sm.blocks_geometry[1] + 1)[1:-1], ymin=sm.y_domain[0], ymax=sm.y_domain[1],
              linestyle="dashed", alpha=0.7, color="black")
    ax.hlines(np.linspace(*sm.y_domain, num=sm.blocks_geometry[0] + 1)[1:-1], xmin=sm.x_domain[0], xmax=sm.x_domain[1],
              linestyle="dashed", alpha=0.7, color="black")


def plot_solutions_together(sm, diffusion_coefficients, solutions, num_points_per_dim_to_plot=100, contour_levels=0,
                            axes_xy_proportions=AXES_PROPORTIONS, titles=None, colorbar=False, measurement_points=None):
    x, y = np.meshgrid(np.linspace(*sm.x_domain, num=num_points_per_dim_to_plot),
                       np.linspace(*sm.y_domain, num=num_points_per_dim_to_plot))
    for i, (ax, u) in enumerate(
            zip(squared_subplots(len(solutions), axes_xy_proportions=axes_xy_proportions), solutions)):
        u = sm.evaluate_solutions(np.concatenate((x.reshape((-1, 1)), y.reshape((-1, 1))), axis=1), solutions=[u])
        if diffusion_coefficients is not None:
            ax.set_title(f"a={np.round(np.reshape(diffusion_coefficients[i], sm.blocks_geometry), decimals=2)}")
        elif titles is not None:
            ax.set_title(titles[i])
        plot_solution(ax, x, y, u.reshape((num_points_per_dim_to_plot, num_points_per_dim_to_plot)), sm, contour_levels,
                      colorbar=colorbar)
        ax.xaxis.set_major_locator(ticker.NullLocator())
        ax.yaxis.set_major_locator(ticker.NullLocator())

        if measurement_points is not None:
            ax.scatter(*measurement_points.T, marker="x", alpha=0.8, s=5, color="white")
    plt.tight_layout()


def plot_approximate_solutions_together(sm, diffusion_coefficients, solutions, approximate_solutions,
                                        num_points_per_dim_to_plot=100, contour_levels=0, measurement_points=None,
                                        colorbar=False, axes_xy_proportions=AXES_PROPORTIONS):
    x, y = np.meshgrid(np.linspace(*sm.x_domain, num=num_points_per_dim_to_plot),
                       np.linspace(*sm.y_domain, num=num_points_per_dim_to_plot))
    for i, (a, u_aprox, u_true) in enumerate(zip(diffusion_coefficients, approximate_solutions, solutions)):
        ua = sm.evaluate_solutions(np.concatenate((x.reshape((-1, 1)), y.reshape((-1, 1))), axis=1),
                                   solutions=[u_aprox])
        ut = sm.evaluate_solutions(np.concatenate((x.reshape((-1, 1)), y.reshape((-1, 1))), axis=1), solutions=[u_true])

        fig, ax = plt.subplots(ncols=2, figsize=(axes_xy_proportions[0] * 2, axes_xy_proportions[1]))
        fig.suptitle(f"State estimation of \n a={np.round(np.reshape(a, sm.blocks_geometry)[::-1], decimals=2)}")

        vmin = np.min((np.min(ua), np.min(ut)))
        vmax = np.max((np.max(ua), np.max(ut)))
        plot_solution(ax[0], x, y, ua.reshape((num_points_per_dim_to_plot, num_points_per_dim_to_plot)), sm,
                      contour_levels,
                      vmin=vmin, vmax=vmax, colorbar=colorbar)
        plot_solution(ax[1], x, y, ut.reshape((num_points_per_dim_to_plot, num_points_per_dim_to_plot)), sm,
                      contour_levels,
                      vmin=vmin, vmax=vmax, colorbar=colorbar)

        ax[0].set_title("\n Approximation")
        ax[1].set_title("\n Solution")

        if measurement_points is not None:
            ax[1].scatter(*measurement_points.T, marker="x", alpha=0.8, s=5, color="white")
    plt.tight_layout()

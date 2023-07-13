import itertools

import ipywidgets as widgets
import matplotlib.pylab as plt
import numpy as np
from ipywidgets import GridspecLayout

from src.lib.VizUtils import plot_solutions_together


def visualize_intuition(sm, diffusion_contrast_lower, diffusion_contrast_upper,
                        num_points_per_dim_to_plot=50, axes_xy_proportions=(3, 3)):
    grid = GridspecLayout(*sm.blocks_geometry)
    cells = list(itertools.product(*list(map(range, sm.blocks_geometry))))
    coefs_sliders = dict()
    for i, j in cells:
        key = f"a{i}{j}"
        coefs_sliders[key] = widgets.FloatSlider(value=50, min=diffusion_contrast_lower, max=diffusion_contrast_upper,
                                                 step=0.5,
                                                 description=f'a[{i},{j}]:', disabled=False,
                                                 continuous_update=False, orientation='horizontal', readout=True,
                                                 readout_format='.1f')
        grid[i, j] = coefs_sliders[key]

    def show_solution(**kwargs):
        diffusion_coefficients = np.array([list(kwargs.values())]).reshape((1,) + sm.blocks_geometry)
        solutions_intuition = sm.generate_solutions(diffusion_coefficients[:, ::-1, :])

        plot_solutions_together(
            sm,
            diffusion_coefficients=diffusion_coefficients[:, ::-1, :],
            solutions=solutions_intuition,
            num_points_per_dim_to_plot=num_points_per_dim_to_plot,
            contour_levels=7,
            axes_xy_proportions=axes_xy_proportions
        )

    out = widgets.interactive_output(show_solution, coefs_sliders)
    display(grid, out)


def vizualize_approximations(sm, measurements_sampling_method_dict, reduced_basis_dict, state_estimation_method_dict,
                             diffusion_contrast_lower, diffusion_contrast_upper, max_vn_dim,
                             num_points_per_dim_to_plot=50, axes_xy_proportions=(3, 3)):
    def show_approx(n_dim, rb_methods, m, measurements_sampling_method, state_estimation_method, **kwargs):

        approximate_solutions = []
        for rb_method in rb_methods:
            rb = reduced_basis_dict[rb_method][:n_dim]
            measurement_points = measurements_sampling_method_dict[measurements_sampling_method](m, sm.x_domain,
                                                                                                 sm.y_domain, basis=rb,
                                                                                                 sm=sm)
            diffusion_coefficients = np.array([list(kwargs.values())]).reshape((1,) + sm.blocks_geometry)
            solution = sm.generate_solutions(diffusion_coefficients)
            measurements_online = sm.evaluate_solutions(measurement_points, solutions=solution)
            approximate_solutions.append(
                state_estimation_method_dict[state_estimation_method](measurement_points, measurements_online, rb))

        plot_solutions_together(
            sm,
            None, [solution] + approximate_solutions,
            num_points_per_dim_to_plot=num_points_per_dim_to_plot,
            contour_levels=7,
            axes_xy_proportions=axes_xy_proportions,
            titles=["True solution"] + list(rb_methods), colorbar=False,
            measurement_points=measurement_points)
        plt.show()

    style = {'description_width': 'initial'}
    global_grid = GridspecLayout(4, 2)

    available_widgets = dict()
    grid = GridspecLayout(*sm.blocks_geometry)
    cells = list(itertools.product(*list(map(range, sm.blocks_geometry))))
    for i, j in cells:
        key = f"a{i}{j}"
        available_widgets[key] = widgets.FloatSlider(
            value=50,
            min=diffusion_contrast_lower,
            max=diffusion_contrast_upper,
            step=0.5,
            description=f'a[{i},{j}]:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.1f',
            style=style)
        grid[i, j] = available_widgets[key]
    global_grid[0, :] = grid

    global_grid[1, 0] = available_widgets["rb_methods"] = widgets.SelectMultiple(
        options=list(reduced_basis_dict.keys()),
        value=list(reduced_basis_dict.keys()),
        description="Reduced Basis: ",
        disabled=False,
        style=style)
    global_grid[1, 1] = available_widgets["n_dim"] = widgets.IntSlider(
        value=1, min=1,
        max=50,
        step=1,
        description='RB dim n:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        style=style)

    global_grid[2, 0] = available_widgets["measurements_sampling_method"] = widgets.Dropdown(
        options=list(measurements_sampling_method_dict.keys()),
        description="Measurements sampling method: ",
        disabled=False,
        style=style)
    global_grid[2, 1] = available_widgets["m"] = widgets.IntSlider(
        value=50, min=max_vn_dim,
        max=10 * max_vn_dim,
        step=1,
        description='Number of measurements:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        style=style)
    global_grid[3, :] = available_widgets["state_estimation_method"] = widgets.Dropdown(
        options=list(state_estimation_method_dict.keys()),
        description="State estimation method: ",
        disabled=False,
        style=style)

    out = widgets.interactive_output(show_approx, available_widgets)
    display(global_grid, out)


error_metrics_dict = {
    "L2": lambda x: np.mean(np.sqrt(np.mean(x ** 2, axis=-1))),
    "Linf": lambda x: np.max(np.sqrt(np.mean(x ** 2, axis=-1)))
}


def visualize_convergence(sm, solutions, measurements_sampling_method_dict, reduced_basis_dict,
                          state_estimation_method_dict, max_vn_dim):
    # n_per_dim = int(refinement*np.sqrt(sm.vspace_dim))
    # x, y = np.meshgrid(*[np.linspace(*sm.y_domain, num=n_per_dim), np.linspace(*sm.x_domain, num=n_per_dim)])
    # quadrature_points = np.concatenate([x.reshape((-1, 1)), y.reshape((-1, 1))], axis=1)

    def show_convergence(rb_methods, measurements_sampling_method, m, state_estimation_method, error_metric, noise):
        for rb_method in rb_methods:
            errors = []
            for n in range(1, max_vn_dim):
                basis = reduced_basis_dict[rb_method][:n]
                if measurements_sampling_method == "Optim" or len(errors) == 0:
                    measurement_points = measurements_sampling_method_dict[measurements_sampling_method](m, sm.x_domain,
                                                                                                         sm.y_domain,
                                                                                                         basis=basis,
                                                                                                         sm=sm)
                    measurements = sm.evaluate_solutions(measurement_points, solutions) + np.random.normal(scale=noise)
                v = solutions - \
                    state_estimation_method_dict[state_estimation_method](
                        measurement_points, measurements, np.reshape(basis, (n, -1)))
                errors.append(error_metrics_dict[error_metric](v))
                # errors.append(error_metrics_dict[error_metric](sm.evaluate_solutions(points=quadrature_points, solutions=v)))

            plt.plot(np.arange(1, max_vn_dim, dtype=int), errors, ".-", label=rb_method)
        plt.xticks(np.arange(1, max_vn_dim, dtype=int))
        plt.yscale("log")
        plt.grid()
        plt.legend()
        plt.show()

    style = {'description_width': 'initial'}
    global_grid = GridspecLayout(4, 2)
    available_widgets = dict()

    global_grid[0, 0] = available_widgets["error_metric"] = widgets.Dropdown(
        options=list(error_metrics_dict.keys()),
        description="Error metric: ",
        disabled=False,
        style=style)
    global_grid[0, 1] = available_widgets["noise"] = widgets.FloatSlider(
        value=0,
        min=0,
        max=1,
        step=0.01,
        description="Noise: ",
        disabled=False,
        style=style)

    global_grid[1, :] = available_widgets["rb_methods"] = widgets.SelectMultiple(
        options=list(reduced_basis_dict.keys()),
        value=list(reduced_basis_dict.keys()),
        description="Reduced Basis: ",
        disabled=False,
        style=style)

    global_grid[2, 0] = available_widgets["measurements_sampling_method"] = widgets.Dropdown(
        options=list(measurements_sampling_method_dict.keys()),
        description="Measurements sampling method: ",
        disabled=False,
        style=style)
    global_grid[2, 1] = available_widgets["m"] = widgets.IntSlider(
        value=50, min=max_vn_dim,
        max=10 * max_vn_dim,
        step=1,
        description='Number of measurements:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        style=style)
    global_grid[3, :] = available_widgets["state_estimation_method"] = widgets.Dropdown(
        options=list(state_estimation_method_dict.keys()),
        description="State estimation method: ",
        disabled=False,
        style=style)

    out = widgets.interactive_output(show_convergence, available_widgets)
    display(global_grid, out)


def visualize_state_estimation_methods(sm, solutions, measurements_sampling_method_dict, reduced_basis_dict,
                                       state_estimation_method_dict, max_vn_dim):
    def show_semethod(rb_method, measurements_sampling_method, m, state_estimation_methods, error_metric, noise,
                      vn_range):
        measurement_points = measurements_sampling_method_dict[measurements_sampling_method](m, sm.x_domain,
                                                                                             sm.y_domain)
        measurements = sm.evaluate_solutions(measurement_points, solutions) + np.random.normal(scale=noise)
        for state_estimation_method in state_estimation_methods:
            errors = []
            for n in range(*vn_range):
                v = solutions - \
                    state_estimation_method_dict[state_estimation_method](
                        measurement_points, measurements, np.reshape(reduced_basis_dict[rb_method][:n], (n, -1)))
                errors.append(error_metrics_dict[error_metric](v))
            # errors = [error_metrics_dict[error_metric](
            #     solutions - state_estimation_method_dict[state_estimation_method](measurement_points, measurements,
            #                                                                       np.reshape(
            #                                                                           reduced_basis_dict[rb_method][:n],
            #                                                                           (n, -1)))
            # ) for n in range(*vn_range)]
            plt.plot(np.arange(*vn_range), errors, ".-", label=state_estimation_method)
        plt.xticks(np.arange(*vn_range, dtype=int))
        plt.grid()
        plt.yscale("log")
        plt.ylim((None, 1e-1))
        plt.legend()
        plt.show()

    style = {'description_width': 'initial'}
    global_grid = GridspecLayout(4, 2)
    available_widgets = dict()

    global_grid[0, 0] = available_widgets["error_metric"] = widgets.Dropdown(
        options=list(error_metrics_dict.keys()),
        description="Error metric: ",
        disabled=False,
        style=style)
    global_grid[0, 1] = available_widgets["noise"] = widgets.FloatText(
        value=0,
        min=0,
        max=1,
        # step=0.01,
        description="Noise: ",
        disabled=False,
        style=style)

    global_grid[1, :] = available_widgets["rb_method"] = widgets.Dropdown(
        options=list(reduced_basis_dict.keys()),
        value=list(reduced_basis_dict.keys())[0],
        description="Reduced Basis: ",
        disabled=False,
        style=style)

    global_grid[2, 0] = available_widgets["measurements_sampling_method"] = widgets.Dropdown(
        options=list(measurements_sampling_method_dict.keys()),
        # value=list(measurements_sampling_method_dict.keys()),
        description="Measurements sampling method: ",
        disabled=False,
        style=style)

    global_grid[2, 1] = available_widgets["m"] = widgets.IntText(
        value=50, min=max_vn_dim,
        # max=10 * max_vn_dim,
        # step=1,
        description='Number of measurements:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        style=style)
    global_grid[3, 0] = available_widgets["state_estimation_methods"] = widgets.SelectMultiple(
        options=list(state_estimation_method_dict.keys()),
        value=list(state_estimation_method_dict.keys()),
        description="State estimation method: ",
        disabled=False,
        style=style)

    global_grid[3, 1] = available_widgets["vn_range"] = widgets.IntRangeSlider(
        min=0,
        max=max_vn_dim,
        value=(1, max_vn_dim),
        step=1,
        description="dim(Vn) range: ",
        disabled=False,
        style=style)

    out = widgets.interactive_output(show_semethod, available_widgets)
    display(global_grid, out)


def visualize_all(sm, solutions, measurements_sampling_method_dict, reduced_basis_dict,
                                       state_estimation_method_dict, max_vn_dim):
    def show(rb_method, measurements_sampling_method, m, state_estimation_methods, error_metric, noise,
                      vn_range):
        measurement_points = measurements_sampling_method_dict[measurements_sampling_method](m, sm.x_domain,
                                                                                             sm.y_domain)
        measurements = sm.evaluate_solutions(measurement_points, solutions) + np.random.normal(scale=noise)
        for state_estimation_method in state_estimation_methods:
            errors = [error_metrics_dict[error_metric](
                solutions - state_estimation_method_dict[state_estimation_method](measurement_points, measurements,
                                                                                  np.reshape(
                                                                                      reduced_basis_dict[rb_method][:n],
                                                                                      (n, -1)))
            ) for n in range(*vn_range)]
            plt.plot(np.arange(*vn_range), errors, ".-", label=state_estimation_method)
        plt.xticks(np.arange(*vn_range, dtype=int))
        plt.grid()
        plt.yscale("log")
        plt.ylim((None, 1e-1))
        plt.legend()
        plt.show()

    style = {'description_width': 'initial'}
    global_grid = GridspecLayout(4, 2)
    available_widgets = dict()

    global_grid[0, 0] = available_widgets["error_metric"] = widgets.Dropdown(
        options=list(error_metrics_dict.keys()),
        description="Error metric: ",
        disabled=False,
        style=style)
    global_grid[0, 1] = available_widgets["noise"] = widgets.FloatText(
        value=0,
        min=0,
        max=1,
        # step=0.01,
        description="Noise: ",
        disabled=False,
        style=style)

    global_grid[1, :] = available_widgets["rb_method"] = widgets.Dropdown(
        options=list(reduced_basis_dict.keys()),
        value=list(reduced_basis_dict.keys())[0],
        description="Reduced Basis: ",
        disabled=False,
        style=style)

    global_grid[2, 0] = available_widgets["measurements_sampling_method"] = widgets.Dropdown(
        options=list(measurements_sampling_method_dict.keys()),
        # value=list(measurements_sampling_method_dict.keys()),
        description="Measurements sampling method: ",
        disabled=False,
        style=style)

    global_grid[2, 1] = available_widgets["m"] = widgets.IntText(
        value=50, min=max_vn_dim,
        # max=10 * max_vn_dim,
        # step=1,
        description='Number of measurements:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        style=style)
    global_grid[3, 0] = available_widgets["state_estimation_methods"] = widgets.SelectMultiple(
        options=list(state_estimation_method_dict.keys()),
        value=list(state_estimation_method_dict.keys()),
        description="State estimation method: ",
        disabled=False,
        style=style)

    global_grid[3, 1] = available_widgets["vn_range"] = widgets.IntRangeSlider(
        min=0,
        max=max_vn_dim,
        value=(1, max_vn_dim),
        step=1,
        description="dim(Vn) range: ",
        disabled=False,
        style=style)

    out = widgets.interactive_output(show_semethod, available_widgets)
    display(global_grid, out)
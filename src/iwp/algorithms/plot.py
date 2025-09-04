import math
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


def custom_layout(n_plots, cols, fig_kwargs={}):
    rows = math.ceil(n_plots / cols)

    axs = np.full((rows, cols), None)
    fig = plt.figure(**fig_kwargs)
    gs = GridSpec(rows, cols * 2)

    full_rows = n_plots // cols
    extra = cols - n_plots % cols

    for i in range(n_plots):
        r, c = divmod(i, cols)
        if r < full_rows:
            axs[r, c] = fig.add_subplot(gs[r, 2 * c : 2 * (c + 1)])
        else:
            axs[r, c] = fig.add_subplot(gs[r, 2 * c + extra : 2 * (c + 1) + extra])
        # Remove this to keep y-axis tick labels on all subplots
        # if c:
        #     plt.setp(axs[r, c].get_yticklabels(), visible=False)
    return fig, axs


def plot_all_algorithms_convergence(
    algorithms, visuals_path, add_marker=False, save=True
):
    fig, axs = custom_layout(5, 3, fig_kwargs={"figsize": (25, 10)})
    algo_plot_names = []

    # Find the maximum number of iterations among all algorithms
    n_iterations = max(algo.max_iterations for algo in algorithms)

    for algo in algorithms:
        label_name = algo.algo_plot_name
        algo_plot_names.append(label_name)
        for ax, values, label in zip(
            axs[0],
            [algo.mse_values, algo.mae_values, algo.f_values],
            ["MSE", "MAE", "Objective function"],
        ):
            iters = len(values)
            ax.plot(
                range(iters),
                values,
                label=label_name,
                marker="o" if add_marker else None,
                markersize=4 if add_marker else None,
            )
            ax.set_xlabel("Iteration")
            ax.set_ylabel(label)
            if label == "Objective function":
                ax.set_yscale("log")
            ax.legend()

            # Draw a cross marker if the algorithm stopped before reaching max_iterations
            if iters < algo.max_iterations:
                ax.plot(
                    iters - 1,
                    values[-1],
                    marker="x",
                    color="black",
                    markersize=10,
                    label=f"{label_name} stopped",
                )
            # Draw a dotted line for constant continuation to max_iterations
            if iters < n_iterations:
                ax.plot(
                    range(iters - 1, n_iterations),
                    [values[-1]] * (n_iterations - iters + 1),
                    linestyle=":",
                    color="black",
                )

    cv_times = [algo.cv_time for algo in algorithms]
    memory_used_kb = [algo.memory_used / 1024 for algo in algorithms]
    axs[1, 0].barh(algo_plot_names, cv_times, color="skyblue")
    axs[1, 0].set_xlabel("Convergence Time (s)")
    axs[1, 0].set_title("Convergence Time")
    axs[1, 1].barh(algo_plot_names, memory_used_kb, color="lightgreen")
    axs[1, 1].set_xlabel("Peak Memory Used (KB)")
    axs[1, 1].set_title("Peak Memory Usage")

    fig.suptitle("Convergence Plots for Various Algorithms")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save:
        plt.savefig(os.path.join(visuals_path, "Global.png"))
        plt.close()

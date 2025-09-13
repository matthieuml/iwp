import abc
import os
import time
import tracemalloc

import matplotlib.pyplot as plt
import numpy as np

from .metrics import mae, mse


class FixedPointAlgorithm(abc.ABC):
    def __init__(self, exp_name, algo_plot_name, f, logger=None, verbose=True):
        self.exp_name = exp_name
        self.algo_plot_name = algo_plot_name
        self.f = f
        self.x_values = []
        self.f_values = []
        self.iteration = None
        self.max_iterations = None
        self.cv_time = None
        self.memory_used = None
        self.logger = logger
        self.verbose = verbose

    @abc.abstractmethod
    def step(self, x):
        pass

    @abc.abstractmethod
    def is_converged(self, x):
        pass

    def run(self, x0, max_iterations=1000):
        self.max_iterations = max_iterations
        self.logger.info(
            f"Started {self.algo_plot_name} for a maximum of {self.max_iterations} iterations."
        )
        # Preallocate arrays to not count them in memory usage
        self.x_values = np.empty((max_iterations + 1,) + x0.shape, dtype=x0.dtype)
        self.f_values = np.empty(max_iterations + 1, dtype=float)
        # Start measuring time and memory
        tracemalloc.start()
        t0 = time.time()
        x = x0
        self.x_values[0] = x0
        self.f_values[0] = self.f(x0)
        self.iteration = 0
        while not self.is_converged(x) and self.iteration < self.max_iterations:
            x = self.step(x)
            self.iteration += 1
            self.x_values[self.iteration] = x
            self.f_values[self.iteration] = self.f(x)
            if self.logger:
                msg = f"Iteration {self.iteration}: f(x) = {self.f_values[self.iteration]:.6f}, time = {time.time() - t0:.3f}s"
                (self.logger.info if self.verbose else self.logger.debug)(msg)
        self.cv_time = time.time() - t0
        _, peak = tracemalloc.get_traced_memory()
        self.memory_used = peak
        tracemalloc.stop()
        if self.logger:
            msg = (
                f"Converged after {self.iteration} iterations in {self.cv_time:.3f} seconds with {self.memory_used / 1024:.2f} KB memory used."
                if self.iteration < self.max_iterations
                else f"Stopped after {self.max_iterations} iterations in {self.cv_time:.3f} seconds with {self.memory_used / 1024:.2f} KB memory used."
            )
            self.logger.info(msg)
        # Cut arrays to actual size
        self.x_values = self.x_values[:self.iteration + 1]
        self.f_values = self.f_values[:self.iteration + 1]
        return x

    def plot_algorithm_convergence(self, m, visuals_path, add_marker=False, show=False, save=True):
        m_pred = (
            self.x_values[:, -m.shape[0] :]
            if self.x_values.ndim == 2
            else self.x_values
        )
        mse_values = mse(m_pred, m)
        mae_values = mae(m_pred, m)
        self.mse_values = mse_values
        self.mae_values = mae_values

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        for ax, values, label, ylabel in zip(
            axs,
            [mse_values, mae_values, self.f_values],
            ["MSE", "MAE", "Objective function"],
            ["MSE", "MAE", "Objective function"],
        ):
            if add_marker:
                ax.plot(values, label=label, marker="o", markersize=4)
            else:
                ax.plot(values, label=label)
            if self.iteration < self.max_iterations:
                ax.scatter(
                    self.iteration, values[-1], color="red", marker="x", label="Stopped"
                )
            ax.set_xlabel("Iteration")
            ax.set_ylabel(ylabel)
            if ylabel == "Objective function":
                # Log scale only for objective function
                ax.set_yscale("log")
            ax.legend()

        fig.suptitle(
            f"Convergence Plots for {self.algo_plot_name} in {self.cv_time:.3f}s using {self.memory_used / 1024:.2f} KB"
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if show:
            plt.show()
        if save:
            file_name = os.path.join(visuals_path, self.algo_plot_name + ".png")
            plt.savefig(file_name)
            if self.logger:
                self.logger.info(f"Saved convergence plots to {file_name}")
        plt.close()


class ClosedFormSolution(FixedPointAlgorithm):
    def __init__(self, exp_name, algo_plot_name, f, solution, logger=None, verbose=True):
        super().__init__(exp_name, algo_plot_name, f, logger=logger, verbose=verbose)
        self.solution = solution

    def step(self, x):
        return self.solution

    def is_converged(self, x, threshold=1e-6):
        return True


class GradientDescent(FixedPointAlgorithm):
    def __init__(
        self, exp_name, algo_plot_name, f, df, K, gamma, logger=None, verbose=True
    ):
        super().__init__(exp_name, algo_plot_name, f, logger=logger, verbose=verbose)
        self.df = df
        self.K = K
        assert gamma < 2.0 / self.K, "gamma must be less than 2/L for convergence"
        self.gamma = gamma
        self.current_gradient = None

    def step(self, x):
        return x - self.gamma * self.current_gradient

    def is_converged(self, x, threshold=1e-6):
        self.current_gradient = self.df(x)
        return np.linalg.norm(self.current_gradient) < threshold


class NesterovAcceleratedGradientDescent(FixedPointAlgorithm):
    def __init__(self, exp_name, algo_plot_name, f, df, K, logger=None, verbose=True):
        super().__init__(exp_name, algo_plot_name, f, logger=logger, verbose=verbose)
        self.df = df
        self.K = K
        self.beta_prev = 1.0
        self.y_prev = None
        self.current_gradient = None

    def step(self, x):
        if self.y_prev is None:
            self.y_prev = x
        y_new = x - (1.0 / self.K) * self.current_gradient
        self.beta = (1 + np.sqrt(1 + 4 * self.beta_prev**2)) / 2
        self.gamma = (self.beta_prev - 1) / self.beta
        self.beta_prev = self.beta
        x_new = y_new + self.gamma * (y_new - self.y_prev)
        self.y_prev = y_new.copy()
        return x_new

    def is_converged(self, x, threshold=1e-6):
        self.current_gradient = self.df(x)
        return np.linalg.norm(self.current_gradient) < threshold


class StronglyConvexNesterovAcceleratedGradientDescent(FixedPointAlgorithm):
    def __init__(
        self, exp_name, algo_plot_name, f, df, K, mu, logger=None, verbose=True
    ):
        super().__init__(exp_name, algo_plot_name, f, logger=logger, verbose=verbose)
        self.df = df
        self.K = K
        self.mu = mu
        ratio = np.sqrt(K / mu)
        self.gamma = -(ratio - 1) / (ratio + 1)
        self.y_prev = None
        self.current_gradient = None

    def step(self, x):
        if self.y_prev is None:
            self.y_prev = x
        y_new = x - (1.0 / self.K) * self.current_gradient
        x_new = y_new + self.gamma * (y_new - self.y_prev)
        self.y_prev = y_new.copy()
        return x_new

    def is_converged(self, x, threshold=1e-6):
        self.current_gradient = self.df(x)
        return np.linalg.norm(self.current_gradient) < threshold


class ForwardBackward(FixedPointAlgorithm):
    def __init__(
        self,
        exp_name,
        algo_plot_name,
        f,
        grad,
        prox,
        gamma,
        lambd,
        logger=None,
        verbose=True,
    ):
        super().__init__(exp_name, algo_plot_name, f, logger=logger, verbose=verbose)
        self.grad = grad
        self.prox = prox
        self.gamma = gamma
        self.lambd = lambd
        self.current_gradient = None

    def step(self, x):
        gamma_n = self.gamma(self.iteration) if callable(self.gamma) else self.gamma
        lambda_n = self.lambd(self.iteration) if callable(self.lambd) else self.lambd
        z = x - gamma_n * self.current_gradient
        y = self.prox(z, gamma_n)
        return x + lambda_n * (y - x)

    def is_converged(self, x, threshold=1e-6):
        self.current_gradient = self.grad(x)
        return np.linalg.norm(self.current_gradient) < threshold


class FISTA(FixedPointAlgorithm):
    def __init__(
        self,
        exp_name,
        algo_plot_name,
        f,
        grad,
        prox,
        K,
        logger=None,
        verbose=True,
    ):
        super().__init__(exp_name, algo_plot_name, f, logger=logger, verbose=verbose)
        self.grad = grad
        self.prox = prox
        self.K = K
        self.beta_prev = 1.0
        self.y_prev = None
        self.current_gradient = None

    def step(self, x):
        if self.y_prev is None:
            self.y_prev = x
        z = x - (1.0 / self.K) * self.current_gradient
        y = self.prox(z, self.K)
        self.beta = (1 + np.sqrt(1 + 4 * self.beta_prev**2)) / 2
        self.gamma = (self.beta_prev - 1) / self.beta
        self.beta_prev = self.beta
        x_new = y + self.gamma * (y - self.y_prev)
        self.y_prev = y.copy()
        return x_new

    def is_converged(self, x, threshold=1e-6):
        self.current_gradient = self.grad(x)
        return np.linalg.norm(self.current_gradient) < threshold

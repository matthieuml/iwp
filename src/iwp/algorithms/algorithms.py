import abc
import os
import time
import tracemalloc

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

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
        tracemalloc.start()
        t0 = time.time()
        x = x0
        self.x_values = [x0]
        self.f_values = [self.f(x0)]
        self.iteration = 0
        while not self.is_converged(x) and self.iteration < self.max_iterations:
            x = self.step(x)
            self.x_values.append(x)
            self.f_values.append(self.f(x))
            self.iteration += 1
            if self.logger:
                msg = f"Iteration {self.iteration}: f(x) = {self.f_values[-1]:.6f}, time = {time.time() - t0:.3f}s"
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
        self.x_values = np.array(self.x_values)
        self.f_values = np.array(self.f_values)

    def plot_algorithm_convergence(self, m, visuals_path, add_marker=False, save=True):
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
            ax.set_yscale("log")
            ax.legend()

        fig.suptitle(
            f"Convergence Plots for {self.algo_plot_name} in {self.cv_time:.3f}s using {self.memory_used / 1024:.2f} KB"
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save:
            file_name = os.path.join(visuals_path, self.algo_plot_name + ".png")
            plt.savefig(file_name)
            if self.logger:
                self.logger.info(f"Saved convergence plots to {file_name}")
            plt.close()


class ConvexGradientDescent(FixedPointAlgorithm):
    def __init__(
        self, exp_name, algo_plot_name, f, df, K, gamma, logger=None, verbose=True
    ):
        super().__init__(exp_name, f, algo_plot_name, logger=logger, verbose=verbose)
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


class ConvexNesterovAcceleratedGradientDescent(FixedPointAlgorithm):
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


class ConstrainedConvexForwardBackward(FixedPointAlgorithm):
    def __init__(
        self,
        exp_name,
        algo_plot_name,
        f,
        D,
        D_star,
        E,
        E_star,
        d,
        mu,
        gamma,
        lambd,
        P,
        logger=None,
        verbose=True,
    ):
        super().__init__(exp_name, algo_plot_name, f, logger=logger, verbose=verbose)
        self.D = D
        self.D_star = D_star
        self.E = E
        self.E_star = E_star
        self.d = d
        self.mu = mu
        self.gamma = gamma
        self.lambd = lambd
        self.P = P
        self.current_gradient = None

    def step(self, x):
        gamma_n = self.gamma(self.iteration) if callable(self.gamma) else self.gamma
        lambda_n = self.lambd(self.iteration) if callable(self.lambd) else self.lambd
        z = x - gamma_n * self.current_gradient
        w = sp.linalg.spsolve(self.E @ self.E_star, self.E @ z)
        y = z - self.E_star @ w
        return x + lambda_n * (y - x)

    def is_converged(self, x, threshold=1e-6):
        reg = np.zeros_like(x)
        reg[-self.P :] = self.mu * x[-self.P :]
        self.current_gradient = self.D_star @ (self.D @ x - self.d) + reg
        return np.linalg.norm(self.current_gradient) < threshold


class FISTA(FixedPointAlgorithm):
    def __init__(
        self,
        exp_name,
        algo_plot_name,
        f,
        D,
        D_star,
        E,
        E_star,
        d,
        mu,
        K,
        P,
        logger=None,
        verbose=True,
    ):
        super().__init__(exp_name, algo_plot_name, f, logger=logger, verbose=verbose)
        self.D = D
        self.D_star = D_star
        self.E = E
        self.E_star = E_star
        self.d = d
        self.mu = mu
        self.K = K
        self.P = P
        self.beta_prev = 1.0
        self.y_prev = None
        self.current_gradient = None

    def step(self, x):
        if self.y_prev is None:
            self.y_prev = x
        z = x - (1.0 / self.K) * self.current_gradient
        w = sp.linalg.spsolve(self.E @ self.E_star, self.E @ z)
        y = z - self.E_star @ w
        self.beta = (1 + np.sqrt(1 + 4 * self.beta_prev**2)) / 2
        self.gamma = (self.beta_prev - 1) / self.beta
        self.beta_prev = self.beta
        x_new = y + self.gamma * (y - self.y_prev)
        self.y_prev = y.copy()
        return x_new

    def is_converged(self, x, threshold=1e-6):
        reg = np.zeros_like(x)
        reg[-self.P :] = self.mu * x[-self.P :]
        self.current_gradient = self.D_star @ (self.D @ x - self.d) + reg
        return np.linalg.norm(self.current_gradient) < threshold


class ConstrainedConvexGradientDescent(FixedPointAlgorithm):
    def __init__(
        self,
        exp_name,
        algo_plot_name,
        f,
        A,
        A_star,
        C,
        C_star,
        B_list,
        d_list,
        mu,
        gamma,
        logger=None,
        verbose=True,
    ):
        super().__init__(exp_name, algo_plot_name, f, logger=logger, verbose=verbose)
        self.A = A
        self.A_star = A_star
        self.C = C
        self.C_star = C_star
        self.B_list = B_list
        self.d_list = d_list
        self.mu = mu
        self.gamma = gamma
        self.current_gradient = None

    def step(self, m):
        gamma_n = self.gamma(self.iteration) if callable(self.gamma) else self.gamma
        return m - gamma_n * self.current_gradient

    def is_converged(self, m, threshold=1e-6):
        p_sum = sum(
            B_i.conj().T
            @ sp.linalg.spsolve(
                self.A_star,
                self.C_star @ (self.C @ sp.linalg.spsolve(self.A, B_i @ m) - d_i),
            )
            for B_i, d_i in zip(self.B_list, self.d_list)
        )
        grad = p_sum + self.mu * m
        self.current_gradient = grad
        return np.linalg.norm(grad) < threshold


def plot_all_algorithms_convergence(
    algorithms, visuals_path, add_marker=False, save=True
):
    fig, axs = plt.subplots(2, 3, figsize=(25, 10))
    algo_plot_names = []

    for algo in algorithms:
        label_name = algo.algo_plot_name
        algo_plot_names.append(label_name)
        for ax, values, label in zip(
            axs[0],
            [algo.mse_values, algo.mae_values, algo.f_values],
            ["MSE", "MAE", "Objective function"],
        ):
            if add_marker:
                ax.plot(values, label=label_name, marker="o", markersize=4)
            else:
                ax.plot(values, label=label_name)
            ax.set_xlabel("Iteration")
            ax.set_ylabel(label)
            ax.set_yscale("log")
            ax.legend()

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

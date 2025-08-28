import abc
import os
import time
import tracemalloc

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

from .metrics import mae, mse


class FixedPointAlgorithm(abc.ABC):
    def __init__(self, name, f, logger=None, verbose=True):
        self.name = name
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
        tracemalloc.start()
        t0 = time.time()
        x = x0
        self.x_values = [x0]
        self.f_values = [self.f(x0)]
        self.iteration = 0
        while not self.is_converged(x) and self.iteration < self.max_iterations:
            x_new = self.step(x)
            x = x_new
            self.x_values.append(x)
            self.f_values.append(self.f(x))
            self.iteration += 1
            if self.logger:
                if self.verbose:
                    self.logger.info(
                        f"Iteration {self.iteration}: f(x) = {self.f_values[-1]:.6f}, time = {time.time() - t0:.3f}s"
                    )
                else:
                    self.logger.debug(
                        f"Iteration {self.iteration}: f(x) = {self.f_values[-1]:.6f}, time = {time.time() - t0:.3f}s"
                    )
        self.cv_time = time.time() - t0
        _, peak = tracemalloc.get_traced_memory()
        self.memory_used = peak
        tracemalloc.stop()
        if self.logger:
            if self.iteration < self.max_iterations:
                self.logger.info(
                    f"Converged after {self.iteration} iterations in {self.cv_time:.3f} seconds with {self.memory_used / 1024:.2f} KB memory used."
                )
            else:
                self.logger.info(
                    f"Stopped after {self.max_iterations} iterations in {self.cv_time:.3f} seconds with {self.memory_used / 1024:.2f} KB memory used."
                )
        self.x_values = np.array(self.x_values)
        self.f_values = np.array(self.f_values)


class ConvexGradientDescent(FixedPointAlgorithm):
    def __init__(self, name, f, df, K, gamma, logger=None, verbose=True):
        super().__init__(name, f, logger=logger, verbose=verbose)
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
    def __init__(self, name, f, df, K, logger=None, verbose=True):
        super().__init__(name, f, logger=logger, verbose=verbose)
        self.df = df
        self.K = K
        self.lambd_prev = 1.0
        self.y_prev = None
        self.current_gradient = None

    def step(self, x):
        # If y_prev is None, initialize it to x
        if self.y_prev is None:
            self.y_prev = x
        # Compute the step size
        self.lambd = (1 + np.sqrt(1 + 4 * self.lambd_prev**2)) / 2
        self.gamma = (self.lambd_prev - 1) / self.lambd
        # Compute gradient step
        y_new = x - (1.0 / self.K) * self.current_gradient
        # Nesterov acceleration
        x_new = y_new + self.gamma * (y_new - self.y_prev)
        # Store for next iteration
        self.y_prev = y_new.copy()
        return x_new

    def is_converged(self, x, threshold=1e-6):
        self.current_gradient = self.df(x)
        return np.linalg.norm(self.current_gradient) < threshold


class StronglyConvexNesterovAcceleratedGradientDescent(FixedPointAlgorithm):
    def __init__(self, name, f, df, K, mu, logger=None, verbose=True):
        super().__init__(name, f, logger=logger, verbose=verbose)
        self.df = df
        self.K = K
        self.mu = mu
        ratio = np.sqrt(K / mu)
        self.gamma = -(ratio - 1) / (ratio + 1)
        self.y_prev = None
        self.current_gradient = None

    def step(self, x):
        # If y_prev is None, initialize it to x
        if self.y_prev is None:
            self.y_prev = x
        # Compute gradient step
        y_new = x - (1.0 / self.K) * self.current_gradient
        # Nesterov acceleration
        x_new = y_new + self.gamma * (y_new - self.y_prev)
        # Store for next iteration
        self.y_prev = y_new.copy()
        return x_new

    def is_converged(self, x, threshold=1e-6):
        self.current_gradient = self.df(x)
        return np.linalg.norm(self.current_gradient) < threshold

    def plot_algorithm_convergence(self, m, visuals_path, add_marker=False):
        m_pred = self.x_values[:, -m.shape[0] :]
        mse_values = mse(m_pred, m)
        mae_values = mae(m_pred, m)

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        # Plot MSE over iterations
        if add_marker:
            axs[0].plot(mse_values, label="MSE", marker="o", markersize=4)
        else:
            axs[0].plot(mse_values, label="MSE")
        if self.iteration < self.max_iterations:
            axs[0].scatter(
                self.iteration, mse_values[-1], color="red", marker="x", label="Stopped"
            )
        axs[0].set_xlabel("Iteration")
        axs[0].set_ylabel("MSE")
        axs[0].set_yscale("log")
        axs[0].legend()

        # Plot MAE over iterations
        if add_marker:
            axs[1].plot(mae_values, label="MAE", marker="o", markersize=4)
        else:
            axs[1].plot(mae_values, label="MAE")
        if self.iteration < self.max_iterations:
            axs[1].scatter(
                self.iteration, mae_values[-1], color="red", marker="x", label="Stopped"
            )
        axs[1].set_xlabel("Iteration")
        axs[1].set_ylabel("MAE")
        axs[1].set_yscale("log")
        axs[1].legend()

        # Plot objective function values
        if add_marker:
            axs[2].plot(
                self.f_values, label="Objective function", marker="o", markersize=4
            )
        else:
            axs[2].plot(self.f_values, label="Objective function")
        if self.iteration < self.max_iterations:
            axs[2].scatter(
                self.iteration,
                self.f_values[-1],
                color="red",
                marker="x",
                label="Stopped",
            )
        axs[2].set_xlabel("Iteration")
        axs[2].set_ylabel("Objective function")
        axs[2].set_yscale("log")
        axs[2].legend()

        fig.suptitle(
            f"Convergence Plots for Strongly Convex Nesterov Accelerated Gradient Descent in {self.cv_time:.3f}s using {self.memory_used / 1024:.2f} KB"
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(
            os.path.join(
                visuals_path, f"StronglyConvexNesterovAcceleratedGradientDescent.png"
            )
        )
        self.logger.info(
            f"Saved convergence plots to {visuals_path}/StronglyConvexNesterovAcceleratedGradientDescent.png"
        )
        plt.close()


class ConstrainedConvexForwardBackward(FixedPointAlgorithm):
    def __init__(
        self,
        name,
        f,
        D,
        D_star,
        E,
        E_star,
        d,
        mu,
        gamma,
        lambd,
        logger=None,
        verbose=True,
    ):
        super().__init__(name, f, logger=logger, verbose=verbose)
        self.D = D
        self.D_star = D_star
        self.E = E
        self.E_star = E_star
        self.d = d
        self.mu = mu
        self.gamma = gamma
        self.lambd = lambd
        self.x_prev = None
        self.current_gradient = None

    def step(self, x):
        # If y_prev is None, initialize it to x
        if self.x_prev is None:
            self.x_prev = x
        # Get current gamma and lambda
        if callable(self.gamma):
            gamma_n = self.gamma(self.iteration)
        elif isinstance(self.gamma, float):
            gamma_n = self.gamma
        elif isinstance(self.gamma, int):
            gamma_n = self.gamma
        else:
            raise ValueError("gamma must be a callable, float, or int.")
        if callable(self.lambd):
            lambda_n = self.lambd(self.iteration)
        elif isinstance(self.lambd, float):
            lambda_n = self.lambd
        elif isinstance(self.lambd, int):
            lambda_n = self.lambd
        else:
            raise ValueError("lambd must be a callable, float, or int.")
        # Gradient step
        z = x - gamma_n * self.current_gradient
        # Proximal step
        w = sp.linalg.spsolve(self.E @ self.E_star, self.E @ z)
        y = z - self.E_star @ w
        # Relaxation step
        x_new = self.x_prev + lambda_n * (y - self.x_prev)
        # Store for next iteration
        self.y_prev = x_new.copy()
        return x_new

    def is_converged(self, x, threshold=1e-6):
        self.current_gradient = self.D_star @ (self.D @ x - self.d) + self.mu * x
        return np.linalg.norm(self.current_gradient) < threshold

    def plot_algorithm_convergence(self, m, visuals_path, add_marker=False):
        m_pred = self.x_values[:, -m.shape[0] :]
        mse_values = mse(m_pred, m)
        mae_values = mae(m_pred, m)

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        # Plot MSE over iterations
        if add_marker:
            axs[0].plot(mse_values, label="MSE", marker="o", markersize=4)
        else:
            axs[0].plot(mse_values, label="MSE")
        if self.iteration < self.max_iterations:
            axs[0].scatter(
                self.iteration, mse_values[-1], color="red", marker="x", label="Stopped"
            )
        axs[0].set_xlabel("Iteration")
        axs[0].set_ylabel("MSE")
        axs[0].set_yscale("log")
        axs[0].legend()

        # Plot MAE over iterations
        if add_marker:
            axs[1].plot(mae_values, label="MAE", marker="o", markersize=4)
        else:
            axs[1].plot(mae_values, label="MAE")
        if self.iteration < self.max_iterations:
            axs[1].scatter(
                self.iteration, mae_values[-1], color="red", marker="x", label="Stopped"
            )
        axs[1].set_xlabel("Iteration")
        axs[1].set_ylabel("MAE")
        axs[1].set_yscale("log")
        axs[1].legend()

        # Plot objective function values
        if add_marker:
            axs[2].plot(
                self.f_values, label="Objective function", marker="o", markersize=4
            )
        else:
            axs[2].plot(self.f_values, label="Objective function")
        if self.iteration < self.max_iterations:
            axs[2].scatter(
                self.iteration,
                self.f_values[-1],
                color="red",
                marker="x",
                label="Stopped",
            )
        axs[2].set_xlabel("Iteration")
        axs[2].set_ylabel("Objective function")
        axs[2].set_yscale("log")
        axs[2].legend()

        fig.suptitle(
            f"Convergence Plots for Constrained Convex Forward-Backward in {self.cv_time:.3f}s using {self.memory_used / 1024:.2f} KB"
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(visuals_path, f"ConstrainedConvexForwardBackward.png"))
        self.logger.info(
            f"Saved convergence plots to {visuals_path}/ConstrainedConvexForwardBackward.png"
        )
        plt.close()


class ConstrainedConvexGradientDescent(FixedPointAlgorithm):
    def __init__(
        self,
        name,
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
        super().__init__(name, f, logger=logger, verbose=verbose)
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
        # Determine gamma for this iteration
        if callable(self.gamma):
            gamma_n = self.gamma(self.iteration)
        elif isinstance(self.gamma, float):
            gamma_n = self.gamma
        elif isinstance(self.gamma, int):
            gamma_n = self.gamma
        else:
            raise ValueError("gamma must be a callable, float, or int.")

        # Gradient step
        m_new = m - gamma_n * self.current_gradient
        return m_new

    def is_converged(self, m, threshold=1e-6):
        p_sum = 0
        for B_i, d_i in zip(self.B_list, self.d_list):
            t_i = sp.linalg.spsolve(self.A, B_i @ m)
            rhs = self.C_star @ (self.C @ t_i - d_i)
            p_i = sp.linalg.spsolve(self.A_star, rhs)
            p_sum += B_i.conj().T @ p_i
        grad = p_sum + self.mu * m
        self.current_gradient = grad
        return np.linalg.norm(grad) < threshold

    def plot_algorithm_convergence(self, m, visuals_path, add_marker=False):
        m_pred = self.x_values
        mse_values = mse(m_pred, m)
        mae_values = mae(m_pred, m)

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        # Plot MSE over iterations
        if add_marker:
            axs[0].plot(mse_values, label="MSE", marker="o", markersize=4)
        else:
            axs[0].plot(mse_values, label="MSE")
        if self.iteration < self.max_iterations:
            axs[0].scatter(
                self.iteration, mse_values[-1], color="red", marker="x", label="Stopped"
            )
        axs[0].set_xlabel("Iteration")
        axs[0].set_ylabel("MSE")
        axs[0].set_yscale("log")
        axs[0].legend()

        # Plot MAE over iterations
        if add_marker:
            axs[1].plot(mae_values, label="MAE", marker="o", markersize=4)
        else:
            axs[1].plot(mae_values, label="MAE")
        if self.iteration < self.max_iterations:
            axs[1].scatter(
                self.iteration, mae_values[-1], color="red", marker="x", label="Stopped"
            )
        axs[1].set_xlabel("Iteration")
        axs[1].set_ylabel("MAE")
        axs[1].set_yscale("log")
        axs[1].legend()

        # Plot objective function values
        if add_marker:
            axs[2].plot(
                self.f_values, label="Objective function", marker="o", markersize=4
            )
        else:
            axs[2].plot(self.f_values, label="Objective function")
        if self.iteration < self.max_iterations:
            axs[2].scatter(
                self.iteration,
                self.f_values[-1],
                color="red",
                marker="x",
                label="Stopped",
            )
        axs[2].set_xlabel("Iteration")
        axs[2].set_ylabel("Objective function")
        axs[2].set_yscale("log")
        axs[2].legend()

        fig.suptitle(
            f"Convergence Plots for Constrained Convex Gradient Descent in {self.cv_time:.3f}s using {self.memory_used / 1024:.2f} KB"
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(visuals_path, f"ConstrainedConvexGradientDescent.png"))
        self.logger.info(
            f"Saved convergence plots to {visuals_path}/ConstrainedConvexGradientDescent.png"
        )
        plt.close()

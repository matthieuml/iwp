import abc
import time
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import seaborn as sns


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

    def plot_f_over_iterations(self, ax=None):
        with sns.axes_style("darkgrid"):
            if ax is not None:
                ax.plot(self.f_values)
                ax.set_xlabel("Iteration")
                ax.set_ylabel("f(x)")
                ax.set_yscale("log")
                ax.set_title(f"{self.name} - {self.cv_time:.3f}s")
            else:
                plt.plot(self.f_values)
                plt.xlabel("Iteration")
                plt.ylabel("f(x)")
                plt.yscale("log")
                plt.title(f"f(x) over iterations for {self.name} - {self.cv_time:.3f}s")
                plt.show()

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
            self.f_values.append(self.f(x))
            self.iteration += 1
            if self.logger:
                if self.verbose:
                    self.logger.info(f"Iteration {self.iteration}: f(x) = {self.f_values[-1]:.6f}, time = {time.time() - t0:.3f}s")
                else:
                    self.logger.debug(f"Iteration {self.iteration}: f(x) = {self.f_values[-1]:.6f}, time = {time.time() - t0:.3f}s")
        self.cv_time = time.time() - t0
        _, peak = tracemalloc.get_traced_memory()
        self.memory_used = peak
        tracemalloc.stop()
        if self.logger:
            if self.iteration < self.max_iterations:
                self.logger.info(f"Converged after {self.iteration} iterations in {self.cv_time:.3f} seconds with {self.memory_used / 1024:.2f} KB memory used.")
            else:
                self.logger.info(f"Stopped after {self.max_iterations} iterations in {self.cv_time:.3f} seconds with {self.memory_used / 1024:.2f} KB memory used.")
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
        y_new = x - (1.0 / self.K) *  self.current_gradient
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
        self.gamma = - (ratio - 1) / (ratio + 1)
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


class PenalizedConvexForwardBackward(FixedPointAlgorithm):
    def __init__(self, name, f, D, D_star, E, E_star, d, mu, gamma_schedule, lambda_schedule, logger=None, verbose=True):
        super().__init__(name, f, logger=logger, verbose=verbose)
        self.D = D
        self.D_star = D_star
        self.E = E
        self.E_star = E_star
        self.d = d
        self.mu = mu
        self.gamma_schedule = gamma_schedule
        self.lambda_schedule = lambda_schedule
        self.x_prev = None
        self.current_gradient = None

    def step(self, x):
        # If y_prev is None, initialize it to x
        if self.x_prev is None:
            self.x_prev = x
        # Get current gamma and lambda
        if callable(self.gamma_schedule):
            gamma_n = self.gamma_schedule(self.iteration)
        elif isinstance(self.gamma_schedule, float):
            gamma_n = self.gamma_schedule
        elif isinstance(self.gamma_schedule, int):
            gamma_n = self.gamma_schedule
        else:
            raise ValueError("gamma_schedule must be a callable, float, or int.")
        if callable(self.lambda_schedule):
            lambda_n = self.lambda_schedule(self.iteration)
        elif isinstance(self.lambda_schedule, float):
            lambda_n = self.lambda_schedule
        elif isinstance(self.lambda_schedule, int):
            lambda_n = self.lambda_schedule
        else:
            raise ValueError("lambda_schedule must be a callable, float, or int.")
        # Gradient step
        z = x - gamma_n * self.current_gradient
        # Proximal step
        w = sp.linalg.spsolve(self.E @ self.E_star, self.E @ z)
        y = z - self.E_star(w)
        # Relaxation step
        x_new = self.x_prev + lambda_n * (y - self.x_prev)
        # Store for next iteration
        self.y_prev = x_new.copy()
        return x_new

    def is_converged(self, x, threshold=1e-6):
        self.current_gradient = self.D_star(self.D @ x - self.d) + self.mu * x
        return np.linalg.norm(self.current_gradient) < threshold


class PenalizedConvexGradientDescent(FixedPointAlgorithm):
    def __init__(self, name, f, A, A_star, C, C_star, B_list, d_list, mu, gamma_schedule, logger=None, verbose=True):
        super().__init__(name, f, logger=logger, verbose=verbose)
        self.A = A
        self.A_star = A_star
        self.C = C
        self.C_star = C_star
        self.B_list = B_list
        self.d_list = d_list
        self.mu = mu
        self.gamma_schedule = gamma_schedule
        self.current_gradient = None

    def step(self, m):
        # Determine gamma for this iteration
        if callable(self.gamma_schedule):
            gamma_n = self.gamma_schedule(self.iteration)
        elif isinstance(self.gamma_schedule, float):
            gamma_n = self.gamma_schedule
        elif isinstance(self.gamma_schedule, int):
            gamma_n = self.gamma_schedule
        else:
            raise ValueError("gamma_schedule must be a callable, float, or int.")

        # Gradient step
        m_new = m - gamma_n * self.current_gradient
        return m_new

    def is_converged(self, m, threshold=1e-6):
        p_sum = 0
        for B_i, d_i in zip(self.B_list, self.d_list):
            t_i = sp.linalg.spsolve(self.A, B_i @ m)
            rhs = self.C_star(self.C @ t_i - d_i)
            p_i = sp.linalg.spsolve(self.A_star, rhs)
            p_sum += B_i.T @ p_i
        grad = p_sum + self.mu * m
        self.current_gradient = grad
        return np.linalg.norm(grad) < threshold

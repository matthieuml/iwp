import abc
import time
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt
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


class StronglyConvexNesterovAcceleratedGradientDescent(FixedPointAlgorithm):
    def __init__(self, name, f, df, K, mu, logger=None, verbose=True):
        super().__init__(name, f, logger=logger, verbose=verbose)
        self.df = df
        self.K = K
        self.mu = mu
        ratio = np.sqrt(K / mu)
        self.gamma = - (ratio - 1) / (ratio + 1)
        self.y_prev = None

    def step(self, x):
        # If y_prev is None, initialize it to x
        if self.y_prev is None:
            self.y_prev = x
        # Compute gradient step
        y_new = x - (1.0 / self.K) * self.df(x)
        # Nesterov acceleration
        x_new = y_new + self.gamma * (y_new - self.y_prev)
        # Store for next iteration
        self.y_prev = y_new
        return x_new

    def is_converged(self, x, threshold=1e-6):
        return np.linalg.norm(self.df(x)) < threshold
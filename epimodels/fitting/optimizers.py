"""
Optimizers for model fitting.

Provides multiple backend implementations:
- ScipyOptimizer: Uses scipy.optimize
- JAXOptimizer: Uses optimistix for GPU-accelerated optimization
- NevergradOptimizer: Uses nevergrad for derivative-free optimization
"""

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class OptimizerResult:
    """Result from an optimizer."""

    best_params: NDArray[np.floating]
    best_loss: float
    success: bool
    message: str
    n_evaluations: int
    n_iterations: int | None = None
    backend_result: Any = None
    loss_history: list[float] = field(default_factory=list)


class Optimizer(ABC):
    """Base class for optimizers."""

    @abstractmethod
    def minimize(
        self,
        objective_fn: Callable[[NDArray[np.floating]], float],
        initial_params: NDArray[np.floating],
        bounds: list[tuple[float, float]] | None = None,
        callback: Callable[[int, NDArray[np.floating], float], None] | None = None,
    ) -> OptimizerResult:
        """
        Minimize the objective function.

        Args:
            objective_fn: Function to minimize, takes parameter array and returns loss
            initial_params: Initial parameter values
            bounds: List of (lower, upper) bounds for each parameter
            callback: Optional callback function called each iteration with
                     (iteration, params, loss)

        Returns:
            OptimizerResult with optimization results
        """
        pass


class ScipyOptimizer(Optimizer):
    """
    Scipy-based optimizer.

    Supports multiple methods from scipy.optimize:
    - 'L-BFGS-B': Quasi-Newton with bounds (default)
    - 'BFGS': Quasi-Newton without bounds
    - 'Nelder-Mead': Simplex method
    - 'Powell': Powell's method
    - 'CG': Conjugate gradient
    - 'differential_evolution': Global optimization
    - 'basinhopping': Basin-hopping global optimization
    """

    LOCAL_METHODS = ["L-BFGS-B", "BFGS", "Nelder-Mead", "Powell", "CG", "TNC", "SLSQP"]
    GLOBAL_METHODS = ["differential_evolution", "basinhopping"]

    def __init__(
        self,
        method: str = "L-BFGS-B",
        max_iterations: int = 1000,
        tolerance: float = 1e-8,
        options: dict[str, Any] | None = None,
    ):
        """
        Args:
            method: Optimization method name
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            options: Additional method-specific options
        """
        self.method = method.lower()
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.options = options or {}
        self._n_evals = 0
        self._loss_history: list[float] = []

    def minimize(
        self,
        objective_fn: Callable[[NDArray[np.floating]], float],
        initial_params: NDArray[np.floating],
        bounds: list[tuple[float, float]] | None = None,
        callback: Callable[[int, NDArray[np.floating], float], None] | None = None,
    ) -> OptimizerResult:
        from scipy.optimize import basinhopping, differential_evolution, minimize

        self._n_evals = 0
        self._loss_history = []
        iteration_counter = [0]

        def wrapped_objective(params):
            self._n_evals += 1
            loss = objective_fn(params)
            self._loss_history.append(loss)
            return loss

        def scipy_callback(xk):
            iteration_counter[0] += 1
            if callback is not None:
                loss = objective_fn(xk)
                callback(iteration_counter[0], xk, loss)

        if self.method == "differential_evolution":
            if bounds is None:
                raise ValueError("differential_evolution requires bounds")

            result = differential_evolution(
                wrapped_objective,
                bounds=bounds,
                maxiter=self.max_iterations,
                tol=self.tolerance,
                callback=lambda xk, convergence: scipy_callback(xk) if callback else None,
                **self.options,
            )

            return OptimizerResult(
                best_params=result.x,
                best_loss=result.fun,
                success=result.success,
                message=result.message if hasattr(result, "message") else "Optimization complete",
                n_evaluations=self._n_evals,
                n_iterations=result.nit if hasattr(result, "nit") else None,
                backend_result=result,
                loss_history=self._loss_history,
            )

        elif self.method == "basinhopping":
            minimizer_kwargs = {
                "method": "L-BFGS-B",
                "options": {"maxiter": self.max_iterations // 10},
            }
            if bounds is not None:
                minimizer_kwargs["bounds"] = bounds

            result = basinhopping(
                wrapped_objective,
                initial_params,
                minimizer_kwargs=minimizer_kwargs,
                niter=self.max_iterations // 10,
                callback=lambda x, f, accept: scipy_callback(x) if callback else None,
                **self.options,
            )

            return OptimizerResult(
                best_params=result.x,
                best_loss=result.fun,
                success=True,
                message="Basin hopping complete",
                n_evaluations=self._n_evals,
                n_iterations=result.nit if hasattr(result, "nit") else None,
                backend_result=result,
                loss_history=self._loss_history,
            )

        else:
            options = {
                "maxiter": self.max_iterations,
                "disp": False,
            }
            options.update(self.options)

            method = (
                self.method.upper() if self.method.upper() in self.LOCAL_METHODS else "L-BFGS-B"
            )

            result = minimize(
                wrapped_objective,
                initial_params,
                method=method,
                bounds=bounds,
                callback=scipy_callback if callback else None,
                options=options,
            )

            return OptimizerResult(
                best_params=result.x,
                best_loss=result.fun,
                success=result.success,
                message=result.message if hasattr(result, "message") else "Optimization complete",
                n_evaluations=self._n_evals,
                n_iterations=result.nit if hasattr(result, "nit") else None,
                backend_result=result,
                loss_history=self._loss_history,
            )


class JAXOptimizer(Optimizer):
    """
    JAX/optimistix-based optimizer for GPU acceleration.

    Provides gradient-based optimization with automatic differentiation.
    Requires jax and optimistix to be installed.
    """

    def __init__(
        self,
        method: str = "adam",
        learning_rate: float = 0.01,
        max_iterations: int = 1000,
        tolerance: float = 1e-8,
        options: dict[str, Any] | None = None,
    ):
        """
        Args:
            method: Optimization method ('adam', 'sgd', 'rmsprop', 'lamb', 'adabelief')
            learning_rate: Learning rate for gradient descent
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            options: Additional method-specific options
        """
        self.method = method.lower()
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.options = options or {}

    def _check_dependencies(self):
        try:
            import jax
            import optimistix
        except ImportError as e:
            raise ImportError(
                "JAXOptimizer requires jax and optimistix. "
                "Install with: pip install jax optimistix"
            ) from e

    def minimize(
        self,
        objective_fn: Callable[[NDArray[np.floating]], float],
        initial_params: NDArray[np.floating],
        bounds: list[tuple[float, float]] | None = None,
        callback: Callable[[int, NDArray[np.floating], float], None] | None = None,
    ) -> OptimizerResult:
        self._check_dependencies()

        import jax.numpy as jnp
        import optimistix as optx

        self._loss_history = []

        def loss_fn(params, args):
            loss = float(objective_fn(np.array(params)))
            self._loss_history.append(loss)
            return jnp.array(loss)

        solver_methods = {
            "adam": optx.Adam,
            "adabelief": optx.Adam,
            "rmsprop": optx.RMSProp,
            "sgd": optx.SGD,
        }

        solver_cls = solver_methods.get(self.method, optx.Adam)
        solver = solver_cls(
            learning_rate=self.learning_rate,
            **self.options,
        )

        y0 = jnp.array(initial_params)

        try:
            result = optx.minimise(
                loss_fn,
                solver,
                y0,
                has_aux=False,
                max_steps=self.max_iterations,
                tol=self.tolerance,
            )
            best_params = np.array(result)
            best_loss = float(loss_fn(result, None))
            success = True
            message = "Optimization complete"
        except Exception as e:
            best_params = np.array(y0)
            best_loss = float(loss_fn(y0, None))
            success = False
            message = str(e)

        if bounds is not None:
            for i, (lower, upper) in enumerate(bounds):
                best_params[i] = np.clip(best_params[i], lower, upper)

        return OptimizerResult(
            best_params=best_params,
            best_loss=best_loss,
            success=success,
            message=message,
            n_evaluations=len(self._loss_history),
            n_iterations=len(self._loss_history),
            loss_history=self._loss_history,
        )


class NevergradOptimizer(Optimizer):
    """
    Nevergrad-based optimizer for derivative-free optimization.

    Provides population-based optimization methods that don't require gradients.
    Requires nevergrad to be installed.
    """

    def __init__(
        self,
        method: str = "TwoPointsDE",
        budget: int = 1000,
        num_workers: int = 1,
        options: dict[str, Any] | None = None,
    ):
        """
        Args:
            method: Nevergrad optimization method name
            budget: Number of function evaluations allowed
            num_workers: Number of parallel workers
            options: Additional method-specific options
        """
        self.method = method
        self.budget = budget
        self.num_workers = num_workers
        self.options = options or {}

    def _check_dependencies(self):
        try:
            import nevergrad
        except ImportError as e:
            raise ImportError(
                "NevergradOptimizer requires nevergrad. " "Install with: pip install nevergrad"
            ) from e

    def minimize(
        self,
        objective_fn: Callable[[NDArray[np.floating]], float],
        initial_params: NDArray[np.floating],
        bounds: list[tuple[float, float]] | None = None,
        callback: Callable[[int, NDArray[np.floating], float], None] | None = None,
    ) -> OptimizerResult:
        self._check_dependencies()

        import nevergrad as ng

        self._loss_history = []
        n_evals = [0]

        dim = len(initial_params)

        if bounds is not None:
            params = ng.p.Array(
                shape=(dim,),
                lower=[b[0] for b in bounds],
                upper=[b[1] for b in bounds],
            )
            params.value = initial_params.tolist()
        else:
            params = ng.p.Array(shape=(dim,))
            params.value = initial_params.tolist()

        optimizer = ng.optimizers.registry[self.method](
            parametrization=params,
            budget=self.budget,
            num_workers=self.num_workers,
        )

        def wrapped_objective(x):
            n_evals[0] += 1
            loss = objective_fn(np.array(x))
            self._loss_history.append(loss)
            if callback is not None:
                callback(n_evals[0], np.array(x), loss)
            return loss

        result = optimizer.minimize(wrapped_objective)

        best_params = np.array(result.value)
        best_loss = float(result.loss) if result.loss is not None else objective_fn(best_params)

        return OptimizerResult(
            best_params=best_params,
            best_loss=best_loss,
            success=True,
            message="Nevergrad optimization complete",
            n_evaluations=n_evals[0],
            n_iterations=self.budget,
            loss_history=self._loss_history,
        )


class MultiStartOptimizer(Optimizer):
    """
    Multi-start optimization wrapper.

    Runs an optimizer from multiple starting points and returns the best result.
    """

    def __init__(
        self,
        base_optimizer: Optimizer,
        n_starts: int = 10,
        sampling_method: str = "latin_hypercube",
        seed: int | None = None,
    ):
        """
        Args:
            base_optimizer: The underlying optimizer to use
            n_starts: Number of starting points
            sampling_method: Method for generating starting points
                            ('latin_hypercube', 'random', 'sobol')
            seed: Random seed for reproducibility
        """
        self.base_optimizer = base_optimizer
        self.n_starts = n_starts
        self.sampling_method = sampling_method
        self.seed = seed

    def _generate_starting_points(
        self,
        bounds: list[tuple[float, float]],
        n_points: int,
    ) -> list[NDArray[np.floating]]:
        """Generate starting points within bounds."""
        rng = np.random.default_rng(self.seed)
        dim = len(bounds)
        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])

        if self.sampling_method == "latin_hypercube":
            from scipy.stats import qmc

            sampler = qmc.LatinHypercube(d=dim, seed=self.seed)
            samples = sampler.random(n=n_points)
            points = qmc.scale(samples, lower, upper)
            return [points[i] for i in range(n_points)]

        elif self.sampling_method == "sobol":
            from scipy.stats import qmc

            sampler = qmc.Sobol(d=dim, scramble=True, seed=self.seed)
            samples = sampler.random_base2(m=int(np.ceil(np.log2(n_points))))
            samples = samples[:n_points]
            points = qmc.scale(samples, lower, upper)
            return [points[i] for i in range(n_points)]

        else:
            points = []
            for _ in range(n_points):
                point = rng.uniform(lower, upper)
                points.append(point)
            return points

    def minimize(
        self,
        objective_fn: Callable[[NDArray[np.floating]], float],
        initial_params: NDArray[np.floating],
        bounds: list[tuple[float, float]] | None = None,
        callback: Callable[[int, NDArray[np.floating], float], None] | None = None,
    ) -> OptimizerResult:
        if bounds is None:
            bounds = [
                (initial_params[i] - 1.0, initial_params[i] + 1.0)
                for i in range(len(initial_params))
            ]
            warnings.warn("No bounds provided for multi-start optimization. Using default bounds.")

        starting_points = self._generate_starting_points(bounds, self.n_starts)
        starting_points[0] = initial_params.copy()

        best_result = None
        best_loss = float("inf")
        all_losses = []

        start_callback = callback

        for i, start_point in enumerate(starting_points):

            def run_callback(iteration, params, loss):
                if start_callback is not None:
                    start_callback(i * 1000 + iteration, params, loss)

            result = self.base_optimizer.minimize(
                objective_fn,
                start_point,
                bounds,
                callback=run_callback if start_callback else None,
            )

            all_losses.extend(result.loss_history)

            if result.best_loss < best_loss:
                best_loss = result.best_loss
                best_result = result

        if best_result is None:
            best_result = self.base_optimizer.minimize(objective_fn, initial_params, bounds)

        best_result.loss_history = all_losses
        best_result.n_evaluations = len(all_losses)

        return best_result


__all__ = [
    "Optimizer",
    "OptimizerResult",
    "ScipyOptimizer",
    "JAXOptimizer",
    "NevergradOptimizer",
    "MultiStartOptimizer",
]

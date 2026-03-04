"""
Solver abstraction layer for ODE integration.

This module provides a unified interface for different ODE solvers,
allowing users to select between scipy and diffrax backends.

Example:
    from epimodels.solvers import ScipySolver, DiffraxSolver

    # Use scipy (default)
    solver = ScipySolver(method='RK45')

    # Use diffrax (JAX-accelerated)
    solver = DiffraxSolver(solver='Tsit5')
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional
import numpy as np
from numpy.typing import NDArray


class SolverResult:
    """
    Container for solver results.

    Provides a unified interface for results from different solvers.

    Attributes:
        t: Time points array
        y: Solution array (n_vars, n_times)
    """

    def __init__(self, t: NDArray[np.floating], y: NDArray[np.floating]):
        self.t = np.asarray(t)
        self.y = np.asarray(y)

    def __repr__(self) -> str:
        return f"SolverResult(t={self.t.shape}, y={self.y.shape})"


class SolverBase(ABC):
    """
    Abstract base class for ODE solvers.

    All solvers must implement the solve() method that returns
    a SolverResult object.
    """

    @abstractmethod
    def solve(
        self,
        fn: Callable[[float, list[float]], list[float]],
        t_span: tuple[float, float],
        y0: list[float],
        **kwargs,
    ) -> SolverResult:
        """
        Solve the ODE system.

        Args:
            fn: The ODE function dy/dt = fn(t, y)
            t_span: Time span (t0, tf)
            y0: Initial conditions
            **kwargs: Additional solver-specific options

        Returns:
            SolverResult containing t and y arrays
        """
        pass


class ScipySolver(SolverBase):
    """
    Scipy solve_ivp wrapper.

    Provides access to all scipy ODE solvers.

    Args:
        method: Integration method. One of:
            - 'RK45' (default): Explicit Runge-Kutta of order 5(4)
            - 'RK23': Explicit Runge-Kutta of order 3(2)
            - 'DOP853': Explicit Runge-Kutta of order 8
            - 'Radau': Implicit Runge-Kutta of the Radau IIA family
            - 'BDF': Implicit multi-step variable-order method
            - 'LSODA': Adams/BDF method with automatic stiffness detection

    Example:
        >>> solver = ScipySolver(method='LSODA')
        >>> result = solver.solve(fn, (0, 100), y0)
    """

    METHODS = ["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"]

    def __init__(self, method: str = "RK45"):
        if method not in self.METHODS:
            raise ValueError(f"Unknown method '{method}'. Available methods: {self.METHODS}")
        self.method = method

    def solve(
        self,
        fn: Callable[[float, list[float]], list[float]],
        t_span: tuple[float, float],
        y0: list[float],
        **kwargs,
    ) -> SolverResult:
        from scipy.integrate import solve_ivp

        sol = solve_ivp(fn, t_span, y0, method=self.method, **kwargs)

        return SolverResult(sol.t, sol.y)


class DiffraxSolver(SolverBase):
    """
    Diffrax/JAX solver wrapper.

    Provides access to JAX-accelerated ODE solvers with GPU support.
    Diffrax must be installed separately.

    Args:
        solver: Solver name. One of:
            - 'Tsit5' (default): 5th order Tsitouras method
            - 'Dopri5': 5th order Dormand-Prince method
            - 'Dopri8': 8th order Dormand-Prince method
            - 'Euler': Euler method
            - 'Heun': Heun's method
            - 'Midpoint': Midpoint method
            - 'Ralston': Ralston's method
        dt: Time step (None for adaptive stepping)
        adaptive: Use adaptive stepping (default: True)
        rtol: Relative tolerance (default: 1e-3)
        atol: Absolute tolerance (default: 1e-6)
        **solver_kwargs: Additional solver-specific options

    Example:
        >>> solver = DiffraxSolver(solver='Tsit5', adaptive=True)
        >>> result = solver.solve(fn, (0, 100), y0)

    Note:
        Requires diffrax and jax to be installed:
        pip install diffrax jax
    """

    SOLVERS = ["Tsit5", "Dopri5", "Dopri8", "Euler", "Heun", "Midpoint", "Ralston"]

    def __init__(
        self,
        solver: str = "Tsit5",
        dt: Optional[float] = None,
        adaptive: bool = True,
        rtol: float = 1e-3,
        atol: float = 1e-6,
        **solver_kwargs,
    ):
        if solver not in self.SOLVERS:
            raise ValueError(f"Unknown solver '{solver}'. Available solvers: {self.SOLVERS}")
        self.solver_name = solver
        self.dt = dt
        self.adaptive = adaptive
        self.rtol = rtol
        self.atol = atol
        self.solver_kwargs = solver_kwargs

    def _get_solver_class(self):
        """Get the diffrax solver class by name."""
        try:
            from diffrax import Tsit5, Dopri5, Dopri8, Euler, Heun, Midpoint, Ralston
        except ImportError as e:
            raise ImportError(
                "DiffraxSolver requires diffrax and jax to be installed. "
                "Install with: pip install diffrax jax"
            ) from e

        solvers = {
            "Tsit5": Tsit5,
            "Dopri5": Dopri5,
            "Dopri8": Dopri8,
            "Euler": Euler,
            "Heun": Heun,
            "Midpoint": Midpoint,
            "Ralston": Ralston,
        }
        return solvers[self.solver_name]

    def solve(
        self,
        fn: Callable[[float, list[float]], list[float]],
        t_span: tuple[float, float],
        y0: list[float],
        **kwargs,
    ) -> SolverResult:
        import jax.numpy as jnp
        from diffrax import diffeqsolve, ODETerm, SaveAt, PIDController

        solver_cls = self._get_solver_class()
        solver = solver_cls(**self.solver_kwargs)

        if self.adaptive:
            stepsize_controller = PIDController(rtol=self.rtol, atol=self.atol)
        else:
            stepsize_controller = None

        def term_fn(t, y, args):
            # Don't convert t to float - JAX needs it as-is for tracing
            return jnp.array(fn(t, list(y)))

        term = ODETerm(term_fn)

        y0_jax = jnp.array(y0)

        t0, t1 = t_span
        saveat = SaveAt(ts=jnp.linspace(t0, t1, 100))

        sol = diffeqsolve(
            term,
            solver,
            t0=t0,
            t1=t1,
            y0=y0_jax,
            dt0=self.dt,
            stepsize_controller=stepsize_controller,
            saveat=saveat,
            **kwargs,
        )

        return SolverResult(np.array(sol.ts), np.array(sol.ys.T))


def get_default_solver() -> SolverBase:
    """
    Get the default solver (ScipySolver with RK45).

    Returns:
        ScipySolver instance with default settings
    """
    return ScipySolver(method="RK45")


__all__ = [
    "SolverResult",
    "SolverBase",
    "ScipySolver",
    "DiffraxSolver",
    "get_default_solver",
]

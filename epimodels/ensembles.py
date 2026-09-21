"""
Ensemble simulation with parameter/initial-condition uncertainty.

Runs many deterministic simulations of a model, each with parameters
(and optionally initial conditions) drawn by a sampler, and collects the
trajectories into a :class:`TraceEnsemble` supporting quantiles and
uncertainty-band plots.

Example:
    >>> from epimodels.continuous import SIR
    >>> from epimodels.ensembles import simulate_ensemble
    >>>
    >>> model = SIR()
    >>> def sample_params():
    ...     return {"beta": rng.uniform(1.5, 2.5), "gamma": 0.1}
    >>>
    >>> ensemble = simulate_ensemble(
    ...     model, n_sims=200, param_sampler=sample_params,
    ...     initial_conditions=[1000, 1, 0], trange=[0, 100], totpop=1001,
    ... )
    >>> ensemble.plot_band("I")
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from epimodels import BaseModel

__all__ = ["TraceEnsemble", "simulate_ensemble"]


@dataclass
class TraceEnsemble:
    """
    Collection of simulation trajectories across parameter realizations.

    Attributes:
        time: 1D time grid shared by all realizations.
        traces: Dict mapping state variable name to a 2D array
            of shape ``(n_sims, n_points)``.
        params: List of parameter dicts used for each realization
            (empty if not recorded).
    """

    time: NDArray[np.floating]
    traces: dict[str, NDArray[np.floating]]
    params: list[dict[str, Any]] = field(default_factory=list)

    def __len__(self) -> int:
        return int(self.traces[next(iter(self.traces))].shape[0])

    def quantiles(
        self, q: list[float] | tuple[float, ...] = (0.025, 0.5, 0.975)
    ) -> dict[str, dict[float, NDArray[np.floating]]]:
        """
        Compute quantiles of each variable across realizations.

        Args:
            q: Quantiles to compute (default: median + 95% band).

        Returns:
            Dict mapping variable name to a dict of quantile value ->
            1D array over time.
        """
        return {
            var: {qi: np.quantile(data, qi, axis=0) for qi in q}
            for var, data in self.traces.items()
        }

    def summary(self) -> dict[str, dict[str, float]]:
        """
        Summary statistics (mean/std/quantiles at the final time point).
        """
        out: dict[str, dict[str, float]] = {}
        for var, data in self.traces.items():
            final = data[:, -1]
            out[var] = {
                "final_mean": float(final.mean()),
                "final_std": float(final.std()),
                "final_q025": float(np.quantile(final, 0.025)),
                "final_q975": float(np.quantile(final, 0.975)),
                "peak_mean": float(data.max(axis=1).mean()),
            }
        return out

    def plot_band(
        self,
        var: str,
        ci: float = 0.95,
        ax: Any = None,
        show_reps: bool = False,
        alpha_reps: float = 0.1,
        **plot_kwargs,
    ) -> Any:
        """
        Plot the median trajectory with a confidence band.

        Args:
            var: State variable to plot.
            ci: Credible interval width (default 0.95).
            ax: Matplotlib axes (creates a new figure if None).
            show_reps: Whether to overlay individual realizations.
            alpha_reps: Transparency for individual realizations.
            **plot_kwargs: Extra kwargs forwarded to the median plot line.

        Returns:
            The matplotlib axes used.
        """
        import matplotlib.pyplot as plt

        if var not in self.traces:
            raise KeyError(
                f"Unknown variable '{var}'. Available: {sorted(self.traces)}"
            )

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))

        data = self.traces[var]
        lo_q = (1 - ci) / 2
        hi_q = 1 - lo_q
        lo = np.quantile(data, lo_q, axis=0)
        hi = np.quantile(data, hi_q, axis=0)
        median = np.quantile(data, 0.5, axis=0)

        if show_reps:
            for i in range(data.shape[0]):
                ax.plot(self.time, data[i], alpha=alpha_reps, color="gray", linewidth=0.5)

        label = plot_kwargs.pop("label", f"{var} median")
        ax.plot(self.time, median, label=label, **plot_kwargs)
        ax.fill_between(self.time, lo, hi, alpha=0.25, label=f"{int(ci * 100)}% band")
        ax.set_xlabel("Time")
        ax.set_ylabel(var)
        ax.legend(loc=0)
        return ax


def simulate_ensemble(
    model: BaseModel,
    n_sims: int,
    param_sampler: Callable[[], dict[str, Any]],
    initial_conditions: list[float] | Callable[[], list[float]],
    trange: list[float],
    totpop: float,
    n_points: int = 201,
    n_jobs: int = 1,
    validate: bool = False,
    record_params: bool = True,
    **solver_kwargs,
) -> TraceEnsemble:
    """
    Run an ensemble of simulations with sampled parameters.

    Each realization draws parameters from ``param_sampler`` (and initial
    conditions, if a callable is given), runs the model deterministically,
    and records the trajectory on a common time grid of ``n_points`` points.

    Args:
        model: Model instance to simulate (a copy is used per realization).
        n_sims: Number of realizations.
        param_sampler: Zero-argument callable returning a parameter dict.
        initial_conditions: Fixed initial conditions, or a zero-argument
            callable returning them.
        trange: Time range ``[t0, tf]``.
        totpop: Total population.
        n_points: Number of points on the common output time grid
            (default 201). Only used by solvers that support ``t_eval``
            (continuous models).
        n_jobs: Number of parallel processes (default 1 = serial).
        validate: Whether to validate each realization (default False for speed).
        record_params: Whether to keep the sampled parameters in the result.
        **solver_kwargs: Forwarded to the model call (e.g. ``method``).

    Returns:
        TraceEnsemble with all trajectories.
    """
    if n_sims < 1:
        raise ValueError(f"n_sims must be >= 1, got {n_sims}")
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}")

    from epimodels import BaseModel as _BaseModel

    if not isinstance(model, _BaseModel):
        raise TypeError(f"model must be a BaseModel instance, got {type(model).__name__}")

    # Common output grid so realizations are comparable (adaptive solvers
    # would otherwise return different grids per parameter set).
    t_eval = np.linspace(float(trange[0]), float(trange[1]), n_points)
    solver_kwargs.setdefault("t_eval", t_eval)

    def _run_one(_seed: int) -> tuple[dict[str, NDArray[np.floating]], dict[str, Any]]:
        run_model = model.copy()
        params = param_sampler()
        ics = initial_conditions() if callable(initial_conditions) else list(initial_conditions)
        run_model(ics, trange, totpop, params, validate=validate, **solver_kwargs)
        return run_model.traces, params

    results: list[tuple[dict[str, NDArray[np.floating]], dict[str, Any]]]
    if n_jobs > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(_run_one, range(n_sims)))
    else:
        results = [_run_one(i) for i in range(n_sims)]

    time = np.asarray(results[0][0]["time"])
    var_names = [k for k in results[0][0] if k != "time"]

    # Truncate to the shortest trajectory in case grids differ slightly.
    n_out = min(len(r[0]["time"]) for r in results)

    traces: dict[str, NDArray[np.floating]] = {}
    for var in var_names:
        traces[var] = np.array([r[0][var][:n_out] for r in results])

    sampled_params = [r[1] for r in results] if record_params else []

    return TraceEnsemble(time=time[:n_out], traces=traces, params=sampled_params)

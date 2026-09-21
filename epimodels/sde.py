"""
Stochastic differential equation (SDE) simulation of continuous epidemic models.

Wraps any :class:`~epimodels.continuous.ContinuousModel` as a Langevin-type
SDE and integrates it with diffrax/JAX:

    dy = drift(t, y) dt + D(t, y) dW

where the drift is the model's deterministic right-hand side and the
diffusion defaults to the demographic square-root approximation
``D_ii = sqrt(noise_scale * |drift_i|)`` — exact for one-way flows and a
standard approximation for net flows. A custom diffusion matrix function
can be supplied for full control.

Requires the ``jax`` extra: ``pip install epimodels[jax]``.

Note:
    Models whose ``_model`` uses numpy-specific functions (e.g. ``np.tanh``),
    interpolation callables or internal history state cannot run under JAX
    tracing. The classic SIR/SIS/SIRS/SEIR-family models work as-is.

Example:
    >>> from epimodels.continuous import SIR
    >>> from epimodels.sde import SDEModel
    >>>
    >>> sde = SDEModel(SIR())
    >>> sde([999, 1, 0], [0, 100], 1000, {"beta": 2.0, "gamma": 0.5},
    ...     n_sims=50, seed=42)
    >>> sde.get_mean()["I"]      # mean trajectory
    >>> sde.get_quantiles(0.95)  # 95% band
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from epimodels.continuous import ContinuousModel

__all__ = ["SDEModel"]


class SDEModel:
    """
    Stochastic (demographic-noise) version of a continuous epidemic model.

    Args:
        base_model: The ContinuousModel whose deterministic dynamics provide
            the SDE drift.
        noise_scale: Multiplier on the default square-root diffusion
            (1.0 = demographic noise magnitude).
        diffusion: Optional custom diffusion function
            ``(t, y, params) -> (n_vars, n_noise)`` overriding the default.
    """

    def __init__(
        self,
        base_model: ContinuousModel,
        noise_scale: float = 1.0,
        diffusion: Callable[[Any, Any, dict], Any] | None = None,
    ):
        from epimodels.continuous import ContinuousModel

        if not isinstance(base_model, ContinuousModel):
            raise TypeError(
                f"base_model must be a ContinuousModel, got {type(base_model).__name__}"
            )
        if noise_scale < 0:
            raise ValueError(f"noise_scale must be non-negative, got {noise_scale}")

        self.base_model = base_model
        self.noise_scale = noise_scale
        self.custom_diffusion = diffusion
        self.traces: dict[str, Any] = {}
        self._n_sims = 0

    def _drift(self, t, y, params):
        return self.base_model._model(t, y, params)

    def __call__(
        self,
        inits: list[float],
        trange: list[float],
        totpop: float,
        params: dict[str, Any],
        n_sims: int = 1,
        dt: float = 0.05,
        n_points: int = 201,
        seed: int | None = None,
        t_eval: Any = None,
        validate: bool = True,
        **solver_kwargs,
    ) -> None:
        """
        Simulate the SDE one or more times.

        Args:
            inits: Initial conditions.
            trange: Time range [t0, tf].
            totpop: Total population.
            params: Parameter dict (fixed during the simulation).
            n_sims: Number of independent trajectories.
            dt: Integration time step for the Euler-Maruyama solver.
            n_points: Number of output grid points (ignored if t_eval given).
            seed: Random seed for reproducibility.
            t_eval: Optional explicit output time grid.
            validate: Whether to validate parameters/initial conditions.
            **solver_kwargs: Forwarded to ``diffrax.diffeqsolve``.
        """
        try:
            import diffrax
            import jax
            import jax.numpy as jnp
        except ImportError as e:
            raise ImportError(
                "SDEModel requires jax and diffrax. "
                "Install with: pip install epimodels[jax]"
            ) from e

        if validate:
            self.base_model.validate_parameters(params)
            self.base_model.validate_initial_conditions(list(inits), totpop)
        if n_sims < 1:
            raise ValueError(f"n_sims must be >= 1, got {n_sims}")
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")

        base = self.base_model
        model_params = {**params, "N": totpop}

        # Wrap so that models returning Python lists produce arrays matching
        # the pytree structure of y0 (required by diffrax terms).
        def drift_fn(t, y, args):
            return jnp.asarray(base._model(t, y, model_params))

        if self.custom_diffusion is not None:
            user_diffusion = self.custom_diffusion

            def diffusion_fn(t, y, args):
                return jnp.asarray(user_diffusion(t, y, model_params))

        else:
            def diffusion_fn(t, y, args):
                d = jnp.asarray(base._model(t, y, model_params))
                return jnp.diag(jnp.sqrt(self.noise_scale * jnp.abs(d)))

        t0, t1 = float(trange[0]), float(trange[1])
        ts = t_eval if t_eval is not None else np.linspace(t0, t1, n_points)
        ts_j = jnp.asarray(ts)
        saveat = diffrax.SaveAt(ts=ts_j)
        y0 = jnp.asarray(inits, dtype=jnp.float32)
        n_vars = len(inits)

        key = jax.random.PRNGKey(0 if seed is None else seed)

        all_trajectories = np.empty((n_sims, len(ts), n_vars))
        for i in range(n_sims):
            key_i, key = jax.random.split(key)
            brownian = diffrax.VirtualBrownianTree(
                t0, t1, tol=dt / 4, shape=(n_vars,), key=key_i
            )
            terms: Any = diffrax.MultiTerm(
                diffrax.ODETerm(drift_fn),
                diffrax.ControlTerm(diffusion_fn, brownian),
            )
            sol = diffrax.diffeqsolve(
                terms,
                # In diffrax >= 0.6 the classic Euler-Maruyama update for SDEs
                # is provided by the Euler solver driven by a ControlTerm.
                diffrax.Euler(),
                t0=t0,
                t1=t1,
                y0=y0,
                dt0=dt,
                saveat=saveat,
                **solver_kwargs,
            )
            all_trajectories[i] = np.array(sol.ys)

        var_names = list(base.state_variables.keys())
        self.traces = {"time": np.asarray(ts)}
        for vi, vname in enumerate(var_names):
            self.traces[vname] = (
                all_trajectories[:, :, vi] if n_sims > 1 else all_trajectories[0, :, vi]
            )
        self._n_sims = n_sims

    def _require_run(self) -> None:
        if not self.traces:
            raise ValueError("No simulation results. Run the model first.")

    def get_mean(self) -> dict[str, NDArray[np.floating]]:
        """Mean trajectory across replicates (trivial for n_sims=1)."""
        self._require_run()
        out: dict[str, NDArray[np.floating]] = {"time": self.traces["time"]}
        for vname in self.base_model.state_variables:
            data = np.asarray(self.traces[vname])
            out[vname] = data.mean(axis=0) if self._n_sims > 1 else data
        return out

    def get_quantiles(
        self, ci: float = 0.95
    ) -> dict[str, tuple[NDArray[np.floating], NDArray[np.floating]]]:
        """Credible band per variable across replicates."""
        import warnings

        self._require_run()
        if self._n_sims < 2:
            warnings.warn(
                "Quantiles from a single trajectory; run with n_sims > 1 for a band.",
                UserWarning,
                stacklevel=2,
            )
        lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
        out: dict[str, tuple[NDArray[np.floating], NDArray[np.floating]]] = {}
        for vname in self.base_model.state_variables:
            data = np.atleast_2d(np.asarray(self.traces[vname]))
            out[vname] = (
                np.quantile(data, lo_q, axis=0),
                np.quantile(data, hi_q, axis=0),
            )
        return out

    def plot_traces(self, var: str | None = None, show_mean: bool = True) -> Any:
        """Plot replicate trajectories (and their mean) for one variable."""
        import matplotlib.pyplot as plt

        self._require_run()
        if var is None:
            var = next(iter(self.base_model.state_variables))
        if var not in self.base_model.state_variables:
            raise KeyError(f"Unknown variable '{var}'")

        time = self.traces["time"]
        data = np.asarray(self.traces[var])
        if self._n_sims > 1:
            for i in range(self._n_sims):
                plt.plot(time, data[i], alpha=0.3, linewidth=0.6, color="gray")
            if show_mean:
                plt.plot(time, data.mean(axis=0), linewidth=2, label=f"{var} (mean)")
        else:
            plt.plot(time, data, linewidth=2, label=var)
        plt.xlabel("Time")
        plt.ylabel(var)
        plt.legend(loc=0)
        return plt.gca()

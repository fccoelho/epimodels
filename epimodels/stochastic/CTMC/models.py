"""
CTMC (Continuous-Time Markov Chain) stochastic epidemic models.

This module provides stochastic counterparts to the deterministic ODE models
in epimodels.continuous. Models inherit from BaseModel and follow the same
calling convention, with additional support for multiple replicates,
confidence intervals, and event tracking.

Example:
    >>> from epimodels.stochastic.CTMC import SIR
    >>> model = SIR()
    >>> model(
    ...     inits=[990, 10, 0], trange=[0, 100], totpop=1000,
    ...     params={'beta': 0.3, 'gamma': 0.1},
    ...     reps=100, seed=42
    ... )
    >>> model.get_mean()['I']  # mean infectious trajectory
    >>> model.get_quantiles([0.025, 0.975])  # 95% CI
"""

import copy as copy_module
import warnings
from abc import abstractmethod
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
from matplotlib import pyplot as plt

from epimodels import BaseModel
from epimodels.exceptions import ValidationError
from epimodels.stochastic.CTMC.solvers import (
    CTMCSolverBase,
    GillespieSolver,
)


def _run_single_trajectory(args):
    """Worker function for parallel execution."""
    solver, propensity_fn, tmat, initial_state, t_span, params, seed_val = args
    rng = np.random.default_rng(seed_val)
    return solver.solve(propensity_fn, tmat, initial_state, t_span, params, rng)


class CTMCModel(BaseModel):
    """
    Base class for Continuous-Time Markov Chain epidemic models.

    Subclasses must define:
        - state_variables, parameters, model_type (from BaseModel)
        - events: OrderedDict mapping event name -> description
        - transitions() -> np.ndarray (n_vars x n_events, dtype=int64)
        - propensity(params, state) -> np.ndarray (n_events,)

    Subclasses may define:
        - R0 property
        - diagram property

    The calling convention matches ContinuousModel/DiscreteModel with
    additional stochastic parameters (reps, seed, n_jobs, n_points).
    """

    events: OrderedDict[str, str]

    def __init__(self) -> None:
        super().__init__()
        self.events = OrderedDict()
        self._trajectories = []
        self._n_reps = 0

    @abstractmethod
    def transitions(self) -> np.ndarray:
        """
        Return the state-change (transition) matrix.

        Shape (n_vars, n_events), dtype int64.
        Column j specifies how each state variable changes when event j fires.
        """
        ...

    @abstractmethod
    def propensity(
        self, params: dict[str, float], state: np.ndarray
    ) -> np.ndarray:
        """
        Compute propensity values for all events given current state.

        :param params: Parameter dict (includes 'N' for total population)
        :param state: Current state vector (n_vars,)
        :return: Propensity array (n_events,)
        """
        ...

    def __call__(
        self,
        inits: list[int],
        trange: list[float],
        totpop: int,
        params: dict[str, float],
        reps: int = 1,
        solver: CTMCSolverBase | None = None,
        validate: bool = True,
        seed: int | None = None,
        n_jobs: int = 1,
        n_points: int = 100,
        **kwargs,
    ) -> None:
        """
        Run the stochastic simulation.

        :param inits: Initial conditions (non-negative integers)
        :param trange: Time range [t0, tf]
        :param totpop: Total population size
        :param params: Dictionary of parameters
        :param reps: Number of stochastic replicates
        :param solver: CTMC solver instance (default: GillespieSolver)
        :param validate: Whether to validate inputs
        :param seed: Random seed for reproducibility
        :param n_jobs: Number of parallel jobs (1 = serial)
        :param n_points: Number of time grid points for output
        """
        if validate:
            self.validate_parameters(params)
            self._validate_stochastic_inits(inits, totpop)
            self.validate_time_range(trange)
            self._validate_stochastic_params(reps, n_jobs, n_points)

        self.param_values = OrderedDict(
            (k, params[k]) for k in self.parameters.keys()
        )

        if solver is not None:
            self.solver = solver
        else:
            self.solver = GillespieSolver()

        self._reps = reps
        self._seed = seed
        self._n_jobs = n_jobs
        self._n_points = n_points

        t_span = (float(trange[0]), float(trange[1]))
        t_grid = np.linspace(t_span[0], t_span[1], n_points)
        params_with_N = dict(params)
        params_with_N["N"] = totpop

        initial_state = np.array(inits, dtype=np.int64)
        tmat = self.transitions()

        def propensity_fn(p, state):
            return self.propensity(p, state)

        self._trajectories = self._run_replicates(
            propensity_fn,
            tmat,
            initial_state,
            t_span,
            params_with_N,
            reps,
            seed,
            n_jobs,
        )

        n_vars = len(self.state_variables)
        var_names = list(self.state_variables.keys())
        all_states = np.zeros((reps, n_points, n_vars))

        for i, traj in enumerate(self._trajectories):
            all_states[i] = traj.interpolate_to_grid(t_grid)

        self.traces = {"time": t_grid}
        for vi, vname in enumerate(var_names):
            if reps == 1:
                self.traces[vname] = all_states[0, :, vi]
            else:
                self.traces[vname] = all_states[:, :, vi]

        self._store_metadata(t_grid)

    def _validate_stochastic_inits(
        self, inits: list[int], totpop: int
    ) -> None:
        """Validate initial conditions for stochastic models."""
        if len(inits) < len(self.state_variables):
            raise ValidationError(
                f"Expected at least {len(self.state_variables)} initial "
                f"conditions, got {len(inits)}"
            )
        for i, val in enumerate(inits):
            if val < 0:
                raise ValidationError(
                    f"Initial condition at index {i} must be non-negative, "
                    f"got {val}"
                )
            if isinstance(val, float) and val != int(val):
                raise ValidationError(
                    f"Initial condition at index {i} must be an integer for "
                    f"stochastic models, got {val}"
                )

    def _validate_stochastic_params(
        self, reps: int, n_jobs: int, n_points: int
    ) -> None:
        """Validate stochastic-specific parameters."""
        if reps < 1:
            raise ValidationError(f"reps must be >= 1, got {reps}")
        if n_jobs < 1:
            raise ValidationError(f"n_jobs must be >= 1, got {n_jobs}")
        if n_points < 2:
            raise ValidationError(f"n_points must be >= 2, got {n_points}")

    def _run_replicates(
        self,
        propensity_fn,
        tmat,
        initial_state,
        t_span,
        params,
        reps,
        seed,
        n_jobs,
    ):
        """Run all replicates, optionally in parallel."""
        if n_jobs == 1 or reps == 1:
            return self._run_serial(
                propensity_fn, tmat, initial_state, t_span, params, reps, seed
            )
        else:
            return self._run_parallel(
                propensity_fn,
                tmat,
                initial_state,
                t_span,
                params,
                reps,
                seed,
                n_jobs,
            )

    def _run_serial(
        self, propensity_fn, tmat, initial_state, t_span, params, reps, seed
    ):
        """Run replicates serially."""
        rng = np.random.default_rng(seed)
        trajectories = []
        for _ in range(reps):
            traj = self.solver.solve(
                propensity_fn, tmat, initial_state, t_span, params, rng
            )
            trajectories.append(traj)
        return trajectories

    def _run_parallel(
        self,
        propensity_fn,
        tmat,
        initial_state,
        t_span,
        params,
        reps,
        seed,
        n_jobs,
    ):
        """Run replicates in parallel using ProcessPoolExecutor."""
        base_seed = seed if seed is not None else np.random.default_rng().integers(0, 2**31)
        args_list = [
            (
                self.solver,
                propensity_fn,
                tmat,
                initial_state,
                t_span,
                params,
                base_seed + i,
            )
            for i in range(reps)
        ]

        trajectories = [None] * reps
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(_run_single_trajectory, args): i
                for i, args in enumerate(args_list)
            }
            for future in as_completed(futures):
                idx = futures[future]
                trajectories[idx] = future.result()

        return trajectories

    def _store_metadata(self, t_grid):
        """Store derived metadata after simulation."""
        self._n_reps = self._reps
        self._t_grid = t_grid

    def get_replicate(self, i: int) -> dict[str, np.ndarray]:
        """
        Get results from a single replicate as a named dict.

        :param i: Replicate index (0-based)
        :return: Dict mapping variable names to 1D arrays, plus 'time'
        :raises ValueError: If no simulation has been run
        :raises IndexError: If replicate index is out of range
        """
        if not self.traces:
            raise ValueError("No simulation results. Run the model first.")

        if i < 0 or i >= self._n_reps:
            raise IndexError(
                f"Replicate index {i} out of range [0, {self._n_reps})"
            )

        result = {"time": self.traces["time"]}
        for vname in self.state_variables:
            data = self.traces[vname]
            if self._n_reps == 1:
                result[vname] = data
            else:
                result[vname] = data[i]
        return result

    def get_mean(self) -> dict[str, np.ndarray]:
        """
        Get mean trajectory across replicates.

        :return: Dict mapping variable names to 1D mean arrays, plus 'time'
        """
        if not self.traces:
            raise ValueError("No simulation results. Run the model first.")

        result = {"time": self.traces["time"]}
        for vname in self.state_variables:
            data = self.traces[vname]
            if self._n_reps == 1:
                result[vname] = data
            else:
                result[vname] = data.mean(axis=0)
        return result

    def get_variance(self) -> dict[str, np.ndarray]:
        """
        Get variance trajectory across replicates.

        :return: Dict mapping variable names to 1D variance arrays, plus 'time'
        """
        if not self.traces:
            raise ValueError("No simulation results. Run the model first.")

        result = {"time": self.traces["time"]}
        for vname in self.state_variables:
            data = self.traces[vname]
            if self._n_reps == 1:
                result[vname] = np.zeros_like(data)
            else:
                result[vname] = data.var(axis=0)
        return result

    def get_quantiles(
        self, quantiles: list[float] | None = None
    ) -> dict[float, dict[str, np.ndarray]]:
        """
        Get quantile trajectories across replicates.

        :param quantiles: List of quantiles to compute (default: [0.025, 0.5, 0.975])
        :return: Dict mapping quantile value -> {var: array}, with 'time' in each
        """
        if not self.traces:
            raise ValueError("No simulation results. Run the model first.")

        if quantiles is None:
            quantiles = [0.025, 0.5, 0.975]

        result = {}
        for q in quantiles:
            sub = {"time": self.traces["time"]}
            for vname in self.state_variables:
                data = self.traces[vname]
                if self._n_reps == 1:
                    sub[vname] = data
                else:
                    sub[vname] = np.quantile(data, q, axis=0)
            result[q] = sub
        return result

    def get_event_times(
        self, event: str | None = None
    ) -> dict[str, list[float]] | list[float]:
        """
        Get event occurrence times across all replicates.

        :param event: If given, return times for this event only.
            If None, return dict for all events.
        :return: Dict mapping event name -> list of times,
            or list of times if event is specified.
        :raises ValueError: If no simulation or event name not found
        """
        if not self._trajectories:
            raise ValueError("No simulation results. Run the model first.")

        event_names = list(self.events.keys())
        result: dict[str, list[float]] = {name: [] for name in event_names}

        for traj in self._trajectories:
            evt_map = traj.event_times_by_index()
            for idx, times in evt_map.items():
                if idx < len(event_names):
                    name = event_names[idx]
                    result[name].extend(times)

        if event is not None:
            if event not in result:
                raise ValueError(
                    f"Unknown event '{event}'. Available: {list(event_names)}"
                )
            return result[event]
        return result

    def plot_traces(
        self,
        vars: list[str] | None = None,
        show_ci: bool = True,
        show_reps: bool = False,
        alpha: float = 0.1,
        ci: float = 0.95,
    ) -> None:
        """
        Plot simulation results.

        For multiple replicates, plots the mean trajectory with optional
        confidence interval bands and individual replicate traces.

        :param vars: Variables to plot (default: all state variables)
        :param show_ci: Show confidence interval band (multi-rep only)
        :param show_reps: Show individual replicate trajectories
        :param alpha: Transparency for individual replicate lines
        :param ci: Confidence interval width (default 0.95)
        """
        if not self.traces:
            raise ValueError("No simulation results. Run the model first.")

        if vars is None:
            vars = list(self.state_variables.keys())

        time = self.traces["time"]

        for vname in vars:
            if vname not in self.state_variables:
                continue

            data = self.traces[vname]

            if self._n_reps > 1 and show_reps:
                for i in range(self._n_reps):
                    plt.plot(time, data[i], alpha=alpha, linewidth=0.5)

            if self._n_reps > 1:
                mean = data.mean(axis=0)
            else:
                mean = data

            plt.plot(time, mean, linewidth=2, label=vname)

            if self._n_reps > 1 and show_ci:
                lo_q = (1 - ci) / 2
                hi_q = 1 - lo_q
                lo = np.quantile(data, lo_q, axis=0)
                hi = np.quantile(data, hi_q, axis=0)
                plt.fill_between(time, lo, hi, alpha=0.2)

        plt.legend(loc=0)
        plt.grid(True)
        plt.xlabel("Time")
        plt.ylabel("Count")
        plt.title(f"{self.model_type} (stochastic)")

    def summary(self) -> dict[str, Any]:
        """
        Return epidemic summary statistics with stochastic extensions.

        Includes mean peak, timing, attack rate, and for multi-replicate
        runs: extinction probability, confidence intervals on peak.
        """
        if not self.traces:
            raise ValueError("No simulation results. Run the model first.")

        stats: dict[str, Any] = {"model": self.model_type or "unknown"}

        time = self.traces["time"]
        stats["t_start"] = float(time[0])
        stats["t_end"] = float(time[-1])
        stats["reps"] = self._n_reps

        mean_data = self.get_mean()

        if "I" in mean_data:
            I_mean = mean_data["I"]
            stats["peak_I_mean"] = float(I_mean.max())
            stats["peak_time_mean"] = float(time[I_mean.argmax()])

        if "S" in mean_data:
            stats["final_S_mean"] = float(mean_data["S"][-1])

        if "R" in mean_data:
            stats["final_R_mean"] = float(mean_data["R"][-1])

        if "S" in mean_data and "I" in mean_data:
            S0 = mean_data["S"][0]
            S_final = mean_data["S"][-1]
            if S0 > 0:
                stats["attack_rate_mean"] = float((S0 - S_final) / S0)

        if self._n_reps > 1:
            I_data = self.traces.get("I")
            if I_data is not None:
                extinction_count = 0
                peak_values = []
                for i in range(self._n_reps):
                    rep_I = I_data[i]
                    if rep_I[-1] < 1:
                        extinction_count += 1
                    peak_values.append(float(rep_I.max()))
                stats["extinction_probability"] = extinction_count / self._n_reps
                stats["peak_I_median"] = float(np.median(peak_values))
                stats["peak_I_ci"] = (
                    float(np.quantile(peak_values, 0.025)),
                    float(np.quantile(peak_values, 0.975)),
                )

        return stats

    def to_dataframe(self, replicate: int | None = None) -> "Any":
        """
        Return simulation results as a pandas DataFrame.

        :param replicate: If None, return mean trajectory.
            If int, return specific replicate.
        :return: DataFrame with time and state variable columns
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for to_dataframe(). "
                "Install with: pip install pandas"
            )

        if not self.traces:
            raise ValueError("No simulation results. Run the model first.")

        if replicate is not None:
            data = self.get_replicate(replicate)
        else:
            data = self.get_mean()

        return pd.DataFrame(data)

    def reset(self) -> None:
        """Clear simulation results and internal state."""
        super().reset()
        self._trajectories = []
        self._n_reps = 0


class SIR(CTMCModel):
    """
    Stochastic SIR (Susceptible-Infectious-Removed) CTMC model.

    Events:
        infection: S + I -> 2I   (rate = beta * S * I / N)
        recovery:  I -> R        (rate = gamma * I)

    Example:
        >>> model = SIR()
        >>> model([990, 10, 0], [0, 100], 1000,
        ...       {'beta': 0.3, 'gamma': 0.1}, reps=100, seed=42)
    """

    def __init__(self) -> None:
        super().__init__()
        self.state_variables = OrderedDict(
            {"S": "Susceptible", "I": "Infectious", "R": "Removed"}
        )
        self.parameters = OrderedDict(
            {"beta": r"$\beta$", "gamma": r"$\gamma$"}
        )
        self.events = OrderedDict(
            {"infection": "Transmission event", "recovery": "Recovery event"}
        )
        self.model_type = "SIR CTMC"

    def transitions(self) -> np.ndarray:
        return np.array([[-1, 0], [1, -1], [0, 1]], dtype=np.int64)

    def propensity(
        self, params: dict[str, float], state: np.ndarray
    ) -> np.ndarray:
        S, I, R = state
        beta, gamma, N = params["beta"], params["gamma"], params["N"]
        return np.array([beta * S * I / N, gamma * I])

    @property
    def diagram(self) -> str:
        return r"""flowchart LR
        S(Susceptible) -->|$$\beta$$| I(Infectious)
        I -->|$$\gamma$$| R(Removed)
        """

    @property
    def R0(self) -> float | None:
        if (
            self.param_values
            and "beta" in self.param_values
            and "gamma" in self.param_values
        ):
            return float(self.param_values["beta"] / self.param_values["gamma"])
        return None


class SIS(CTMCModel):
    """
    Stochastic SIS (Susceptible-Infectious-Susceptible) CTMC model.

    Events:
        infection: S + I -> 2I   (rate = beta * S * I / N)
        recovery:  I -> S        (rate = gamma * I)

    Example:
        >>> model = SIS()
        >>> model([500, 10], [0, 100], 510,
        ...       {'beta': 0.3, 'gamma': 0.1}, reps=100, seed=42)
    """

    def __init__(self) -> None:
        super().__init__()
        self.state_variables = OrderedDict(
            {"S": "Susceptible", "I": "Infectious"}
        )
        self.parameters = OrderedDict(
            {"beta": r"$\beta$", "gamma": r"$\gamma$"}
        )
        self.events = OrderedDict(
            {"infection": "Transmission event", "recovery": "Recovery event"}
        )
        self.model_type = "SIS CTMC"

    def transitions(self) -> np.ndarray:
        return np.array([[-1, 1], [1, -1]], dtype=np.int64)

    def propensity(
        self, params: dict[str, float], state: np.ndarray
    ) -> np.ndarray:
        S, I = state
        beta, gamma, N = params["beta"], params["gamma"], params["N"]
        return np.array([beta * S * I / N, gamma * I])

    @property
    def diagram(self) -> str:
        return r"""flowchart LR
        S(Susceptible) -->|$$\beta$$| I(Infectious)
        I -->|$$\gamma$$| S
        """

    @property
    def R0(self) -> float | None:
        if (
            self.param_values
            and "beta" in self.param_values
            and "gamma" in self.param_values
        ):
            return float(self.param_values["beta"] / self.param_values["gamma"])
        return None


class SIRS(CTMCModel):
    """
    Stochastic SIRS (Susceptible-Infectious-Removed-Susceptible) CTMC model.

    Events:
        infection: S + I -> 2I   (rate = beta * S * I / N)
        recovery:  I -> R        (rate = gamma * I)
        waning:    R -> S        (rate = xi * R)

    Example:
        >>> model = SIRS()
        >>> model([990, 10, 0], [0, 100], 1000,
        ...       {'beta': 0.3, 'gamma': 0.1, 'xi': 0.01},
        ...       reps=100, seed=42)
    """

    def __init__(self) -> None:
        super().__init__()
        self.state_variables = OrderedDict(
            {"S": "Susceptible", "I": "Infectious", "R": "Removed"}
        )
        self.parameters = OrderedDict(
            {"beta": r"$\beta$", "gamma": r"$\gamma$", "xi": r"$\xi$"}
        )
        self.events = OrderedDict(
            {
                "infection": "Transmission event",
                "recovery": "Recovery event",
                "waning": "Immunity waning event",
            }
        )
        self.model_type = "SIRS CTMC"

    def transitions(self) -> np.ndarray:
        return np.array(
            [[-1, 0, 1], [1, -1, 0], [0, 1, -1]], dtype=np.int64
        )

    def propensity(
        self, params: dict[str, float], state: np.ndarray
    ) -> np.ndarray:
        S, I, R = state
        beta, gamma, xi, N = (
            params["beta"],
            params["gamma"],
            params["xi"],
            params["N"],
        )
        return np.array([beta * S * I / N, gamma * I, xi * R])

    @property
    def diagram(self) -> str:
        return r"""flowchart LR
        S(Susceptible) -->|$$\beta$$| I(Infectious)
        I -->|$$\gamma$$| R(Removed)
        R -->|$$\xi$$| S
        """

    @property
    def R0(self) -> float | None:
        if (
            self.param_values
            and "beta" in self.param_values
            and "gamma" in self.param_values
        ):
            return float(self.param_values["beta"] / self.param_values["gamma"])
        return None


class SEIR(CTMCModel):
    """
    Stochastic SEIR (Susceptible-Exposed-Infectious-Removed) CTMC model.

    Events:
        exposure:   S + I -> E + I  (rate = beta * S * I / N)
        infection:  E -> I          (rate = epsilon * E)
        recovery:   I -> R          (rate = gamma * I)

    Example:
        >>> model = SEIR()
        >>> model([990, 0, 10, 0], [0, 100], 1000,
        ...       {'beta': 0.3, 'gamma': 0.1, 'epsilon': 0.5},
        ...       reps=100, seed=42)
    """

    def __init__(self) -> None:
        super().__init__()
        self.state_variables = OrderedDict(
            {
                "S": "Susceptible",
                "E": "Exposed",
                "I": "Infectious",
                "R": "Removed",
            }
        )
        self.parameters = OrderedDict(
            {
                "beta": r"$\beta$",
                "gamma": r"$\gamma$",
                "epsilon": r"$\epsilon$",
            }
        )
        self.events = OrderedDict(
            {
                "exposure": "Exposure event",
                "infection": "Onset of infectiousness",
                "recovery": "Recovery event",
            }
        )
        self.model_type = "SEIR CTMC"

    def transitions(self) -> np.ndarray:
        return np.array(
            [[-1, 0, 0], [1, -1, 0], [0, 1, -1], [0, 0, 1]], dtype=np.int64
        )

    def propensity(
        self, params: dict[str, float], state: np.ndarray
    ) -> np.ndarray:
        S, E, I, R = state
        beta, gamma, epsilon, N = (
            params["beta"],
            params["gamma"],
            params["epsilon"],
            params["N"],
        )
        return np.array([beta * S * I / N, epsilon * E, gamma * I])

    @property
    def diagram(self) -> str:
        return r"""flowchart LR
        S(Susceptible) -->|$$\beta$$| E(Exposed)
        E -->|$$\epsilon$$| I(Infectious)
        I -->|$$\gamma$$| R(Removed)
        """

    @property
    def R0(self) -> float | None:
        if (
            self.param_values
            and "beta" in self.param_values
            and "gamma" in self.param_values
        ):
            return float(self.param_values["beta"] / self.param_values["gamma"])
        return None


__all__ = [
    "CTMCModel",
    "SIR",
    "SIS",
    "SIRS",
    "SEIR",
]

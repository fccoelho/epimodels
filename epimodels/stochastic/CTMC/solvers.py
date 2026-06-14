"""
CTMC Solver abstraction layer for stochastic simulation.

Provides a unified interface for different CTMC solvers (Gillespie SSA,
tau-leaping, etc.), analogous to epimodels.solvers for ODE models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from typing import Callable


@dataclass
class CTMCTrajectory:
    """
    Container for a single CTMC stochastic trajectory.

    Stores the raw event-driven trajectory (actual event times and states)
    and provides interpolation onto a regular time grid.

    Attributes:
        times: Event times including t0. Shape (n_events+1,).
        states: State vectors after each event. Shape (n_events+1, n_vars).
        event_indices: Index of the event that fired at each step. Shape (n_events,).
        steps: Number of events that occurred.
    """

    times: NDArray[np.floating]
    states: NDArray[np.floating]
    event_indices: NDArray[np.intp]
    steps: int

    def interpolate_to_grid(
        self, t_grid: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """
        Interpolate trajectory onto a regular time grid using step-function
        (piecewise constant) interpolation.

        The state at each grid point is the state of the system at that time,
        i.e. the state after the last event that occurred before that time.

        :param t_grid: Sorted array of time points. Shape (n_points,).
        :return: State array on the grid. Shape (n_points, n_vars).
        """
        t_grid = np.asarray(t_grid, dtype=float)
        n_points = len(t_grid)
        n_vars = self.states.shape[1]
        result = np.zeros((n_points, n_vars))

        j = 0
        for i, t in enumerate(t_grid):
            while j < self.steps and self.times[j + 1] <= t:
                j += 1
            result[i] = self.states[j]

        return result

    @property
    def duration(self) -> float:
        """Total simulated duration."""
        return float(self.times[-1] - self.times[0])

    def event_times_by_index(self) -> dict[int, list[float]]:
        """Return dict mapping event index -> list of occurrence times."""
        result: dict[int, list[float]] = {}
        for k, idx in enumerate(self.event_indices):
            t = self.times[k + 1]
            result.setdefault(int(idx), []).append(t)
        return result


class CTMCSolverBase(ABC):
    """
    Abstract base class for CTMC solvers.

    All solvers must implement the solve() method that returns
    a CTMCTrajectory object.
    """

    @abstractmethod
    def solve(
        self,
        propensity_fn: Callable[[dict, NDArray], NDArray],
        transition_matrix: NDArray[np.int64],
        initial_state: NDArray[np.int64],
        t_span: tuple[float, float],
        params: dict,
        rng: np.random.Generator,
        **kwargs,
    ) -> CTMCTrajectory:
        """
        Run a single stochastic trajectory.

        :param propensity_fn: Function(params, state) -> propensities array
        :param transition_matrix: State change matrix (n_vars, n_events)
        :param initial_state: Initial state vector (n_vars,)
        :param t_span: (t0, tf) time span
        :param params: Model parameters dict
        :param rng: numpy random Generator instance
        :return: CTMCTrajectory with event-driven trajectory
        """
        ...

class GillespieSolver(CTMCSolverBase):
    """
    Gillespie Direct Method (SSA) solver.

    The classic stochastic simulation algorithm that generates exact
    trajectories of the CTMC. Each step:
      1. Compute propensities a_i for all events
      2. Draw time to next event: tau ~ Exp(sum(a_i))
      3. Select event j with probability a_j / sum(a_i)
      4. Update state by transition_matrix[:, j]

    This implementation uses np.searchsorted for O(log n) event selection
    instead of the slower multinomial sampling approach.

    Example:
        >>> solver = GillespieSolver()
        >>> traj = solver.solve(propensity_fn, tmat, state0, (0, 100), params, rng)
    """

    def solve(
        self,
        propensity_fn: Callable[[dict, NDArray], NDArray],
        transition_matrix: NDArray[np.int64],
        initial_state: NDArray[np.int64],
        t_span: tuple[float, float],
        params: dict,
        rng: np.random.Generator,
        **kwargs,
    ) -> CTMCTrajectory:
        t0, tf = t_span
        state = np.array(initial_state, dtype=np.int64)
        tmat = np.asarray(transition_matrix, dtype=np.int64)
        n_events = tmat.shape[1]

        times_list = [t0]
        states_list = [state.copy()]
        event_indices_list = []

        capacity = 1024
        tc = float(t0)

        while tc < tf:
            a = propensity_fn(params, state)
            a0 = float(a.sum())

            if a0 <= 0.0:
                break

            tau = rng.exponential(1.0 / a0)
            tc += tau

            if tc > tf:
                break

            cumsum = np.cumsum(a)
            r = rng.uniform(0.0, a0)
            event_idx = int(np.searchsorted(cumsum, r))
            if event_idx >= n_events:
                event_idx = n_events - 1

            state = state + tmat[:, event_idx]

            times_list.append(tc)
            states_list.append(state.copy())
            event_indices_list.append(event_idx)

        if len(times_list) == 1:
            times_list.append(tf)
            states_list.append(state.copy())

        return CTMCTrajectory(
            times=np.array(times_list),
            states=np.array(states_list),
            event_indices=np.array(event_indices_list, dtype=np.intp),
            steps=len(event_indices_list),
        )

class TauLeapingSolver(CTMCSolverBase):
    """
    Tau-leaping solver for CTMC models.

    Approximates the CTMC by leaping over fixed time steps tau,
    sampling the number of occurrences of each event via Poisson
    distributions.

    Parameters:
        tau: Time step size
    """

    def __init__(self, tau: float = 0.05):
        if tau <= 0:
            raise ValueError("tau must be positive")
        self.tau = tau

    def solve(
        self,
        propensity_fn: Callable[[dict, NDArray], NDArray],
        transition_matrix: NDArray[np.int64],
        initial_state: NDArray[np.int64],
        t_span: tuple[float, float],
        params: dict,
        rng: np.random.Generator,
        **kwargs,
    ) -> CTMCTrajectory:
        t0, tf = t_span
        state = np.array(initial_state, dtype=np.int64)
        tmat = np.asarray(transition_matrix, dtype=np.int64)
        n_events = tmat.shape[1]

        times_list = [t0]
        states_list = [state.copy()]
        event_indices_list = []

        tc = float(t0)

        while tc < tf:
            a = propensity_fn(params, state)
            a0 = float(a.sum())

            if a0 <= 0.0:
                break

            tau = min(self.tau, tf - tc)
            K = rng.poisson(a * tau)

            state = state + tmat @ K
            state = np.maximum(state, 0)

            tc += tau

            times_list.append(tc)
            states_list.append(state.copy())
            
            event_indices_list.append(-1)

        if len(times_list) == 1:
            times_list.append(tf)
            states_list.append(state.copy())

        return CTMCTrajectory(
            times=np.array(times_list),
            states=np.array(states_list),
            event_indices=np.array(event_indices_list, dtype=np.intp),
            steps=len(event_indices_list),
        )

class MidpointTauLeapingSolver(CTMCSolverBase):
    """
    Midpoint tau-leaping solver for CTMC models.

    Improves the standard tau-leaping method by estimating propensities
    at the midpoint of the interval, yielding better accuracy (second-order).

    Parameters:
        tau: Time step size
    """

    def __init__(self, tau: float = 0.05):
        if tau <= 0:
            raise ValueError("tau must be positive")
        self.tau = tau

    def solve(
        self,
        propensity_fn: Callable[[dict, NDArray], NDArray],
        transition_matrix: NDArray[np.int64],
        initial_state: NDArray[np.int64],
        t_span: tuple[float, float],
        params: dict,
        rng: np.random.Generator,
        **kwargs,
    ) -> CTMCTrajectory:
        t0, tf = t_span
        state = np.array(initial_state, dtype=np.int64)
        tmat = np.asarray(transition_matrix, dtype=np.int64)
        n_events = tmat.shape[1]

        times_list = [t0]
        states_list = [state.copy()]
        event_indices_list = []

        tc = float(t0)

        while tc < tf:
            a = propensity_fn(params, state)
            a0 = float(a.sum())

            if a0 <= 0.0:
                break

            tau = min(self.tau, tf - tc)

            state_midpoint = state + 0.5 * tau * (tmat @ a)

            a_mid = propensity_fn(params, state_midpoint)
            a_mid = np.asarray(a_mid, dtype=float)

            K = rng.poisson(a_mid * tau)

            state = state + tmat @ K
            state = np.maximum(state, 0)

            tc += tau

            times_list.append(tc)
            states_list.append(state.copy())
            
            event_indices_list.append(-1)

        if len(times_list) == 1:
            times_list.append(tf)
            states_list.append(state.copy())

        return CTMCTrajectory(
            times=np.array(times_list),
            states=np.array(states_list),
            event_indices=np.array(event_indices_list, dtype=np.intp),
            steps=len(event_indices_list),
        )

class KLeapingSolver(CTMCSolverBase):
    """
    K-leaping solver for CTMC models.

    Advances the system by simulating k events at a time, assuming that
    transition rates remain approximately constant during the leap.

    Parameters:
        k: Number of events per leap
    """

    def __init__(self, k: int = 10):
        if k <= 0:
            raise ValueError("k must be a positive integer")
        self.k = k

    def solve(
        self,
        propensity_fn: Callable[[dict, NDArray], NDArray],
        transition_matrix: NDArray[np.int64],
        initial_state: NDArray[np.int64],
        t_span: tuple[float, float],
        params: dict,
        rng: np.random.Generator,
        **kwargs,
    ) -> CTMCTrajectory:
        t0, tf = t_span
        state = np.array(initial_state, dtype=np.int64)
        tmat = np.asarray(transition_matrix, dtype=np.int64)
        n_events = tmat.shape[1]

        times_list = [t0]
        states_list = [state.copy()]
        event_indices_list = []

        tc = float(t0)

        while tc < tf:
            a = propensity_fn(params, state)
            a0 = float(a.sum())

            if a0 <= 0.0:
                break

            tau = rng.gamma(shape=self.k, scale=1.0 / a0)

            if tc + tau > tf:
                tau = tf - tc
                k_eff = max(1, rng.poisson(a0 * tau))
            else:
                k_eff = self.k

            p = a / a0
            K = rng.multinomial(k_eff, p)

            state = state + tmat @ K
            state = np.maximum(state, 0)

            tc += tau

            times_list.append(tc)
            states_list.append(state.copy())
            
            event_indices_list.append(-1)

        if len(times_list) == 1:
            times_list.append(tf)
            states_list.append(state.copy())

        return CTMCTrajectory(
            times=np.array(times_list),
            states=np.array(states_list),
            event_indices=np.array(event_indices_list, dtype=np.intp),
            steps=len(event_indices_list),
        )


__all__ = [
    "CTMCTrajectory",
    "CTMCSolverBase",
    "GillespieSolver",
    "TauLeapingSolver",
    "MidpointTauLeapingSolver",
    "KLeapingSolver"
]

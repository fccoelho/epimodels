"""
Network-structured epidemic models (SIR/SIS on graphs).

Event-driven (Gillespie) simulations of epidemics on contact networks.
Graphs can be networkx graphs, adjacency dictionaries
``{node: [neighbors]}`` or symmetric adjacency matrices.

Note:
    networkx is optional (``pip install epimodels[network]``); plain
    adjacency dicts and matrices work without it.

Example:
    >>> import networkx as nx
    >>> from epimodels.network import NetworkSIR
    >>>
    >>> G = nx.barabasi_albert_graph(1000, 3, seed=0)
    >>> model = NetworkSIR(G)
    >>> model(5, [0, 50], {"beta": 0.3, "gamma": 0.1}, n_sims=20, seed=0)
    >>> model.final_size()          # attack rate per replicate
    >>> model.plot_traces("I")      # incidence band
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from epimodels import BaseModel

if TYPE_CHECKING:
    pass

__all__ = ["NetworkModel", "NetworkSIR", "NetworkSIS"]


def _adjacency_from_graph(graph: Any) -> tuple[list[Any], list[list[int]]]:
    """
    Normalize supported graph inputs to adjacency lists.

    Accepts networkx graphs, adjacency dicts ``{node: iterable}`` and
    2D adjacency matrices.

    Returns:
        (node_labels, adjacency_lists) with integer indices.
    """
    if hasattr(graph, "adj") and hasattr(graph, "nodes"):  # networkx-like
        nodes = list(graph.nodes())
        index = {n: i for i, n in enumerate(nodes)}
        adj = [[index[nb] for nb in graph.neighbors(n)] for n in nodes]
        return nodes, adj

    if isinstance(graph, dict):
        nodes = list(graph.keys())
        index = {n: i for i, n in enumerate(nodes)}
        adj = []
        for n in nodes:
            neighbors = graph[n]
            adj.append(
                [index[nb] for nb in (neighbors() if callable(neighbors) else neighbors)]
            )
        return nodes, adj

    matrix = np.asarray(graph)
    if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
        if not np.allclose(matrix, matrix.T):
            raise ValueError("Adjacency matrix must be symmetric (undirected graph)")
        nodes = list(range(matrix.shape[0]))
        adj = [np.flatnonzero(row > 0).tolist() for row in matrix]
        return nodes, adj

    raise TypeError(
        "graph must be a networkx graph, an adjacency dict {node: neighbors}, "
        "or a symmetric 2D adjacency matrix"
    )


def _pick_weighted(rng: np.random.Generator, weights: list[int]) -> int:
    """Pick an index proportional to ``weights``."""
    r = rng.random() * sum(weights)
    acc = 0.0
    for k, w in enumerate(weights):
        acc += w
        if r < acc:
            return k
    return len(weights) - 1


class NetworkModel(BaseModel):
    """
    Base class for network epidemic models.

    Subclasses implement :meth:`_simulate_one` with the event logic.

    Args:
        graph: networkx graph, adjacency dict or symmetric adjacency matrix.
    """

    keep_removed: bool = False

    def __init__(self, graph: Any):
        super().__init__()
        self.parameters = {"beta": r"$\beta$", "gamma": r"$\gamma$"}
        self.node_labels, self.adj = _adjacency_from_graph(graph)
        self.n_nodes = len(self.adj)
        self.state_variables = {"S": "Susceptible", "I": "Infectious"}
        self.model_type = type(self).__name__
        self.traces: dict[str, Any] = {}
        self._n_sims = 0

    # -- validation ---------------------------------------------------------

    def _validate_inits(self, inits: int | list) -> list[int]:
        """Normalize ``inits`` to a list of initially-infected node indices."""
        if isinstance(inits, bool):
            raise ValueError("inits must be an int or a list of node identifiers")
        if isinstance(inits, (int, np.integer)):
            k = int(inits)
            if not 1 <= k <= self.n_nodes:
                raise ValueError(
                    f"Number of initially infected nodes must be in [1, {self.n_nodes}]"
                )
            return list(range(k))
        if isinstance(inits, Iterable):
            chosen = []
            for node in inits:
                try:
                    idx = self.node_labels.index(node)
                except ValueError:
                    raise ValueError(f"Node {node!r} not in graph") from None
                chosen.append(idx)
            if not chosen:
                raise ValueError("inits list is empty")
            return chosen
        raise ValueError("inits must be an int (count) or a list of nodes")

    def _validate_params(self, params: dict[str, Any]) -> tuple[float, float]:
        from epimodels.exceptions import ValidationError

        missing = {"beta", "gamma"} - set(params)
        if missing:
            raise ValidationError(f"Missing required parameters: {missing}")
        beta = float(params["beta"])
        gamma = float(params["gamma"])
        if beta < 0 or gamma < 0:
            raise ValueError("beta and gamma must be non-negative")
        return beta, gamma

    # -- simulation ---------------------------------------------------------

    def __call__(
        self,
        inits: int | list,
        trange: list[float],
        params: dict[str, Any],
        n_sims: int = 1,
        seed: int | None = None,
        n_points: int = 101,
        validate: bool = True,
    ) -> None:
        """
        Run the network epidemic.

        Args:
            inits: Number of initially infected nodes (chosen as the first
                ``k`` node ids) or an explicit list of infected nodes.
            trange: Simulation time range [t0, tmax].
            params: Dict with ``beta`` (per-edge transmission rate) and
                ``gamma`` (recovery rate).
            n_sims: Number of stochastic replicates.
            seed: Random seed.
            n_points: Output grid size between trange[0] and trange[1].
            validate: Whether to validate parameters.
        """
        beta, gamma = self._validate_params(params)
        if n_sims < 1:
            raise ValueError(f"n_sims must be >= 1, got {n_sims}")
        if n_points < 2:
            raise ValueError(f"n_points must be >= 2, got {n_points}")
        start, end = float(trange[0]), float(trange[1])
        if start >= end:
            raise ValueError("trange[0] must be < trange[1]")

        infected_idx = self._validate_inits(inits)
        grid = np.linspace(start, end, n_points)
        rng = np.random.default_rng(seed)

        n_states = 3 if self.keep_removed else 2
        runs = np.empty((n_sims, n_points, n_states))
        for i in range(n_sims):
            times, counts = self._simulate_one(
                self.adj, infected_idx, beta, gamma, start, end, rng
            )
            runs[i] = self._to_grid(times, counts, grid)

        names = ["S", "I", "R"] if self.keep_removed else ["S", "I"]
        self.traces = {"time": grid}
        for ci, name in enumerate(names):
            self.traces[name] = runs[:, :, ci] if n_sims > 1 else runs[0, :, ci]
        self._n_sims = n_sims

    def _infect_event(
        self,
        adj: list[list[int]],
        state: NDArray[np.int8],
        infected: list[int],
        si_edges: int,
        rng: np.random.Generator,
    ) -> int:
        """
        Transmit along one S-I edge: pick an infected node weighted by its
        susceptible neighbors, infect one of them.

        Returns the updated S-I edge count.
        """
        weights = [sum(1 for nb in adj[j] if state[nb] == 0) for j in infected]
        k = _pick_weighted(rng, weights)
        j = int(infected[k])
        susceptible = [nb for nb in adj[j] if state[nb] == 0]
        target = int(susceptible[rng.integers(len(susceptible))])
        state[target] = 1
        infected.append(target)
        n_s = sum(1 for nb in adj[target] if state[nb] == 0)
        n_i = sum(1 for nb in adj[target] if state[nb] == 1) - 1  # excluding target
        return si_edges + n_s - n_i

    @staticmethod
    def _to_grid(
        times: NDArray[np.floating],
        counts: NDArray[np.floating],
        grid: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Step-function interpolation of counts at event times onto grid."""
        idx = np.searchsorted(times, grid, side="right") - 1
        idx = np.clip(idx, 0, len(times) - 1)
        return counts[idx]

    def _simulate_one(
        self,
        adj: list[list[int]],
        initial_infected: list[int],
        beta: float,
        gamma: float,
        t0: float,
        tmax: float,
        rng: np.random.Generator,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        raise NotImplementedError

    # -- results ------------------------------------------------------------

    def _require_run(self) -> None:
        if not self.traces:
            raise ValueError("No simulation results. Run the model first.")

    def summary(self) -> dict[str, Any]:
        """Summary statistics; for multi-replicate runs, of the ensemble mean."""
        self._require_run()
        if self._n_sims > 1:
            saved = self.traces
            self.traces = self.get_mean()
            try:
                return super().summary()
            finally:
                self.traces = saved
        return super().summary()

    def get_mean(self) -> dict[str, NDArray[np.floating]]:
        """Mean trajectory across replicates."""
        self._require_run()
        out: dict[str, NDArray[np.floating]] = {}
        for name, data in self.traces.items():
            arr = np.asarray(data)
            out[name] = arr.mean(axis=0) if self._n_sims > 1 else arr
        return out

    def get_quantiles(
        self, ci: float = 0.95
    ) -> dict[str, tuple[NDArray[np.floating], NDArray[np.floating]]]:
        """Credible band per variable across replicates."""
        self._require_run()
        lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
        out: dict[str, tuple[NDArray[np.floating], NDArray[np.floating]]] = {}
        for name, data in self.traces.items():
            arr = np.atleast_2d(np.asarray(data))
            out[name] = (np.quantile(arr, lo_q, axis=0), np.quantile(arr, hi_q, axis=0))
        return out

    def final_size(self) -> NDArray[np.floating]:
        """
        Fraction of ever-infected (removed) nodes at the end, per replicate.

        Raises:
            AttributeError: For SIS models (no permanent removal).
        """
        self._require_run()
        if "R" not in self.traces:
            raise AttributeError("final_size is only defined for models with removal (SIR)")
        data = np.atleast_2d(np.asarray(self.traces["R"]))
        return data[:, -1] / self.n_nodes

    def plot_traces(self, var: str = "I", show_mean: bool = True) -> Any:  # type: ignore[override]
        """Plot replicate trajectories (and the mean) for one variable."""
        import matplotlib.pyplot as plt

        self._require_run()
        if var not in self.traces:
            raise KeyError(f"Unknown variable '{var}'. Available: {sorted(self.traces)}")
        time = self.traces["time"]
        data = np.atleast_2d(np.asarray(self.traces[var]))
        if data.shape[0] > 1:
            for i in range(data.shape[0]):
                plt.plot(time, data[i], alpha=0.25, linewidth=0.6, color="gray")
            if show_mean:
                plt.plot(time, data.mean(axis=0), linewidth=2, label=f"{var} (mean)")
        else:
            plt.plot(time, data[0], linewidth=2, label=var)
        plt.xlabel("Time")
        plt.ylabel(f"{var} (count of {self.n_nodes} nodes)")
        plt.legend(loc=0)
        return plt.gca()


class NetworkSIR(NetworkModel):
    """
    Susceptible-Infectious-Removed epidemic on a network.

    Infection fires along each S-I edge at rate ``beta``; each infected
    node recovers (permanently) at rate ``gamma``.
    """

    keep_removed = True

    def __init__(self, graph: Any):
        super().__init__(graph)
        self.state_variables = {"S": "Susceptible", "I": "Infectious", "R": "Removed"}

    def _simulate_one(
        self,
        adj: list[list[int]],
        initial_infected: list[int],
        beta: float,
        gamma: float,
        t0: float,
        tmax: float,
        rng: np.random.Generator,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        state = np.zeros(self.n_nodes, dtype=np.int8)  # 0=S, 1=I, 2=R
        infected: list[int] = []
        for j in initial_infected:
            if state[j] == 0:
                state[j] = 1
                infected.append(j)
        si_edges = sum(
            1
            for j in infected
            for nb in adj[j]
            if state[nb] == 0
        )

        times_list = [t0]
        counts = [[int((state == 0).sum()), len(infected), 0]]
        t = t0

        while infected:
            rate_infect = beta * si_edges
            rate_recover = gamma * len(infected)
            total = rate_infect + rate_recover
            if total <= 0:
                break
            t += rng.exponential(1.0 / total)
            if t > tmax:
                break

            if rng.random() * total < rate_infect:
                # pick an infected node weighted by its susceptible neighbors
                weights = [sum(1 for nb in adj[j] if state[nb] == 0) for j in infected]
                if sum(weights) == 0:
                    # no S-I edges although si_edges > 0 shouldn't happen; guard anyway
                    continue
                si_edges = self._infect_event(adj, state, infected, si_edges, rng)
            else:
                k = rng.integers(len(infected))
                j = int(infected.pop(k))
                state[j] = 2
                si_edges -= sum(1 for nb in adj[j] if state[nb] == 0)

            times_list.append(t)
            counts.append(
                [int((state == 0).sum()), int((state == 1).sum()), int((state == 2).sum())]
            )

        return (
            np.asarray(times_list, dtype=float),
            np.asarray(counts, dtype=float),
        )


class NetworkSIS(NetworkModel):
    """
    Susceptible-Infectious-Susceptible epidemic on a network.

    Infection fires along each S-I edge at rate ``beta``; each infected
    node recovers back to susceptibility at rate ``gamma``.
    """

    keep_removed = False

    def _simulate_one(
        self,
        adj: list[list[int]],
        initial_infected: list[int],
        beta: float,
        gamma: float,
        t0: float,
        tmax: float,
        rng: np.random.Generator,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        state = np.zeros(self.n_nodes, dtype=np.int8)  # 0=S, 1=I
        infected: list[int] = []
        for j in initial_infected:
            if state[j] == 0:
                state[j] = 1
                infected.append(j)
        si_edges = sum(1 for j in infected for nb in adj[j] if state[nb] == 0)

        times_list = [t0]
        counts = [[int((state == 0).sum()), len(infected)]]
        t = t0

        while infected:
            rate_infect = beta * si_edges
            rate_recover = gamma * len(infected)
            total = rate_infect + rate_recover
            if total <= 0:
                break
            t += rng.exponential(1.0 / total)
            if t > tmax:
                break

            if rng.random() * total < rate_infect:
                weights = [sum(1 for nb in adj[j] if state[nb] == 0) for j in infected]
                if sum(weights) == 0:
                    continue
                si_edges = self._infect_event(adj, state, infected, si_edges, rng)
            else:
                k = rng.integers(len(infected))
                j = int(infected.pop(k))
                state[j] = 0
                n_s = sum(1 for nb in adj[j] if state[nb] == 0)
                n_i = sum(1 for nb in adj[j] if state[nb] == 1)
                si_edges += n_i - n_s

            times_list.append(t)
            counts.append([int((state == 0).sum()), int((state == 1).sum())])

        return (
            np.asarray(times_list, dtype=float),
            np.asarray(counts, dtype=float),
        )

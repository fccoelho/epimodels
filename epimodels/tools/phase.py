"""
Tools for phase space representation of epidemic model dynamics.

This module provides tools for analyzing the dynamics of epidemic models
using phase space reconstruction techniques from nonlinear time series analysis.

Main features:
- Time delay embedding for phase space reconstruction
- Mutual information calculation for optimal delay selection
- Embedding dimension estimation using Cao's method

References:
1. Basharat, A., & Shah, M. (2009). Time series prediction by chaotic
   modeling of nonlinear dynamical systems. IEEE 12th International
   Conference on Computer Vision.
2. Cao, L. (1997). Practical method for determining the minimum embedding
   dimension of a scalar time series. Physica D: Nonlinear Phenomena, 110(1-2), 43-50.
3. Takens, F. (1981). Detecting strange attractors in turbulence.
   Lecture Notes in Mathematics, 898, 366-381.
"""

from typing import Optional
import numpy as np
from numpy.typing import NDArray


class TimeDelayEmbedding:
    """
    Time delay embedding for phase space reconstruction.

    Implements Takens' embedding theorem for reconstructing the phase space
    from a scalar time series.

    Parameters:
        data: 1D array of time series data
        tau: Time delay (embedding lag)
        dim: Embedding dimension

    Example:
        >>> import numpy as np
        >>> t = np.linspace(0, 10, 1000)
        >>> data = np.sin(t) + 0.1 * np.random.randn(1000)
        >>> embedding = TimeDelayEmbedding(data, tau=10, dim=3)
        >>> embedded = embedding.embed()
    """

    def __init__(self, data: NDArray[np.floating], tau: int = 1, dim: int = 2):
        self.data = np.asarray(data).flatten()
        self.tau = tau
        self.dim = dim

    def embed(self) -> NDArray[np.floating]:
        """
        Create time delay embedding of the time series.

        Returns:
            2D array of shape (n_samples, dim) where each row is an
            embedded point in the reconstructed phase space.
        """
        n = len(self.data)
        embedded_length = n - (self.dim - 1) * self.tau

        if embedded_length <= 0:
            raise ValueError(
                f"Embedding parameters too large: tau={self.tau}, dim={self.dim} "
                f"for data length {n}"
            )

        embedded = np.zeros((embedded_length, self.dim))
        for i in range(self.dim):
            embedded[:, i] = self.data[i * self.tau : i * self.tau + embedded_length]

        return embedded

    def mutual_information(self, tau_max: int = 100, bins: int = 16) -> tuple[int, list[float]]:
        """
        Calculate average mutual information for different time delays.

        Used to find the optimal embedding delay tau. The first local minimum
        of the mutual information is a good choice for tau.

        Parameters:
            tau_max: Maximum time delay to consider
            bins: Number of bins for histogram estimation

        Returns:
            Tuple of (optimal_tau, list of mutual information values)

        Example:
            >>> embedding = TimeDelayEmbedding(data)
            >>> tau_opt, mi_values = embedding.mutual_information(tau_max=50)
        """
        mi_values = []
        tau_opt = 1

        for tau in range(1, min(tau_max + 1, len(self.data) // 2)):
            mi = self._calculate_mi(tau, bins)
            mi_values.append(mi)

            if len(mi_values) > 1 and mi_values[-2] < mi_values[-1]:
                tau_opt = tau - 1
                break

        return tau_opt, mi_values

    def _calculate_mi(self, tau: int, bins: int) -> float:
        """Calculate mutual information between data and delayed version."""
        x = self.data[:-tau]
        y = self.data[tau:]

        hist_xy, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
        hist_x = np.histogram(x, bins=bins)[0]
        hist_y = np.histogram(y, bins=bins)[0]

        p_xy = hist_xy / np.sum(hist_xy)
        p_x = hist_x / np.sum(hist_x)
        p_y = hist_y / np.sum(hist_y)

        mi = 0.0
        for i in range(bins):
            for j in range(bins):
                if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                    mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))

        return mi

    def cao_embedding_dimension(
        self, dim_max: int = 10, tau: Optional[int] = None, threshold: float = 0.01
    ) -> tuple[int, list[float]]:
        """
        Estimate minimum embedding dimension using Cao's method.

        Cao's method uses the E1 statistic to determine the minimum embedding
        dimension. When E1 saturates (stops increasing), the corresponding
        dimension is sufficient.

        Parameters:
            dim_max: Maximum embedding dimension to test
            tau: Time delay (uses self.tau if None)
            threshold: Threshold for E1 saturation detection

        Returns:
            Tuple of (optimal_dimension, list of E1 values)

        Example:
            >>> embedding = TimeDelayEmbedding(data, tau=10)
            >>> dim_opt, e1_values = embedding.cao_embedding_dimension(dim_max=10)
        """
        if tau is None:
            tau = self.tau

        e1_values = []
        dim_opt = 2

        for dim in range(1, dim_max + 1):
            if dim == 1:
                e1_values.append(1.0)
                continue

            e_d = self._calculate_e_statistic(dim, tau)
            e_d1 = self._calculate_e_statistic(dim - 1, tau)

            if e_d1 > 0:
                e1 = e_d / e_d1
            else:
                e1 = 1.0

            e1_values.append(e1)

            if len(e1_values) > 2:
                if abs(e1_values[-1] - e1_values[-2]) < threshold:
                    dim_opt = dim - 1
                    break

        return dim_opt, e1_values

    def _calculate_e_statistic(self, dim: int, tau: int) -> float:
        """Calculate E statistic for Cao's method."""
        embedded = TimeDelayEmbedding(self.data, tau=tau, dim=dim).embed()
        n = len(embedded)

        if n < 2:
            return 0.0

        a_i = np.zeros(n)
        for i in range(n):
            distances = np.sqrt(np.sum((embedded - embedded[i]) ** 2, axis=1))
            distances[i] = np.inf
            nearest = np.argmin(distances)
            a_i[i] = np.abs(embedded[i, -1] - embedded[nearest, -1])

        return np.mean(a_i)

    def plot_mutual_information(self, tau_max: int = 100, bins: int = 16, ax=None) -> "plt.Axes":
        """
        Plot mutual information vs time delay.

        Parameters:
            tau_max: Maximum time delay to consider
            bins: Number of bins for histogram estimation
            ax: Matplotlib axes object (creates new if None)

        Returns:
            Matplotlib axes object
        """
        import matplotlib.pyplot as plt

        tau_opt, mi_values = self.mutual_information(tau_max, bins)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        taus = range(1, len(mi_values) + 1)
        ax.plot(taus, mi_values, "b-", linewidth=1.5)
        ax.axvline(tau_opt, color="r", linestyle="--", label=f"Optimal τ = {tau_opt}")
        ax.set_xlabel("Time Delay (τ)")
        ax.set_ylabel("Mutual Information")
        ax.set_title("Mutual Information vs Time Delay")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def plot_embedding_dimension(
        self, dim_max: int = 10, tau: Optional[int] = None, ax=None
    ) -> "plt.Axes":
        """
        Plot E1 statistic vs embedding dimension.

        Parameters:
            dim_max: Maximum embedding dimension to test
            tau: Time delay (uses self.tau if None)
            ax: Matplotlib axes object (creates new if None)

        Returns:
            Matplotlib axes object
        """
        import matplotlib.pyplot as plt

        dim_opt, e1_values = self.cao_embedding_dimension(dim_max, tau)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        dims = range(1, len(e1_values) + 1)
        ax.plot(dims, e1_values, "b-o", linewidth=1.5, markersize=6)
        ax.axvline(dim_opt, color="r", linestyle="--", label=f"Optimal dim = {dim_opt}")
        ax.set_xlabel("Embedding Dimension")
        ax.set_ylabel("E1 Statistic")
        ax.set_title("Cao's Method for Embedding Dimension")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax


def phase_portrait(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    ax=None,
    color_by_time: bool = True,
    **kwargs,
) -> "plt.Axes":
    """
    Create a phase portrait from two time series.

    Parameters:
        x: First time series (e.g., Susceptible)
        y: Second time series (e.g., Infectious)
        ax: Matplotlib axes object (creates new if None)
        color_by_time: If True, color trajectory by time
        **kwargs: Additional arguments passed to plot

    Returns:
        Matplotlib axes object

    Example:
        >>> from epimodels.continuous import SIR
        >>> model = SIR()
        >>> model([1000, 1, 0], [0, 100], 1001, {'beta': 0.3, 'gamma': 0.1})
        >>> phase_portrait(model.traces['S'], model.traces['I'])
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    if color_by_time:
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        from matplotlib.collections import LineCollection

        lc = LineCollection(segments, cmap="viridis", **kwargs)
        lc.set_array(np.arange(len(x)))
        ax.add_collection(lc)
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
    else:
        ax.plot(x, y, **kwargs)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Phase Portrait")
    ax.grid(True, alpha=0.3)

    return ax


def find_optimal_embedding(
    data: NDArray[np.floating], tau_max: int = 50, dim_max: int = 10
) -> dict:
    """
    Find optimal embedding parameters for a time series.

    Combines mutual information and Cao's method to find optimal
    time delay and embedding dimension.

    Parameters:
        data: 1D array of time series data
        tau_max: Maximum time delay to consider
        dim_max: Maximum embedding dimension to test

    Returns:
        Dictionary with keys:
        - 'tau': Optimal time delay
        - 'dim': Optimal embedding dimension
        - 'mi_values': List of mutual information values
        - 'e1_values': List of E1 statistics

    Example:
        >>> model = SIR()
        >>> model([1000, 1, 0], [0, 100], 1001, {'beta': 0.3, 'gamma': 0.1})
        >>> params = find_optimal_embedding(model.traces['I'])
        >>> print(f"Optimal tau={params['tau']}, dim={params['dim']}")
    """
    embedding = TimeDelayEmbedding(data)
    tau_opt, mi_values = embedding.mutual_information(tau_max)
    embedding.tau = tau_opt
    dim_opt, e1_values = embedding.cao_embedding_dimension(dim_max)

    return {
        "tau": tau_opt,
        "dim": dim_opt,
        "mi_values": mi_values,
        "e1_values": e1_values,
    }

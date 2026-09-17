"""
Real-time reproduction number (Rt) estimation from incidence data.

Implements the Cori et al. (2013) sliding-window approach with a
gamma-distributed serial interval and gamma prior/posterior on Rt.
This is the method used by EpiEstim.

The estimation is model-free: it only requires an incidence time series
(counts at regular intervals, e.g. daily) and assumptions about the
serial interval distribution.

Example:
    >>> from epimodels.rt import estimate_rt
    >>> result = estimate_rt(incidence, window=7, si_mean=4.0, si_sd=2.0)
    >>> result.rt_mean      # posterior mean Rt per window
    >>> result.rt_low       # 95% credible interval
    >>> result.rt_high
    >>> result.plot()       # quick visualization

Reference:
    Cori, A. et al. (2013). A new framework and software to estimate
    time-varying reproduction numbers during epidemics. American Journal
    of Epidemiology, 178(9), 1505-1512.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

if TYPE_CHECKING:
    from matplotlib.axes import Axes

__all__ = ["RtResult", "estimate_rt"]


def _discretize_si(si_mean: float, si_sd: float, max_si: int = 40) -> NDArray[np.floating]:
    """
    Discretize a gamma serial interval onto integer days.

    w[k] = F(k + 0.5) - F(k - 0.5) for k = 1..S, truncated where the
    cumulative mass reaches 0.99 and renormalized. w[0] = 0 (a case cannot
    infect on the same day under this discretization).

    Returns:
        Array w of length max_si+1 where w[s] is the probability that a
        secondary case occurs s days after the primary case.
    """
    shape = (si_mean / si_sd) ** 2
    scale = si_sd**2 / si_mean
    cdf = stats.gamma.cdf

    w = np.zeros(max_si + 1)
    k = np.arange(1, max_si + 1)
    w[1:] = cdf(k + 0.5, a=shape, scale=scale) - cdf(k - 0.5, a=shape, scale=scale)

    cumulative = np.cumsum(w)
    cutoff = int(np.searchsorted(cumulative, 0.99)) + 1
    w = w[: cutoff + 1]
    return np.asarray(w / w.sum())


def _posterior_params(
    incidence: NDArray[np.floating],
    total_infectivity: NDArray[np.floating],
    a_prior: float,
    b_prior: float,
    start: int,
    end: int,
) -> tuple[float, float]:
    """
    Gamma posterior (shape, rate) for Rt over the window [start, end).

    Prior is Gamma(shape=a_prior, rate=b_prior).
    """
    shape = a_prior + float(np.sum(incidence[start:end]))
    rate = b_prior + float(np.sum(total_infectivity[start:end]))
    return shape, rate


@dataclass
class RtResult:
    """Sliding-window Rt estimates with credible intervals."""

    times: NDArray[np.floating]
    rt_mean: NDArray[np.floating]
    rt_low: NDArray[np.floating]
    rt_high: NDArray[np.floating]
    window: int
    level: float

    def __len__(self) -> int:
        return len(self.rt_mean)

    def plot(
        self,
        ax: Axes | None = None,
        show_ci: bool = True,
        threshold: bool = True,
        **plot_kwargs,
    ) -> Axes:
        """
        Plot Rt over time.

        Args:
            ax: Matplotlib axes (creates a new figure if None).
            show_ci: Whether to draw the credible-interval band.
            threshold: Whether to draw the Rt = 1 reference line.
            **plot_kwargs: Extra kwargs for the mean line.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(9, 5))

        kwargs = dict(plot_kwargs)
        kwargs.setdefault("label", "Rt (posterior mean)")
        kwargs.setdefault("linewidth", 2)
        ax.plot(self.times, self.rt_mean, **kwargs)

        if show_ci:
            ax.fill_between(
                self.times,
                self.rt_low,
                self.rt_high,
                alpha=0.25,
                label=f"{int(self.level * 100)}% credible interval",
            )
        if threshold:
            ax.axhline(1.0, color="k", linestyle="--", linewidth=1, label="Rt = 1")

        ax.set_xlabel("Time")
        ax.set_ylabel("Rt")
        ax.set_title(f"Time-varying reproduction number ({self.window}-day window)")
        ax.legend(loc=0)
        return ax


def estimate_rt(
    incidence: Any,
    window: int = 7,
    si_mean: float = 4.0,
    si_sd: float = 2.0,
    prior_mean: float = 5.0,
    prior_sd: float = 5.0,
    level: float = 0.95,
    times: Any = None,
) -> RtResult:
    """
    Estimate the time-varying reproduction number Rt from incidence data.

    Uses the Cori et al. (2013) approach: over each sliding window of
    ``window`` time steps, the posterior of Rt is
    ``Gamma(a + sum(I), rate_prior + sum(Lambda))`` where Lambda(t) is the
    total infectiousness given the discretized gamma serial interval.

    Args:
        incidence: Incidence counts at regular intervals (list or array).
            Assumes ordering by increasing time and unit spacing.
        window: Window length in time steps (default 7). Rt is reported
            at the last time step of each window.
        si_mean: Mean of the gamma serial interval, in time steps.
        si_sd: Standard deviation of the serial interval, in time steps.
        prior_mean: Mean of the gamma prior on Rt.
        prior_sd: Standard deviation of the gamma prior on Rt.
        level: Credible interval level (default 0.95).
        times: Optional time values aligned with ``incidence`` (default
            0..n-1). Used only for reporting.

    Returns:
        RtResult with one estimate per window position.
    """
    incid = np.asarray(incidence, dtype=float).ravel()
    if incid.size < 2:
        raise ValueError("incidence must have at least 2 time steps")
    if not np.all(np.isfinite(incid)) or np.any(incid < 0):
        raise ValueError("incidence must be finite and non-negative")
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if window > incid.size:
        raise ValueError(
            f"window ({window}) cannot exceed the number of time steps ({incid.size})"
        )
    if si_mean <= 0 or si_sd <= 0:
        raise ValueError("si_mean and si_sd must be positive")
    if prior_mean <= 0 or prior_sd <= 0:
        raise ValueError("prior_mean and prior_sd must be positive")
    if not 0 < level < 1:
        raise ValueError(f"level must be in (0, 1), got {level}")

    # Gamma prior on Rt: shape a, rate b
    a_prior = (prior_mean / prior_sd) ** 2
    b_prior = prior_mean / prior_sd**2

    # Total infectiousness Lambda(t) = sum_s w_s * I(t-s)
    w = _discretize_si(si_mean, si_sd)
    n = incid.size
    total_infectivity = np.zeros(n)
    max_si = len(w) - 1
    for s in range(1, min(max_si, n - 1) + 1):
        total_infectivity[s:] += w[s] * incid[: n - s]

    n_windows = n - window + 1
    rt_mean = np.full(n_windows, np.nan)
    rt_low = np.full(n_windows, np.nan)
    rt_high = np.full(n_windows, np.nan)

    alpha = (1 - level) / 2
    for i in range(n_windows):
        start = i
        end = i + window
        shape, rate = _posterior_params(incid, total_infectivity, a_prior, b_prior, start, end)
        if rate <= 0:
            continue  # no infectivity in window: Rt undefined
        rt_mean[i] = shape / rate
        rt_low[i] = stats.gamma.ppf(alpha, a=shape, scale=1.0 / rate)
        rt_high[i] = stats.gamma.ppf(1 - alpha, a=shape, scale=1.0 / rate)

    if times is None:
        out_times = np.arange(n, dtype=float)
    else:
        out_times = np.asarray(times, dtype=float).ravel()
        if out_times.size != n:
            raise ValueError(
                f"times has {out_times.size} entries but incidence has {n}"
            )

    return RtResult(
        times=out_times[window - 1 :],
        rt_mean=rt_mean,
        rt_low=rt_low,
        rt_high=rt_high,
        window=window,
        level=level,
    )

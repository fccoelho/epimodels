"""Tests for Rt estimation (Cori et al. 2013)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from epimodels.continuous import SIR
from epimodels.rt import RtResult, estimate_rt


@pytest.fixture
def sir_incidence():
    """Daily incidence from a simulated SIR epidemic with known R0=3."""
    model = SIR()
    t_eval = np.arange(0, 60)
    model([989, 10, 0], [0, 60], 1000, {"beta": 0.9, "gamma": 0.3}, t_eval=t_eval)
    infectious = model.traces["I"]
    new_cases = -np.diff(model.traces["S"])  # incident infections per day
    return np.maximum(new_cases, 0), infectious


class TestEstimateRt:
    def test_returns_result_with_shapes(self, sir_incidence):
        incidence, _ = sir_incidence
        result = estimate_rt(incidence, window=7)
        assert isinstance(result, RtResult)
        n_windows = len(incidence) - 7 + 1
        assert len(result) == n_windows
        assert result.rt_mean.shape == (n_windows,)
        assert result.times.shape == (n_windows,)
        assert result.times[-1] == pytest.approx(len(incidence) - 1)

    def test_rt_starts_near_r0_in_growth_phase(self, sir_incidence):
        """Early-window Rt should be close to beta/gamma = 3 for SIR."""
        incidence, _ = sir_incidence
        result = estimate_rt(incidence, window=5, si_mean=3.0, si_sd=1.5)
        early = result.rt_mean[:5]
        # In the exponential growth phase with depleted-susceptible
        # correction, Rt should be in a plausible band below R0=3
        assert np.nanmean(early) == pytest.approx(3.0, abs=1.0)

    def test_rt_declines_after_depletion(self, sir_incidence):
        incidence, _ = sir_incidence
        result = estimate_rt(incidence, window=7)
        # Windows during the growth phase vs after the peak: Rt must fall
        # below 1 as susceptibles deplete
        assert np.nanmean(result.rt_mean[:3]) > 2.0
        assert np.nanmean(result.rt_mean[7:12]) < 1.0

    def test_constant_incidence_gives_rt_near_one(self):
        """Constant incidence with fixed SI implies Rt ~ 1 in expectation."""
        incidence = np.full(60, 50.0)
        result = estimate_rt(incidence, window=14, si_mean=4.0, si_sd=2.0)
        # Once Lambda accumulates (after SI memory), Rt ≈ 1
        steady = result.rt_mean[14:]
        assert np.nanmean(steady) == pytest.approx(1.0, abs=0.05)

    def test_credible_intervals_bracket_mean(self, sir_incidence):
        incidence, _ = sir_incidence
        result = estimate_rt(incidence, window=7)
        valid = ~np.isnan(result.rt_mean)
        assert np.all(result.rt_low[valid] <= result.rt_mean[valid])
        assert np.all(result.rt_mean[valid] <= result.rt_high[valid])

    def test_reproducibility_is_deterministic(self, sir_incidence):
        incidence, _ = sir_incidence
        r1 = estimate_rt(incidence, window=7)
        r2 = estimate_rt(incidence, window=7)
        np.testing.assert_array_equal(r1.rt_mean, r2.rt_mean)

    def test_custom_times(self, sir_incidence):
        incidence, _ = sir_incidence
        days = np.arange(100, 100 + len(incidence))
        result = estimate_rt(incidence, window=7, times=days)
        assert result.times[0] == 100 + 6

    def test_prior_dominated_when_no_cases(self):
        """All-zero leading incidence: early windows fall back to the prior
        (same behavior as EpiEstim), then Rt settles near 1 at equilibrium."""
        incidence = np.concatenate([np.zeros(10), np.full(20, 10.0)])
        result = estimate_rt(incidence, window=3, si_mean=2.0, si_sd=1.0)
        assert result.rt_mean[0] == pytest.approx(5.0)  # prior mean
        assert np.isfinite(result.rt_mean[-1])

    def test_invalid_inputs(self, sir_incidence):
        incidence, _ = sir_incidence
        with pytest.raises(ValueError, match="window"):
            estimate_rt(incidence, window=0)
        with pytest.raises(ValueError, match="window"):
            estimate_rt(incidence, window=len(incidence) + 1)
        with pytest.raises(ValueError, match="si_mean"):
            estimate_rt(incidence, si_mean=-1)
        with pytest.raises(ValueError, match="level"):
            estimate_rt(incidence, level=1.5)
        with pytest.raises(ValueError, match="finite and non-negative"):
            estimate_rt([-1, 2, 3], window=2)
        with pytest.raises(ValueError, match="at least 2"):
            estimate_rt([1], window=1)
        with pytest.raises(ValueError, match="times"):
            estimate_rt(incidence, window=3, times=[1, 2, 3])

    def test_plot(self, sir_incidence):
        incidence, _ = sir_incidence
        result = estimate_rt(incidence, window=7)
        ax = result.plot()
        assert ax is not None
        plt.close("all")

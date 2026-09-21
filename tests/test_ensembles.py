"""Tests for ensemble simulation."""

import numpy as np
import pytest

from epimodels.continuous import SIR
from epimodels.ensembles import TraceEnsemble, simulate_ensemble


@pytest.fixture
def sir_model():
    return SIR()


def sampler_factory(seed=0):
    rng = np.random.default_rng(seed)

    def sample_params():
        return {"beta": float(rng.uniform(1.0, 3.0)), "gamma": 0.1}

    return sample_params


class TestSimulateEnsemble:
    def test_shapes(self, sir_model):
        ens = simulate_ensemble(
            sir_model,
            n_sims=5,
            param_sampler=sampler_factory(),
            initial_conditions=[1000, 1, 0],
            trange=[0, 50],
            totpop=1001,
            n_points=50,
        )
        assert isinstance(ens, TraceEnsemble)
        assert len(ens) == 5
        assert ens.traces["I"].shape == (5, 50)
        assert ens.time[0] == pytest.approx(0)
        assert ens.time[-1] == pytest.approx(50)

    def test_traces_differ_across_realizations(self, sir_model):
        ens = simulate_ensemble(
            sir_model,
            n_sims=4,
            param_sampler=sampler_factory(),
            initial_conditions=[1000, 1, 0],
            trange=[0, 50],
            totpop=1001,
        )
        peaks = ens.traces["I"].max(axis=1)
        assert np.ptp(peaks) > 0, "sampled params produced identical trajectories"

    def test_callable_initial_conditions(self, sir_model):
        ens = simulate_ensemble(
            sir_model,
            n_sims=3,
            param_sampler=lambda: {"beta": 2.0, "gamma": 0.5},
            initial_conditions=lambda: [1000, 1, 0],
            trange=[0, 30],
            totpop=1001,
        )
        assert len(ens) == 3

    def test_invalid_n_sims(self, sir_model):
        with pytest.raises(ValueError, match="n_sims"):
            simulate_ensemble(
                sir_model,
                n_sims=0,
                param_sampler=sampler_factory(),
                initial_conditions=[1000, 1, 0],
                trange=[0, 10],
                totpop=1001,
            )

    def test_invalid_model(self):
        with pytest.raises(TypeError, match="BaseModel"):
            simulate_ensemble(
                "not a model",
                n_sims=2,
                param_sampler=sampler_factory(),
                initial_conditions=[1000, 1, 0],
                trange=[0, 10],
                totpop=1001,
            )

    def test_original_model_untouched(self, sir_model):
        simulate_ensemble(
            sir_model,
            n_sims=2,
            param_sampler=lambda: {"beta": 2.0, "gamma": 0.1},
            initial_conditions=[1000, 1, 0],
            trange=[0, 20],
            totpop=1001,
        )
        assert sir_model.traces == {}, "original model should not hold traces"


class TestTraceEnsemble:
    def test_quantiles(self, sir_model):
        ens = simulate_ensemble(
            sir_model,
            n_sims=10,
            param_sampler=sampler_factory(),
            initial_conditions=[1000, 1, 0],
            trange=[0, 50],
            totpop=1001,
        )
        quant = ens.quantiles([0.025, 0.5, 0.975])
        assert set(quant.keys()) == {"S", "I", "R"}
        for var in quant:
            assert set(quant[var].keys()) == {0.025, 0.5, 0.975}
            lower = quant[var][0.025]
            upper = quant[var][0.975]
            assert np.all(lower <= upper + 1e-9)

    def test_summary(self, sir_model):
        ens = simulate_ensemble(
            sir_model,
            n_sims=5,
            param_sampler=sampler_factory(),
            initial_conditions=[1000, 1, 0],
            trange=[0, 50],
            totpop=1001,
        )
        summary = ens.summary()
        assert "peak_mean" in summary["I"]
        assert summary["I"]["peak_mean"] > 0

    def test_plot_band(self, sir_model):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        ens = simulate_ensemble(
            sir_model,
            n_sims=5,
            param_sampler=sampler_factory(),
            initial_conditions=[1000, 1, 0],
            trange=[0, 50],
            totpop=1001,
        )
        ax = ens.plot_band("I")
        assert ax is not None

    def test_plot_band_unknown_var(self, sir_model):
        ens = simulate_ensemble(
            sir_model,
            n_sims=2,
            param_sampler=sampler_factory(),
            initial_conditions=[1000, 1, 0],
            trange=[0, 20],
            totpop=1001,
        )
        with pytest.raises(KeyError):
            ens.plot_band("Z")

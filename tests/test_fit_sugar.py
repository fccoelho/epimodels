"""Tests for the BaseModel.simulate()/fit() sugar API."""

import numpy as np
import pytest

from epimodels.continuous import SIR
from epimodels.fitting import Dataset, ParameterSpec


@pytest.fixture
def synthetic():
    """SIR epidemic with known parameters and sparse observations."""
    t_eval = np.linspace(0, 20, 41)
    m = SIR()
    m([999, 1, 0], [0, 20], 1000, {"beta": 2.0, "gamma": 0.5}, t_eval=t_eval)
    obs_times = t_eval[::2]
    obs_I = m.traces["I"][::2]
    return obs_times, obs_I


class TestSimulate:
    def test_returns_self_for_chaining(self):
        m = SIR()
        result = m.simulate([1000, 1, 0], [0, 50], 1001, {"beta": 2, "gamma": 0.1})
        assert result is m
        assert "I" in m.traces

    def test_forwards_kwargs(self):
        m = SIR()
        t_eval = np.linspace(0, 50, 26)
        m.simulate(
            [1000, 1, 0], [0, 50], 1001, {"beta": 2, "gamma": 0.1}, t_eval=t_eval
        )
        np.testing.assert_allclose(m.traces["time"], t_eval)


class TestFitMLE:
    def test_recovers_parameters(self, synthetic):
        obs_times, obs_I = synthetic
        m = SIR()
        result = m.fit(
            {"I": obs_I},
            times=obs_times,
            params_to_fit={"beta": (0.1, 5.0), "gamma": (0.05, 2.0)},
            total_population=1000,
        )
        assert result.best_params["beta"] == pytest.approx(2.0, rel=0.2)
        assert result.best_params["gamma"] == pytest.approx(0.5, rel=0.2)
        assert result.fitted_model is not None

    def test_variable_mapping(self, synthetic):
        obs_times, obs_I = synthetic
        m = SIR()
        result = m.fit(
            {"cases": obs_I},
            times=obs_times,
            params_to_fit={"beta": (0.1, 5.0), "gamma": (0.05, 2.0)},
            total_population=1000,
            variable_mapping={"cases": "I"},
        )
        assert result.best_params["beta"] == pytest.approx(2.0, rel=0.25)


class TestFitBayes:
    def test_dict_data(self, synthetic):
        obs_times, obs_I = synthetic
        m = SIR()
        result = m.fit(
            {"I": obs_I},
            times=obs_times,
            params_to_fit={"beta": (0.5, 4.0), "gamma": (0.1, 1.5)},
            total_population=1000,
            method="bayes",
            likelihood="normal",
            sigma=5.0,
            num_samples=80,
            num_warmup=50,
            seed=0,
        )
        posterior_beta = float(np.mean(result.samples["beta"]))
        assert posterior_beta == pytest.approx(2.0, rel=0.3)

    def test_dataset_passthrough(self, synthetic):
        obs_times, obs_I = synthetic
        m = SIR()
        dataset = Dataset(m).register(
            name="I", values=obs_I, times=obs_times, state_variable="I"
        )
        result = m.fit(
            dataset,
            [ParameterSpec("beta", bounds=(0.5, 4.0)), ParameterSpec("gamma", bounds=(0.1, 1.5))],
            total_population=1000,
            method="bayes",
            sigma=5.0,
            num_samples=50,
            num_warmup=30,
            seed=1,
        )
        assert result.n_draws == 50 * 8  # default pop_size = max(2*n_params, 8)


class TestFitValidation:
    def test_unknown_method(self, synthetic):
        obs_times, obs_I = synthetic
        m = SIR()
        with pytest.raises(ValueError, match="method"):
            m.fit(
                {"I": obs_I},
                times=obs_times,
                params_to_fit={"beta": (0.1, 5.0)},
                total_population=1000,
                method="abc",
            )

    def test_missing_population(self, synthetic):
        obs_times, obs_I = synthetic
        m = SIR()
        with pytest.raises(ValueError, match="total_population"):
            m.fit({"I": obs_I}, times=obs_times, params_to_fit={"beta": (0.1, 5.0)})

    def test_bayes_requires_times_for_dict(self, synthetic):
        obs_times, obs_I = synthetic
        m = SIR()
        with pytest.raises(ValueError, match="times"):
            m.fit(
                {"I": obs_I},
                params_to_fit={"beta": (0.1, 5.0)},
                total_population=1000,
                method="bayes",
            )

    def test_bayes_dataset_requires_specs(self, synthetic):
        obs_times, obs_I = synthetic
        m = SIR()
        dataset = Dataset(m).register(
            name="I", values=obs_I, times=obs_times, state_variable="I"
        )
        with pytest.raises(ValueError, match="ParameterSpec"):
            m.fit(
                dataset,
                {"beta": (0.1, 5.0)},
                total_population=1000,
                method="bayes",
            )

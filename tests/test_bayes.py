"""Tests for Bayesian inference (DE-MCMC)."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from epimodels.continuous import SIR
from epimodels.fitting import Dataset, ParameterSpec
from epimodels.fitting.bayes import BayesianFitResult, fit_model_bayesian


@pytest.fixture
def synthetic_data():
    """Generate SIR trajectory data with known parameters."""
    model = SIR()
    true_params = {"beta": 2.0, "gamma": 0.5}
    t_eval = np.linspace(0, 20, 21)
    model([999, 1, 0], [0, 20], 1000, true_params, t_eval=t_eval)

    dataset = Dataset(model=model)
    dataset.register(
        name="infectious",
        values=model.traces["I"].copy(),
        times=t_eval,
        state_variable="I",
    )
    return dataset, true_params


class TestFitModelBayesian:
    def test_recovers_parameters(self, synthetic_data):
        """Posterior mean should be close to true parameters."""
        dataset, true_params = synthetic_data
        result = fit_model_bayesian(
            SIR(),
            dataset,
            parameters_to_fit=[
                ParameterSpec("beta", bounds=(0.1, 5.0)),
                ParameterSpec("gamma", bounds=(0.05, 2.0)),
            ],
            total_population=1000,
            likelihood="normal",
            sigma=5.0,
            num_samples=150,
            num_warmup=100,
            seed=42,
        )
        assert isinstance(result, BayesianFitResult)
        posterior_mean = {
            name: float(np.mean(values)) for name, values in result.samples.items()
        }
        assert posterior_mean["beta"] == pytest.approx(true_params["beta"], rel=0.25)
        assert posterior_mean["gamma"] == pytest.approx(true_params["gamma"], rel=0.25)

    def test_sample_shapes_and_ordering(self, synthetic_data):
        dataset, _ = synthetic_data
        result = fit_model_bayesian(
            SIR(),
            dataset,
            parameters_to_fit=[
                ParameterSpec("beta", bounds=(0.1, 5.0)),
                ParameterSpec("gamma", bounds=(0.05, 2.0)),
            ],
            total_population=1000,
            num_samples=50,
            num_warmup=20,
            pop_size=6,
            seed=0,
        )
        assert result.n_draws == 50 * 6
        for values in result.samples.values():
            assert values.shape == (300,)
        assert result.log_probs.shape == (300,)

    def test_samples_within_bounds(self, synthetic_data):
        dataset, _ = synthetic_data
        result = fit_model_bayesian(
            SIR(),
            dataset,
            parameters_to_fit=[
                ParameterSpec("beta", bounds=(1.5, 2.5)),
                ParameterSpec("gamma", bounds=(0.3, 0.8)),
            ],
            total_population=1000,
            num_samples=30,
            num_warmup=15,
            seed=1,
        )
        for name, (low, high) in [("beta", (1.5, 2.5)), ("gamma", (0.3, 0.8))]:
            assert result.samples[name].min() >= low
            assert result.samples[name].max() <= high

    def test_map_and_intervals(self, synthetic_data):
        dataset, true_params = synthetic_data
        result = fit_model_bayesian(
            SIR(),
            dataset,
            parameters_to_fit=[
                ParameterSpec("beta", bounds=(0.1, 5.0)),
                ParameterSpec("gamma", bounds=(0.05, 2.0)),
            ],
            total_population=1000,
            num_samples=40,
            num_warmup=20,
            seed=3,
        )
        map_est = result.map_estimate()
        assert set(map_est) == {"beta", "gamma"}
        cis = result.credible_intervals()
        lo, hi = cis["beta"]
        assert lo <= map_est["beta"] <= hi
        summary = result.summary()
        assert summary["beta"]["q025"] <= summary["beta"]["median"] <= summary["beta"]["q975"]

    def test_poisson_likelihood_runs(self, synthetic_data):
        dataset, _ = synthetic_data
        result = fit_model_bayesian(
            SIR(),
            dataset,
            parameters_to_fit=[
                ParameterSpec("beta", bounds=(0.5, 4.0)),
                ParameterSpec("gamma", bounds=(0.1, 1.5)),
            ],
            total_population=1000,
            likelihood="poisson",
            num_samples=30,
            num_warmup=15,
            seed=5,
        )
        assert np.isfinite(result.log_probs).all()

    def test_attach_fitted_model(self, synthetic_data):
        dataset, _ = synthetic_data
        result = fit_model_bayesian(
            SIR(),
            dataset,
            parameters_to_fit=[
                ParameterSpec("beta", bounds=(0.1, 5.0)),
                ParameterSpec("gamma", bounds=(0.05, 2.0)),
            ],
            total_population=1000,
            num_samples=30,
            num_warmup=15,
            seed=7,
            attach_fitted_model=True,
        )
        assert result.model is not None
        assert "I" in result.model.traces

    def test_reproducible_with_seed(self, synthetic_data):
        dataset, _ = synthetic_data
        results = []
        for _ in range(2):
            results.append(
                fit_model_bayesian(
                    SIR(),
                    dataset,
                    parameters_to_fit=[
                        ParameterSpec("beta", bounds=(0.1, 5.0)),
                        ParameterSpec("gamma", bounds=(0.05, 2.0)),
                    ],
                    total_population=1000,
                    num_samples=30,
                    num_warmup=15,
                    seed=123,
                )
            )
        np.testing.assert_allclose(results[0].samples["beta"], results[1].samples["beta"])

    def test_invalid_args(self, synthetic_data):
        dataset, _ = synthetic_data
        specs = [ParameterSpec("beta", bounds=(0.1, 5.0))]
        with pytest.raises(ValueError, match="num_samples"):
            fit_model_bayesian(
                SIR(), dataset, specs, 1000, num_samples=0
            )
        with pytest.raises(ValueError, match="empty"):
            fit_model_bayesian(SIR(), dataset, [], 1000)
        with pytest.raises(ValueError, match="likelihood"):
            fit_model_bayesian(
                SIR(), dataset, specs, 1000, likelihood="bernoulli", num_samples=10
            )

    def test_trace_plot(self, synthetic_data):
        dataset, _ = synthetic_data
        result = fit_model_bayesian(
            SIR(),
            dataset,
            parameters_to_fit=[
                ParameterSpec("beta", bounds=(0.1, 5.0)),
                ParameterSpec("gamma", bounds=(0.05, 2.0)),
            ],
            total_population=1000,
            num_samples=20,
            num_warmup=10,
            seed=9,
        )
        ax = result.trace_plot()
        assert ax is not None
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_to_inference_data_without_arviz(self, synthetic_data):
        pytest.importorskip("arviz")
        dataset, _ = synthetic_data
        result = fit_model_bayesian(
            SIR(),
            dataset,
            parameters_to_fit=[
                ParameterSpec("beta", bounds=(0.1, 5.0)),
                ParameterSpec("gamma", bounds=(0.05, 2.0)),
            ],
            total_population=1000,
            num_samples=20,
            num_warmup=10,
            seed=11,
        )
        idata = result.to_inference_data()
        assert idata is not None

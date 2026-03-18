"""
Tests for the fitting module.
"""

import pytest
import numpy as np

from epimodels.continuous.models import SIR
from epimodels.fitting import (
    Dataset,
    DataSeries,
    ValidationResult,
    ParameterSpec,
    FittingResult,
    ModelFitter,
    fit_model,
    SumOfSquaredErrors,
    WeightedSSE,
    PoissonLikelihood,
    NegativeBinomialLikelihood,
    NormalLikelihood,
    HuberLoss,
    ScipyOptimizer,
    MultiStartOptimizer,
    FittingError,
    DataValidationError,
)
from epimodels.fitting.utils import (
    interpolate_to_times,
    convert_time_unit,
    get_conversion_factor,
    rescale_parameter,
    unscale_parameter,
    ensure_monotonic,
    find_time_overlap,
)


class TestDataSeries:
    """Tests for DataSeries class."""

    def test_create_series(self):
        """Test creating a data series."""
        times = np.array([0, 1, 2, 3, 4])
        values = np.array([10, 20, 30, 40, 50])

        series = DataSeries(
            name="test",
            values=values,
            times=times,
            state_variable="I",
        )

        assert series.name == "test"
        assert len(series.values) == 5
        assert series.time_range == (0, 4)

    def test_validate_series(self):
        """Test series validation."""
        times = np.array([0, 1, 2, 3, 4])
        values = np.array([10, 20, 30, 40, 50])

        series = DataSeries(
            name="test",
            values=values,
            times=times,
            state_variable="I",
        )

        errors = series.validate()
        assert len(errors) == 0

    def test_validate_series_length_mismatch(self):
        """Test validation with length mismatch."""
        times = np.array([0, 1, 2])
        values = np.array([10, 20, 30, 40])

        series = DataSeries(
            name="test",
            values=values,
            times=times,
            state_variable="I",
        )

        errors = series.validate()
        assert len(errors) == 1
        assert "different lengths" in errors[0]

    def test_validate_series_non_monotonic_times(self):
        """Test validation with non-monotonic times."""
        times = np.array([0, 2, 1, 3, 4])
        values = np.array([10, 20, 30, 40, 50])

        series = DataSeries(
            name="test",
            values=values,
            times=times,
            state_variable="I",
        )

        errors = series.validate()
        assert len(errors) == 1
        assert "monotonically" in errors[0]

    def test_validate_series_nan_values(self):
        """Test validation with NaN values."""
        times = np.array([0, 1, 2, 3, 4])
        values = np.array([10, np.nan, 30, 40, 50])

        series = DataSeries(
            name="test",
            values=values,
            times=times,
            state_variable="I",
        )

        errors = series.validate()
        assert len(errors) == 1
        assert "NaN" in errors[0]

    def test_to_dict(self):
        """Test serialization to dict."""
        times = np.array([0, 1, 2])
        values = np.array([10, 20, 30])

        series = DataSeries(
            name="test",
            values=values,
            times=times,
            state_variable="I",
            time_unit="days",
        )

        d = series.to_dict()
        assert d["name"] == "test"
        assert d["state_variable"] == "I"
        assert d["time_unit"] == "days"


class TestDataset:
    """Tests for Dataset class."""

    def test_create_dataset(self):
        """Test creating a dataset."""
        model = SIR()
        dataset = Dataset(model)

        assert dataset.model is model
        assert len(dataset.series) == 0

    def test_register_series(self):
        """Test registering a data series."""
        model = SIR()
        times = np.array([0, 1, 2, 3, 4])
        values = np.array([10, 20, 30, 40, 50])

        dataset = Dataset(model).register(
            name="cases",
            values=values,
            times=times,
            state_variable="I",
        )

        assert len(dataset.series) == 1
        assert "cases" in dataset.series
        assert dataset.time_range == (0, 4)

    def test_register_multiple_series(self):
        """Test registering multiple series."""
        model = SIR()
        times = np.array([0, 1, 2, 3, 4])
        values_I = np.array([10, 20, 30, 40, 50])
        values_R = np.array([0, 5, 10, 15, 20])

        dataset = (
            Dataset(model)
            .register(name="cases", values=values_I, times=times, state_variable="I")
            .register(name="recovered", values=values_R, times=times, state_variable="R")
        )

        assert len(dataset.series) == 2
        assert dataset.get_series_for_variable("I") is not None
        assert dataset.get_series_for_variable("R") is not None

    def test_validate_dataset(self):
        """Test dataset validation."""
        model = SIR()
        times = np.array([0, 1, 2, 3, 4])
        values = np.array([10, 20, 30, 40, 50])

        dataset = Dataset(model).register(
            name="cases", values=values, times=times, state_variable="I"
        )

        result = dataset.validate()

        assert result.is_valid
        assert len(result.errors) == 0

    def test_validate_invalid_state_variable(self):
        """Test validation with invalid state variable."""
        model = SIR()
        times = np.array([0, 1, 2, 3, 4])
        values = np.array([10, 20, 30, 40, 50])

        dataset = Dataset(model).register(
            name="cases", values=values, times=times, state_variable="X"
        )

        result = dataset.validate()

        assert not result.is_valid
        assert any("not found in model" in e for e in result.errors)

    def test_unregister_series(self):
        """Test unregistering a series."""
        model = SIR()
        times = np.array([0, 1, 2, 3, 4])
        values = np.array([10, 20, 30, 40, 50])

        dataset = Dataset(model).register(
            name="cases", values=values, times=times, state_variable="I"
        )

        assert len(dataset.series) == 1

        dataset.unregister("cases")
        assert len(dataset.series) == 0

    def test_to_dict(self):
        """Test dataset serialization."""
        model = SIR()
        times = np.array([0, 1, 2])
        values = np.array([10, 20, 30])

        dataset = Dataset(model).register(
            name="cases", values=values, times=times, state_variable="I"
        )

        d = dataset.to_dict()
        assert "series" in d
        assert "time_unit" in d


class TestParameterSpec:
    """Tests for ParameterSpec class."""

    def test_create_spec(self):
        """Test creating a parameter spec."""
        spec = ParameterSpec(name="beta", bounds=(0.1, 1.0))

        assert spec.name == "beta"
        assert spec.bounds == (0.1, 1.0)
        assert spec.initial is None
        assert not spec.log_scale

    def test_spec_with_initial(self):
        """Test spec with initial value."""
        spec = ParameterSpec(name="beta", bounds=(0.1, 1.0), initial=0.5)

        assert spec.initial == 0.5

    def test_spec_invalid_bounds(self):
        """Test that invalid bounds raise error."""
        with pytest.raises(ValueError):
            ParameterSpec(name="beta", bounds=(1.0, 0.1))

    def test_spec_initial_outside_bounds(self):
        """Test that initial outside bounds raises error."""
        with pytest.raises(ValueError):
            ParameterSpec(name="beta", bounds=(0.1, 1.0), initial=2.0)

    def test_log_scale_negative_bounds(self):
        """Test that log-scale with negative bounds raises error."""
        with pytest.raises(ValueError):
            ParameterSpec(name="beta", bounds=(-1.0, 1.0), log_scale=True)

    def test_scale_unscale(self):
        """Test parameter scaling and unscaling."""
        spec = ParameterSpec(name="beta", bounds=(0.1, 1.0), initial=0.5)

        scaled = rescale_parameter(0.5, spec.bounds, spec.log_scale)
        assert 0.0 <= scaled <= 1.0

        unscaled = unscale_parameter(scaled, spec.bounds, spec.log_scale)
        assert abs(unscaled - 0.5) < 1e-10

    def test_scale_unscale_log(self):
        """Test log-scale parameter transformation."""
        spec = ParameterSpec(name="beta", bounds=(0.1, 1.0), initial=0.316, log_scale=True)

        scaled = rescale_parameter(0.316, spec.bounds, spec.log_scale)
        assert 0.0 <= scaled <= 1.0

        unscaled = unscale_parameter(scaled, spec.bounds, spec.log_scale)
        assert abs(unscaled - 0.316) < 1e-6


class TestLossFunctions:
    """Tests for loss functions."""

    def test_sse(self):
        """Test sum of squared errors."""
        loss_fn = SumOfSquaredErrors()

        observed = {"I": np.array([10, 20, 30])}
        predicted = {"I": np.array([12, 18, 32])}

        result = loss_fn.compute(observed, predicted)

        expected = (10 - 12) ** 2 + (20 - 18) ** 2 + (30 - 32) ** 2
        assert abs(result.value - expected) < 1e-10

    def test_sse_normalized(self):
        """Test normalized SSE."""
        loss_fn = SumOfSquaredErrors(normalize=True)

        observed = {"I": np.array([10, 20, 30])}
        predicted = {"I": np.array([12, 18, 32])}

        result = loss_fn.compute(observed, predicted)

        expected = ((10 - 12) ** 2 + (20 - 18) ** 2 + (30 - 32) ** 2) / 3
        assert abs(result.value - expected) < 1e-10

    def test_weighted_sse(self):
        """Test weighted SSE."""
        loss_fn = WeightedSSE(variable_weights={"I": 2.0})

        observed = {"I": np.array([10, 20, 30])}
        predicted = {"I": np.array([12, 18, 32])}

        result = loss_fn.compute(observed, predicted)

        expected = 2.0 * ((10 - 12) ** 2 + (20 - 18) ** 2 + (30 - 32) ** 2)
        assert abs(result.value - expected) < 1e-10

    def test_poisson_likelihood(self):
        """Test Poisson likelihood."""
        loss_fn = PoissonLikelihood()

        observed = {"I": np.array([10, 20, 30])}
        predicted = {"I": np.array([12, 18, 32])}

        result = loss_fn.compute(observed, predicted)

        assert result.value > 0
        assert result.per_variable is not None
        assert "I" in result.per_variable

    def test_negative_binomial_likelihood(self):
        """Test negative binomial likelihood."""
        loss_fn = NegativeBinomialLikelihood(dispersion=2.0)

        observed = {"I": np.array([10, 20, 30])}
        predicted = {"I": np.array([12, 18, 32])}

        result = loss_fn.compute(observed, predicted)

        assert result.value > 0

    def test_normal_likelihood(self):
        """Test normal likelihood."""
        loss_fn = NormalLikelihood(sigma=2.0)

        observed = {"I": np.array([10, 20, 30])}
        predicted = {"I": np.array([12, 18, 32])}

        result = loss_fn.compute(observed, predicted)

        assert result.value > 0

    def test_huber_loss(self):
        """Test Huber loss."""
        loss_fn = HuberLoss(delta=1.0)

        observed = {"I": np.array([10, 20, 30])}
        predicted = {"I": np.array([12, 18, 32])}

        result = loss_fn.compute(observed, predicted)

        assert result.value > 0


class TestOptimizers:
    """Tests for optimizers."""

    def test_scipy_optimizer(self):
        """Test scipy optimizer."""
        optimizer = ScipyOptimizer(method="L-BFGS-B", max_iterations=100)

        def objective(x):
            return float((x[0] - 1.0) ** 2 + (x[1] - 2.0) ** 2)

        initial = np.array([0.0, 0.0])
        bounds = [(-5.0, 5.0), (-5.0, 5.0)]

        result = optimizer.minimize(objective, initial, bounds)

        assert result.success
        assert abs(result.best_params[0] - 1.0) < 0.1
        assert abs(result.best_params[1] - 2.0) < 0.1

    def test_scipy_optimizer_differential_evolution(self):
        """Test scipy differential evolution."""
        optimizer = ScipyOptimizer(method="differential_evolution", max_iterations=100)

        def objective(x):
            return float((x[0] - 1.0) ** 2 + (x[1] - 2.0) ** 2)

        initial = np.array([0.0, 0.0])
        bounds = [(-5.0, 5.0), (-5.0, 5.0)]

        result = optimizer.minimize(objective, initial, bounds)

        assert result.best_loss < 0.1


class TestUtils:
    """Tests for utility functions."""

    def test_interpolate_to_times(self):
        """Test interpolation."""
        values = np.array([0, 10, 20, 30, 40])
        source_times = np.array([0, 1, 2, 3, 4])
        target_times = np.array([0.5, 1.5, 2.5])

        result = interpolate_to_times(values, source_times, target_times)

        assert len(result) == 3
        assert abs(result[0] - 5.0) < 0.1
        assert abs(result[1] - 15.0) < 0.1
        assert abs(result[2] - 25.0) < 0.1

    def test_convert_time_unit(self):
        """Test time unit conversion."""
        assert abs(convert_time_unit(1, "weeks", "days") - 7.0) < 0.01
        assert abs(convert_time_unit(365, "days", "years") - 1.0) < 0.01
        assert abs(convert_time_unit(24, "hours", "days") - 1.0) < 0.01

    def test_get_conversion_factor(self):
        """Test conversion factor."""
        assert abs(get_conversion_factor("weeks", "days") - 7.0) < 0.01
        assert abs(get_conversion_factor("days", "years") - 1 / 365.25) < 0.001

    def test_ensure_monotonic(self):
        """Test monotonic check."""
        assert ensure_monotonic(np.array([0, 1, 2, 3]))
        assert not ensure_monotonic(np.array([0, 2, 1, 3]))

    def test_find_time_overlap(self):
        """Test finding time overlap."""
        times1 = np.array([0, 1, 2, 3, 4, 5])
        times2 = np.array([2, 3, 4, 5, 6, 7])

        overlap = find_time_overlap(times1, times2)

        assert overlap == (2, 5)

    def test_find_time_overlap_no_overlap(self):
        """Test finding time overlap when there's none."""
        times1 = np.array([0, 1, 2])
        times2 = np.array([4, 5, 6])

        overlap = find_time_overlap(times1, times2)

        assert overlap is None


class TestModelFitter:
    """Tests for ModelFitter class."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic data from an SIR model."""
        model = SIR()
        true_beta = 0.3
        true_gamma = 0.1

        model(
            inits=[990, 10, 0],
            trange=[0, 50],
            totpop=1000,
            params={"beta": true_beta, "gamma": true_gamma},
        )

        times = model.traces["time"]
        I_obs = model.traces["I"]

        rng = np.random.default_rng(42)
        noise = rng.normal(0, 5, size=len(I_obs))
        I_obs_noisy = np.maximum(I_obs + noise, 0)

        return times, I_obs_noisy, true_beta, true_gamma

    def test_create_fitter(self, synthetic_data):
        """Test creating a fitter."""
        model = SIR()
        times, I_obs, _, _ = synthetic_data

        dataset = Dataset(model).register(
            name="cases", values=I_obs, times=times, state_variable="I"
        )

        fitter = ModelFitter(
            model=model,
            dataset=dataset,
            parameters_to_fit=[
                ParameterSpec("beta", bounds=(0.1, 1.0)),
                ParameterSpec("gamma", bounds=(0.05, 0.5)),
            ],
            total_population=1000,
        )

        assert fitter.model is model
        assert len(fitter.parameters_to_fit) == 2

    def test_fit_model(self, synthetic_data):
        """Test fitting a model."""
        model = SIR()
        times, I_obs, true_beta, true_gamma = synthetic_data

        dataset = Dataset(model).register(
            name="cases", values=I_obs, times=times, state_variable="I"
        )

        fitter = ModelFitter(
            model=model,
            dataset=dataset,
            parameters_to_fit=[
                ParameterSpec("beta", bounds=(0.1, 1.0), initial=0.5),
                ParameterSpec("gamma", bounds=(0.05, 0.5), initial=0.2),
            ],
            total_population=1000,
            optimizer=ScipyOptimizer(method="L-BFGS-B", max_iterations=100),
        )

        result = fitter.fit()

        assert result.convergence
        assert "beta" in result.best_params
        assert "gamma" in result.best_params

        assert abs(result.best_params["beta"] - true_beta) < 0.1
        assert abs(result.best_params["gamma"] - true_gamma) < 0.05

    def test_fit_model_with_fixed_params(self, synthetic_data):
        """Test fitting with fixed parameters."""
        model = SIR()
        times, I_obs, _, true_gamma = synthetic_data

        dataset = Dataset(model).register(
            name="cases", values=I_obs, times=times, state_variable="I"
        )

        fitter = ModelFitter(
            model=model,
            dataset=dataset,
            parameters_to_fit=[
                ParameterSpec("beta", bounds=(0.1, 1.0)),
            ],
            total_population=1000,
            fixed_params={"gamma": true_gamma},
        )

        result = fitter.fit()

        assert result.convergence
        assert "beta" in result.best_params
        assert "gamma" not in result.best_params

    def test_fit_model_invalid_param(self):
        """Test that invalid parameter raises error."""
        model = SIR()
        times = np.array([0, 10, 20, 30, 40])
        I_obs = np.array([10, 20, 30, 40, 50])

        dataset = Dataset(model).register(
            name="cases", values=I_obs, times=times, state_variable="I"
        )

        with pytest.raises(FittingError):
            ModelFitter(
                model=model,
                dataset=dataset,
                parameters_to_fit=[
                    ParameterSpec("invalid_param", bounds=(0.1, 1.0)),
                ],
                total_population=1000,
            )


class TestConvenienceFunction:
    """Tests for convenience fit_model function."""

    def test_fit_model_simple(self):
        """Test simple fitting with convenience function."""
        model = SIR()

        model(
            inits=[990, 10, 0],
            trange=[0, 30],
            totpop=1000,
            params={"beta": 0.3, "gamma": 0.1},
        )

        times = model.traces["time"]
        I_obs = model.traces["I"]

        result = fit_model(
            model=SIR(),
            data={"I": I_obs},
            times=times,
            params_to_fit={"beta": (0.1, 1.0), "gamma": (0.05, 0.5)},
            total_population=1000,
            variable_mapping={"I": "I"},
        )

        assert result.convergence
        assert "beta" in result.best_params
        assert "gamma" in result.best_params

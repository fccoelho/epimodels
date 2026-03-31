"""
Base classes and main API for model fitting.
"""

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from epimodels import BaseModel
from epimodels.fitting.data import Dataset
from epimodels.fitting.exceptions import (
    DataValidationError,
    FittingError,
)
from epimodels.fitting.objectives import LossFunction, SumOfSquaredErrors
from epimodels.fitting.optimizers import Optimizer, OptimizerResult, ScipyOptimizer
from epimodels.fitting.utils import (
    interpolate_to_times,
    rescale_parameter,
    unscale_parameter,
)


@dataclass
class ParameterSpec:
    """Specification for a parameter to be fitted."""

    name: str
    bounds: tuple[float, float]
    initial: float | None = None
    log_scale: bool = False

    def __post_init__(self):
        if self.bounds[0] >= self.bounds[1]:
            raise ValueError(
                f"Invalid bounds for parameter '{self.name}': "
                f"lower ({self.bounds[0]}) must be less than upper ({self.bounds[1]})"
            )

        if self.initial is not None:
            if not (self.bounds[0] <= self.initial <= self.bounds[1]):
                raise ValueError(
                    f"Initial value {self.initial} for parameter '{self.name}' "
                    f"is outside bounds {self.bounds}"
                )

        if self.log_scale:
            if self.bounds[0] <= 0 or self.bounds[1] <= 0:
                raise ValueError(f"Log-scale requires positive bounds for parameter '{self.name}'")
            if self.initial is not None and self.initial <= 0:
                raise ValueError(
                    f"Log-scale requires positive initial value for parameter '{self.name}'"
                )

    def get_initial_scaled(self) -> float:
        """Get initial value scaled to [0, 1] for optimization."""
        if self.initial is None:
            return 0.5
        return rescale_parameter(self.initial, self.bounds, self.log_scale)

    def unscale(self, scaled_value: float) -> float:
        """Convert scaled [0, 1] value to original scale."""
        return unscale_parameter(scaled_value, self.bounds, self.log_scale)


@dataclass
class InitialConditionSpec:
    """Specification for initial condition fitting."""

    state_variable: str
    bounds: tuple[float, float]
    initial: float | None = None
    fixed: bool = False


@dataclass
class FittingResult:
    """Result of model fitting."""

    best_params: dict[str, float]
    best_loss: float
    optimizer_result: OptimizerResult
    loss_history: list[float]
    n_evaluations: int
    convergence: bool
    confidence_intervals: dict[str, tuple[float, float]] | None = None
    fitted_model: BaseModel | None = None
    best_initial_conditions: list[float] | None = None
    predictions: dict[str, NDArray[np.floating]] | None = None


class ModelFitter:
    """
    Main class for fitting models to data.

    Example:
        >>> from epimodels.continuous import SIR
        >>> from epimodels.fitting import Dataset, ModelFitter, ParameterSpec
        >>>
        >>> model = SIR()
        >>> dataset = Dataset(model).register(
        ...     name="cases",
        ...     values=observed_I,
        ...     times=times,
        ...     state_variable="I",
        ... )
        >>>
        >>> fitter = ModelFitter(
        ...     model=model,
        ...     dataset=dataset,
        ...     parameters_to_fit=[
        ...         ParameterSpec("beta", bounds=(0.1, 1.0)),
        ...         ParameterSpec("gamma", bounds=(0.05, 0.5)),
        ...     ],
        ...     total_population=10000,
        ... )
        >>>
        >>> result = fitter.fit()
    """

    def __init__(
        self,
        model: BaseModel,
        dataset: Dataset,
        parameters_to_fit: list[ParameterSpec],
        total_population: float,
        loss_fn: LossFunction | None = None,
        optimizer: Optimizer | None = None,
        fit_initial_conditions: bool = False,
        initial_condition_specs: list[InitialConditionSpec] | None = None,
        fixed_params: dict[str, float] | None = None,
        solver_options: dict[str, Any] | None = None,
        time_offset: float = 0.0,
    ):
        """
        Initialize the model fitter.

        Args:
            model: The model to fit
            dataset: Dataset containing observed data
            parameters_to_fit: List of ParameterSpec for parameters to optimize
            total_population: Total population size
            loss_fn: Loss function to use (default: SumOfSquaredErrors)
            optimizer: Optimizer to use (default: ScipyOptimizer)
            fit_initial_conditions: Whether to fit initial conditions
            initial_condition_specs: Specifications for initial condition fitting
            fixed_params: Dictionary of fixed parameter values
            solver_options: Options to pass to the model solver
            time_offset: Time offset for model simulation
        """
        self.model = model
        self.dataset = dataset
        self.parameters_to_fit = {p.name: p for p in parameters_to_fit}
        self.total_population = total_population
        self.loss_fn = loss_fn or SumOfSquaredErrors()
        self.optimizer = optimizer or ScipyOptimizer()
        self.fit_initial_conditions = fit_initial_conditions
        self.initial_condition_specs = initial_condition_specs or []
        self.fixed_params = fixed_params or {}
        self.solver_options = solver_options or {}
        self.time_offset = time_offset

        self._validate_inputs()

        self._param_order = [p.name for p in parameters_to_fit]
        self._ic_order: list[str] = []
        self._fixed_ics: dict[str, float] = {}
        self._fitted_ic_specs: list[InitialConditionSpec] = []

        if self.fit_initial_conditions:
            self._setup_initial_condition_fitting()

    def _validate_inputs(self):
        """Validate all inputs before fitting."""
        model_params = set(self.model.parameters.keys())
        to_fit_params = set(self.parameters_to_fit.keys())

        extra_params = to_fit_params - model_params
        if extra_params:
            raise FittingError(
                f"Parameters to fit not in model: {extra_params}. "
                f"Available parameters: {model_params}"
            )

        fixed_set = set(self.fixed_params.keys())
        overlap = to_fit_params & fixed_set
        if overlap:
            raise FittingError(f"Parameters cannot be both fitted and fixed: {overlap}")

        validation = self.dataset.validate(self.total_population)
        if not validation.is_valid:
            raise DataValidationError(
                f"Dataset validation failed: {validation.errors}",
                errors=validation.errors,
            )

        if validation.warnings:
            for warning in validation.warnings:
                warnings.warn(warning, UserWarning)

    def _setup_initial_condition_fitting(self):
        """Setup initial condition fitting if enabled."""
        if not self.initial_condition_specs:
            self.initial_condition_specs = []
            for var_name in self.model.state_variables.keys():
                series = self.dataset.get_series_for_variable(var_name)
                if series is not None and len(series.values) > 0:
                    initial_val = series.values[0]
                    lower = max(0, initial_val * 0.1)
                    upper = min(self.total_population, initial_val * 10)
                    self.initial_condition_specs.append(
                        InitialConditionSpec(
                            state_variable=var_name,
                            bounds=(lower, upper),
                            initial=initial_val,
                        )
                    )

        self._fixed_ics = {}
        self._fitted_ic_specs = []
        for ic_spec in self.initial_condition_specs:
            if ic_spec.fixed:
                if ic_spec.initial is None:
                    raise FittingError(
                        f"Initial condition for '{ic_spec.state_variable}' is marked as fixed "
                        f"but no initial value provided"
                    )
                self._fixed_ics[ic_spec.state_variable] = ic_spec.initial
            else:
                self._fitted_ic_specs.append(ic_spec)

        self._ic_order = [ic.state_variable for ic in self._fitted_ic_specs]

    def _get_bounds(self) -> list[tuple[float, float]]:
        """Get parameter bounds for optimization."""
        bounds = []
        for name in self._param_order:
            spec = self.parameters_to_fit[name]
            bounds.append(spec.bounds)

        if self.fit_initial_conditions:
            for ic_spec in self._fitted_ic_specs:
                bounds.append(ic_spec.bounds)

        return bounds

    def _get_initial_params(self) -> NDArray[np.floating]:
        """Get initial parameter values for optimization."""
        initial = []

        for name in self._param_order:
            spec = self.parameters_to_fit[name]
            if spec.initial is not None:
                initial.append(spec.initial)
            else:
                lower, upper = spec.bounds
                if spec.log_scale:
                    initial.append(np.sqrt(lower * upper))
                else:
                    initial.append((lower + upper) / 2)

        if self.fit_initial_conditions:
            for ic_spec in self._fitted_ic_specs:
                if ic_spec.initial is not None:
                    initial.append(ic_spec.initial)
                else:
                    lower, upper = ic_spec.bounds
                    initial.append((lower + upper) / 2)

        return np.array(initial)

    def _params_array_to_dict(
        self,
        params_array: NDArray[np.floating],
    ) -> tuple[dict[str, float], list[float] | None]:
        """Convert parameter array to dictionary."""
        n_params = len(self._param_order)
        params_dict = {}

        for i, name in enumerate(self._param_order):
            params_dict[name] = float(params_array[i])

        initial_conditions = None
        if self.fit_initial_conditions:
            initial_conditions = []
            ic_values = params_array[n_params:]
            for i, var_name in enumerate(self._ic_order):
                initial_conditions.append(float(ic_values[i]))

        return params_dict, initial_conditions

    def _run_model(
        self,
        params: dict[str, float],
        initial_conditions: list[float] | None = None,
    ) -> tuple[dict[str, NDArray[np.floating]], NDArray[np.floating]]:
        """Run the model with given parameters.

        Returns:
            Tuple of (predictions dict, time array)
        """
        model = self.model.copy()

        all_params = {**self.fixed_params, **params}

        time_range = self.dataset.time_range
        if time_range is None:
            raise FittingError("No time range available in dataset")

        t0, tf = time_range
        trange = [t0 - self.time_offset, tf - self.time_offset]

        if initial_conditions is None:
            initial_conditions = self._estimate_initial_conditions()
        elif self.fit_initial_conditions and self._fixed_ics:
            merged_ic = {}
            var_list = list(model.state_variables.keys())
            for i, var_name in enumerate(var_list):
                if var_name in self._fixed_ics:
                    merged_ic[var_name] = self._fixed_ics[var_name]
                else:
                    ic_idx = self._ic_order.index(var_name)
                    merged_ic[var_name] = initial_conditions[ic_idx]
            initial_conditions = [merged_ic[var] for var in var_list]

        model(
            inits=initial_conditions,
            trange=trange,
            totpop=self.total_population,
            params=all_params,
            **self.solver_options,
        )

        predictions = {var: model.traces[var] for var in model.state_variables.keys()}
        model_times = model.traces.get(
            "time", np.arange(len(predictions[list(predictions.keys())[0]]))
        )

        return predictions, model_times

    def _estimate_initial_conditions(self) -> list[float]:
        """Estimate initial conditions from data."""
        n_vars = len(self.model.state_variables)
        inits = [0.0] * n_vars

        data_values = {}
        for var_name in self.model.state_variables.keys():
            series = self.dataset.get_series_for_variable(var_name)
            if series is not None and len(series.values) > 0:
                data_values[var_name] = series.values

        for i, var_name in enumerate(self.model.state_variables.keys()):
            if var_name in data_values:
                inits[i] = float(data_values[var_name][0])

        total_assigned = sum(inits)
        if total_assigned < self.total_population:
            remaining = self.total_population - total_assigned
            var_list = list(self.model.state_variables.keys())
            if "S" in var_list:
                susceptible_idx = var_list.index("S")
                inits[susceptible_idx] += remaining

        return inits

    def _interpolate_predictions(
        self,
        predictions: dict[str, NDArray[np.floating]],
        model_times: NDArray[np.floating],
    ) -> dict[str, NDArray[np.floating]]:
        """Interpolate model predictions to data time points."""
        interpolated = {}

        for series_name, series in self.dataset.series.items():
            var_name = series.state_variable
            if var_name in predictions:
                interp_values = interpolate_to_times(
                    predictions[var_name],
                    model_times,
                    series.times,
                )
                interpolated[var_name] = interp_values

        return interpolated

    def _compute_loss(
        self,
        params_array: NDArray[np.floating],
    ) -> float:
        """Compute loss for given parameter array."""
        params_dict, initial_conditions = self._params_array_to_dict(params_array)

        try:
            predictions, model_times = self._run_model(params_dict, initial_conditions)

            interpolated = self._interpolate_predictions(predictions, model_times)

            observed = {}
            for series_name, series in self.dataset.series.items():
                var_name = series.state_variable
                if var_name in interpolated:
                    observed[var_name] = series.values

            loss_result = self.loss_fn.compute(observed, interpolated)
            return loss_result.value

        except Exception as e:
            warnings.warn(f"Model evaluation failed: {e}", RuntimeWarning)
            return 1e10

    def fit(
        self,
        callback: Callable[[int, NDArray[np.floating], float], None] | None = None,
        verbose: bool = False,
    ) -> FittingResult:
        """
        Fit the model to the data.

        Args:
            callback: Optional callback for monitoring progress
            verbose: Whether to print progress information

        Returns:
            FittingResult with fitted parameters and diagnostics
        """
        initial_params = self._get_initial_params()
        bounds = self._get_bounds()

        if verbose:

            def wrapped_callback(iteration, params, loss):
                print(f"Iteration {iteration}: loss = {loss:.6f}")
                if callback is not None:
                    callback(iteration, params, loss)

            opt_callback = wrapped_callback
        else:
            opt_callback = callback

        result = self.optimizer.minimize(
            self._compute_loss,
            initial_params,
            bounds,
            opt_callback,
        )

        best_params_dict, best_ic = self._params_array_to_dict(result.best_params)

        fitted_model = self.model.copy()
        all_params = {**self.fixed_params, **best_params_dict}

        if best_ic is None:
            best_ic = self._estimate_initial_conditions()
        elif self.fit_initial_conditions and self._fixed_ics:
            merged_ic = {}
            var_list = list(fitted_model.state_variables.keys())
            for i, var_name in enumerate(var_list):
                if var_name in self._fixed_ics:
                    merged_ic[var_name] = self._fixed_ics[var_name]
                else:
                    ic_idx = self._ic_order.index(var_name)
                    merged_ic[var_name] = best_ic[ic_idx]
            best_ic = [merged_ic[var] for var in var_list]

        time_range = self.dataset.time_range
        if time_range is not None:
            t0, tf = time_range
            fitted_model(
                inits=best_ic,
                trange=[t0 - self.time_offset, tf - self.time_offset],
                totpop=self.total_population,
                params=all_params,
                **self.solver_options,
            )

        predictions = None
        if fitted_model.traces:
            predictions = {
                var: fitted_model.traces[var] for var in fitted_model.state_variables.keys()
            }

        return FittingResult(
            best_params=best_params_dict,
            best_loss=result.best_loss,
            optimizer_result=result,
            loss_history=result.loss_history,
            n_evaluations=result.n_evaluations,
            convergence=result.success,
            best_initial_conditions=best_ic,
            fitted_model=fitted_model,
            predictions=predictions,
        )

    def profile_likelihood(
        self,
        param_name: str,
        n_points: int = 20,
        threshold: float = 3.84,
    ) -> dict:
        """
        Compute profile likelihood for a parameter.

        Args:
            param_name: Name of parameter to profile
            n_points: Number of points to evaluate
            threshold: Threshold for confidence interval (default: chi-squared 95% for 1 df)

        Returns:
            Dictionary with profile results
        """
        if param_name not in self.parameters_to_fit:
            raise ValueError(f"Parameter '{param_name}' not in parameters to fit")

        spec = self.parameters_to_fit[param_name]
        lower, upper = spec.bounds

        param_values = np.linspace(lower, upper, n_points)
        losses = []

        original_initial = spec.initial

        for val in param_values:
            spec.initial = val

            other_params = [
                self.parameters_to_fit[name] for name in self._param_order if name != param_name
            ]

            temp_fitter = ModelFitter(
                model=self.model,
                dataset=self.dataset,
                parameters_to_fit=other_params,
                total_population=self.total_population,
                loss_fn=self.loss_fn,
                optimizer=self.optimizer,
                fit_initial_conditions=self.fit_initial_conditions,
                initial_condition_specs=self.initial_condition_specs,
                fixed_params={**self.fixed_params, param_name: val},
                solver_options=self.solver_options,
            )

            result = temp_fitter.fit()
            losses.append(result.best_loss)

        spec.initial = original_initial

        min_loss = min(losses)
        threshold_loss = min_loss + threshold / 2

        ci_lower = None
        ci_upper = None

        for i, (val, loss) in enumerate(zip(param_values, losses, strict=False)):
            if loss <= threshold_loss:
                if ci_lower is None:
                    ci_lower = val
                ci_upper = val

        return {
            "parameter": param_name,
            "values": param_values.tolist(),
            "losses": losses,
            "min_loss": min_loss,
            "threshold": threshold,
            "threshold_loss": threshold_loss,
            "confidence_interval": (ci_lower, ci_upper),
        }


def fit_model(
    model: BaseModel,
    data: dict[str, NDArray[np.floating]] | Any,
    times: NDArray[np.floating],
    params_to_fit: dict[str, tuple[float, float]],
    total_population: float,
    variable_mapping: dict[str, str] | None = None,
    **kwargs,
) -> FittingResult:
    """
    Convenience function for simple model fitting.

    Args:
        model: The model to fit
        data: Dictionary mapping series names to observed values, or DataFrame
        times: Time points for observations
        params_to_fit: Dict mapping parameter names to (lower, upper) bounds
        total_population: Total population size
        variable_mapping: Dict mapping series names to state variables
        **kwargs: Additional arguments passed to ModelFitter

    Returns:
        FittingResult with fitted parameters

    Example:
        >>> result = fit_model(
        ...     model=SIR(),
        ...     data={"cases": observed_I},
        ...     times=times,
        ...     params_to_fit={"beta": (0.1, 1.0), "gamma": (0.05, 0.5)},
        ...     total_population=10000,
        ...     variable_mapping={"cases": "I"},
        ... )
    """
    if variable_mapping is None:
        variable_mapping = {}
        for name in data.keys():
            variable_mapping[name] = name

    dataset = Dataset(model)
    for name, values in data.items():
        state_var = variable_mapping.get(name, name)
        dataset.register(
            name=name,
            values=values,
            times=times,
            state_variable=state_var,
        )

    param_specs = [
        ParameterSpec(name=name, bounds=bounds) for name, bounds in params_to_fit.items()
    ]

    fitter = ModelFitter(
        model=model,
        dataset=dataset,
        parameters_to_fit=param_specs,
        total_population=total_population,
        **kwargs,
    )

    return fitter.fit()


__all__ = [
    "ParameterSpec",
    "InitialConditionSpec",
    "FittingResult",
    "ModelFitter",
    "fit_model",
]

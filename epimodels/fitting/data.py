"""
Data handling and validation for model fitting.
"""

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    pass

try:
    import pandas as _pd

    HAS_PANDAS = True
except ImportError:
    _pd = None  # type: ignore
    HAS_PANDAS = False


@dataclass
class TimeCompatibility:
    """Result of time scale compatibility check."""

    is_compatible: bool
    model_time_unit: str
    data_time_unit: str
    conversion_factor: float
    issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


@dataclass
class DataSeries:
    """Single observed data series."""

    name: str
    values: NDArray[np.floating]
    times: NDArray[np.floating]
    state_variable: str
    uncertainty: NDArray[np.floating] | None = None
    time_unit: str = "days"

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=float)
        self.times = np.asarray(self.times, dtype=float)
        if self.uncertainty is not None:
            self.uncertainty = np.asarray(self.uncertainty, dtype=float)

    def validate(self) -> list[str]:
        """Validate the data series. Returns list of error messages."""
        errors = []

        if len(self.values) != len(self.times):
            errors.append(
                f"Series '{self.name}': values and times have different lengths "
                f"({len(self.values)} vs {len(self.times)})"
            )

        if len(self.times) == 0:
            errors.append(f"Series '{self.name}': empty time array")
            return errors

        if not np.all(np.diff(self.times) > 0):
            errors.append(f"Series '{self.name}': times are not monotonically increasing")

        if np.any(np.isnan(self.values)):
            n_nan = np.sum(np.isnan(self.values))
            errors.append(f"Series '{self.name}': contains {n_nan} NaN values")

        if np.any(np.isinf(self.values)):
            n_inf = np.sum(np.isinf(self.values))
            errors.append(f"Series '{self.name}': contains {n_inf} infinite values")

        if self.uncertainty is not None:
            if len(self.uncertainty) != len(self.values):
                errors.append(f"Series '{self.name}': uncertainty has different length than values")
            elif np.any(self.uncertainty <= 0):
                errors.append(f"Series '{self.name}': uncertainty must be positive")

        return errors

    @property
    def time_range(self) -> tuple[float, float]:
        """Return the time range of this series."""
        return (float(self.times.min()), float(self.times.max()))

    def to_dict(self) -> dict:
        """Export to dictionary for serialization."""
        return {
            "name": self.name,
            "values": self.values.tolist(),
            "times": self.times.tolist(),
            "state_variable": self.state_variable,
            "uncertainty": self.uncertainty.tolist() if self.uncertainty is not None else None,
            "time_unit": self.time_unit,
        }


@dataclass
class ValidationResult:
    """Result of dataset validation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    time_compatibility: bool = True
    time_unit_consistency: bool = True
    variable_existence: bool = True
    value_validity: bool = True
    common_time_range: tuple[float, float] | None = None
    missing_variables: list[str] = field(default_factory=list)
    extra_variables: list[str] = field(default_factory=list)
    time_compatibility_details: TimeCompatibility | None = None


class Dataset:
    """
    Container for observed data with validation.

    Handles registration and validation of multiple data series
    against a model's structure.
    """

    VALID_TIME_UNITS = {"hours", "days", "weeks", "years"}

    def __init__(self, model: Any):
        """
        Initialize dataset for a specific model.

        Args:
            model: The model instance to fit data to
        """
        self.model = model
        self._series: dict[str, DataSeries] = {}
        self._time_unit: str = "days"
        self._variable_mapping: dict[str, str] = {}

    @property
    def series(self) -> dict[str, DataSeries]:
        """Return all registered data series."""
        return self._series

    @property
    def time_unit(self) -> str:
        """Return the common time unit."""
        return self._time_unit

    @time_unit.setter
    def time_unit(self, value: str):
        if value.lower() not in self.VALID_TIME_UNITS:
            raise ValueError(f"Invalid time unit '{value}'. Must be one of {self.VALID_TIME_UNITS}")
        self._time_unit = value.lower()

    def register(
        self,
        name: str,
        values: NDArray[np.floating] | Any,
        times: NDArray[np.floating] | Any,
        state_variable: str,
        uncertainty: NDArray[np.floating] | None = None,
        time_unit: str = "days",
    ) -> "Dataset":
        """
        Register a data series.

        Args:
            name: Unique identifier for this data series
            values: Observed values
            times: Time points corresponding to values
            state_variable: Which model state variable this corresponds to
            uncertainty: Optional uncertainty (std errors) for each value
            time_unit: Time unit for this series

        Returns:
            Self for method chaining
        """
        if HAS_PANDAS and isinstance(values, _pd.Series):  # type: ignore
            values = values.values
        if HAS_PANDAS and isinstance(times, _pd.Series):  # type: ignore
            times = times.values

        if name in self._series:
            warnings.warn(f"Overwriting existing series '{name}'", UserWarning)

        series = DataSeries(
            name=name,
            values=np.asarray(values, dtype=float),
            times=np.asarray(times, dtype=float),
            state_variable=state_variable,
            uncertainty=uncertainty,
            time_unit=time_unit.lower(),
        )

        self._series[name] = series
        self._variable_mapping[series.state_variable] = name

        if len(self._series) == 1:
            self._time_unit = time_unit.lower()

        return self

    def register_from_dataframe(
        self,
        df: Any,
        time_column: str,
        mapping: dict[str, str],
        uncertainty_mapping: dict[str, str] | None = None,
        time_unit: str = "days",
    ) -> "Dataset":
        """
        Register multiple series from a DataFrame.

        Args:
            df: pandas DataFrame containing the data
            time_column: Name of the column with time values
            mapping: Dict mapping column names to state variable names
                     e.g., {"cases": "I", "recovered": "R"}
            uncertainty_mapping: Optional dict mapping column names to uncertainty columns
            time_unit: Time unit for all series

        Returns:
            Self for method chaining
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required for register_from_dataframe")

        times = df[time_column].values

        for col_name, state_var in mapping.items():
            values = df[col_name].values

            uncertainty = None
            if uncertainty_mapping and col_name in uncertainty_mapping:
                uncertainty = df[uncertainty_mapping[col_name]].values

            self.register(
                name=col_name,
                values=values,
                times=times,
                state_variable=state_var,
                uncertainty=uncertainty,
                time_unit=time_unit,
            )

        return self

    def unregister(self, name: str) -> "Dataset":
        """Remove a registered series."""
        if name in self._series:
            state_var = self._series[name].state_variable
            self._variable_mapping.pop(state_var, None)
            del self._series[name]
        return self

    def clear(self) -> "Dataset":
        """Remove all registered series."""
        self._series.clear()
        self._variable_mapping.clear()
        return self

    def get_series_for_variable(self, state_var: str) -> DataSeries | None:
        """Get data series mapped to a state variable."""
        if state_var in self._variable_mapping:
            return self._series[self._variable_mapping[state_var]]
        return None

    @property
    def time_range(self) -> tuple[float, float] | None:
        """Return the common time range across all series."""
        if not self._series:
            return None

        starts = [s.time_range[0] for s in self._series.values()]
        ends = [s.time_range[1] for s in self._series.values()]

        return (min(starts), max(ends))

    @property
    def time_points(self) -> NDArray[np.floating] | None:
        """Return unique sorted time points from all series."""
        if not self._series:
            return None

        all_times = np.concatenate([s.times for s in self._series.values()])
        return np.unique(all_times)

    def _check_time_compatibility(self) -> TimeCompatibility:
        """Check time compatibility across all series."""
        issues = []
        recommendations = []

        time_units = set(s.time_unit for s in self._series.values())

        if len(time_units) > 1:
            issues.append(f"Multiple time units detected: {time_units}")
            recommendations.append("Convert all series to a common time unit")

        model_vars = set(self.model.state_variables.keys())
        data_vars = set(s.state_variable for s in self._series.values())

        time_unit = self._time_unit

        return TimeCompatibility(
            is_compatible=len(issues) == 0,
            model_time_unit=time_unit,
            data_time_unit=time_unit,
            conversion_factor=1.0,
            issues=issues,
            recommendations=recommendations,
        )

    def validate(self, total_population: float | None = None) -> ValidationResult:
        """
        Validate data against model.

        Args:
            total_population: Optional total population for bound checking

        Returns:
            ValidationResult with detailed information
        """
        errors = []
        warnings_list = []

        if not self._series:
            errors.append("No data series registered")
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings_list,
            )

        model_vars = set(self.model.state_variables.keys())

        all_series_errors = []
        for name, series in self._series.items():
            series_errors = series.validate()
            all_series_errors.extend(series_errors)

            if series.state_variable not in model_vars:
                all_series_errors.append(
                    f"Series '{name}': state variable '{series.state_variable}' "
                    f"not found in model. Available: {list(model_vars)}"
                )

        errors.extend(all_series_errors)
        value_validity = len(all_series_errors) == 0

        data_vars = set(s.state_variable for s in self._series.values())
        extra_variables = list(data_vars - model_vars)
        missing_variables = list(model_vars - data_vars)

        if extra_variables:
            errors.append(f"Data mapped to non-existent variables: {extra_variables}")
        variable_existence = len(extra_variables) == 0

        time_units = set(s.time_unit for s in self._series.values())
        time_unit_consistency = len(time_units) <= 1
        if not time_unit_consistency:
            warnings_list.append(f"Multiple time units in use: {time_units}")

        time_compat = self._check_time_compatibility()

        for series in self._series.values():
            if np.any(series.values < 0):
                warnings_list.append(f"Series '{series.name}': negative values detected")

            if total_population is not None:
                if np.any(series.values > total_population):
                    warnings_list.append(f"Series '{series.name}': values exceed total population")

        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings_list,
            time_compatibility=time_compat.is_compatible,
            time_unit_consistency=time_unit_consistency,
            variable_existence=variable_existence,
            value_validity=value_validity,
            common_time_range=self.time_range,
            missing_variables=missing_variables,
            extra_variables=extra_variables,
            time_compatibility_details=time_compat,
        )

    def to_dict(self) -> dict:
        """Export dataset to dictionary for serialization."""
        return {
            "series": {name: series.to_dict() for name, series in self._series.items()},
            "time_unit": self._time_unit,
            "time_range": self.time_range,
        }

    def __repr__(self) -> str:
        n_series = len(self._series)
        vars_mapped = list(set(s.state_variable for s in self._series.values()))
        time_range = self.time_range
        return (
            f"Dataset(n_series={n_series}, "
            f"variables={vars_mapped}, "
            f"time_range={time_range})"
        )


__all__ = [
    "DataSeries",
    "Dataset",
    "ValidationResult",
    "TimeCompatibility",
]

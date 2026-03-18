"""
Utility functions for model fitting.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d


def interpolate_to_times(
    values: NDArray[np.floating],
    source_times: NDArray[np.floating],
    target_times: NDArray[np.floating],
    method: str = "linear",
    extrapolate: bool = True,
    fill_value: Any = None,
) -> NDArray[np.floating]:
    """
    Interpolate values from source times to target times.

    Args:
        values: Array of values at source times
        source_times: Original time points
        target_times: Desired time points
        method: Interpolation method ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic')
        extrapolate: If True, extrapolate beyond bounds. If False, use fill_value.
        fill_value: Value to use for points outside bounds (if extrapolate=False)

    Returns:
        Interpolated values at target times
    """
    if len(values) != len(source_times):
        raise ValueError(
            f"Length mismatch: values has {len(values)} elements, "
            f"source_times has {len(source_times)} elements"
        )

    if method not in ["linear", "nearest", "zero", "slinear", "quadratic", "cubic"]:
        raise ValueError(f"Unknown interpolation method: {method}")

    if np.array_equal(source_times, target_times):
        return values.copy()

    if extrapolate:
        interpolator = interp1d(
            source_times,
            values,
            kind=method,
            fill_value=float(np.nan),
            bounds_error=False,
        )
    else:
        fv = fill_value if fill_value is not None else float(np.nan)
        interpolator = interp1d(
            source_times,
            values,
            kind=method,
            fill_value=fv,
            bounds_error=True,
        )

    return interpolator(target_times)


def convert_time_unit(value: float, from_unit: str, to_unit: str) -> float:
    """
    Convert a time value from one unit to another.

    Args:
        value: Time value in source unit
        from_unit: Source time unit ('days', 'weeks', 'years', 'hours')
        to_unit: Target time unit

    Returns:
        Converted time value
    """
    conversion_factors = {
        "hours": 1.0 / 24.0,
        "days": 1.0,
        "weeks": 7.0,
        "years": 365.25,
    }

    from_unit = from_unit.lower()
    to_unit = to_unit.lower()

    if from_unit not in conversion_factors:
        raise ValueError(f"Unknown time unit: {from_unit}")
    if to_unit not in conversion_factors:
        raise ValueError(f"Unknown time unit: {to_unit}")

    days_value = value * conversion_factors[from_unit]
    return days_value / conversion_factors[to_unit]


def get_conversion_factor(from_unit: str, to_unit: str) -> float:
    """
    Get conversion factor between time units.

    Args:
        from_unit: Source time unit
        to_unit: Target time unit

    Returns:
        Multiplication factor to convert from_unit to to_unit
    """
    conversion_factors = {
        "hours": 1.0 / 24.0,
        "days": 1.0,
        "weeks": 7.0,
        "years": 365.25,
    }

    from_unit = from_unit.lower()
    to_unit = to_unit.lower()

    if from_unit not in conversion_factors:
        raise ValueError(f"Unknown time unit: {from_unit}")
    if to_unit not in conversion_factors:
        raise ValueError(f"Unknown time unit: {to_unit}")

    return conversion_factors[from_unit] / conversion_factors[to_unit]


def ensure_monotonic(times: NDArray[np.floating]) -> bool:
    """Check if times are monotonically increasing."""
    return bool(np.all(np.diff(times) > 0))


def find_time_overlap(
    times1: NDArray[np.floating],
    times2: NDArray[np.floating],
) -> tuple[float, float] | None:
    """
    Find overlapping time range between two time arrays.

    Args:
        times1: First time array
        times2: Second time array

    Returns:
        Tuple of (start, end) of overlap, or None if no overlap
    """
    start = max(times1.min(), times2.min())
    end = min(times1.max(), times2.max())

    if start >= end:
        return None

    return (start, end)


def rescale_parameter(
    value: float,
    bounds: tuple[float, float],
    log_scale: bool = False,
) -> float:
    """
    Rescale parameter to [0, 1] interval for optimization.

    Args:
        value: Parameter value
        bounds: (lower, upper) bounds
        log_scale: If True, use log-scale transformation

    Returns:
        Rescaled value in [0, 1]
    """
    lower, upper = bounds

    if log_scale:
        if value <= 0 or lower <= 0 or upper <= 0:
            raise ValueError("Log-scale requires positive values and bounds")
        log_val = np.log(value)
        log_lower = np.log(lower)
        log_upper = np.log(upper)
        return (log_val - log_lower) / (log_upper - log_lower)
    else:
        return (value - lower) / (upper - lower)


def unscale_parameter(
    scaled_value: float,
    bounds: tuple[float, float],
    log_scale: bool = False,
) -> float:
    """
    Convert scaled [0, 1] value back to original parameter scale.

    Args:
        scaled_value: Value in [0, 1]
        bounds: (lower, upper) bounds
        log_scale: If True, use log-scale transformation

    Returns:
        Parameter value in original scale
    """
    lower, upper = bounds
    scaled_value = np.clip(scaled_value, 0.0, 1.0)

    if log_scale:
        if lower <= 0 or upper <= 0:
            raise ValueError("Log-scale requires positive bounds")
        log_lower = np.log(lower)
        log_upper = np.log(upper)
        log_val = log_lower + scaled_value * (log_upper - log_lower)
        return float(np.exp(log_val))
    else:
        return lower + scaled_value * (upper - lower)


def estimate_initial_conditions(
    model: Any,
    data_values: dict[str, NDArray[np.floating]],
    time_points: NDArray[np.floating],
    total_population: float,
) -> list[float]:
    """
    Estimate initial conditions from data.

    Args:
        model: The model instance
        data_values: Dict mapping state variable names to observed values
        time_points: Time points of observations
        total_population: Total population size

    Returns:
        Estimated initial conditions as a list
    """
    n_vars = len(model.state_variables)
    inits = [0.0] * n_vars

    for i, var_name in enumerate(model.state_variables.keys()):
        if var_name in data_values and len(data_values[var_name]) > 0:
            inits[i] = float(data_values[var_name][0])

    total_assigned = sum(inits)
    if total_assigned < total_population:
        remaining = total_population - total_assigned
        susceptible_idx = (
            list(model.state_variables.keys()).index("S") if "S" in model.state_variables else 0
        )
        inits[susceptible_idx] += remaining

    return inits


__all__ = [
    "interpolate_to_times",
    "convert_time_unit",
    "get_conversion_factor",
    "ensure_monotonic",
    "find_time_overlap",
    "rescale_parameter",
    "unscale_parameter",
    "estimate_initial_conditions",
]

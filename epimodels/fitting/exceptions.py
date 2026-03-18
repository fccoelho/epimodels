"""
Exceptions for the fitting module.
"""


class FittingError(Exception):
    """Base exception for fitting-related errors."""

    pass


class DataValidationError(FittingError):
    """Raised when data validation fails."""

    def __init__(self, message: str, errors: list[str] | None = None):
        self.errors = errors or []
        super().__init__(message)


class OptimizationError(FittingError):
    """Raised when optimization fails."""

    def __init__(self, message: str, optimizer_result: object | None = None):
        self.optimizer_result = optimizer_result
        super().__init__(message)


class TimeCompatibilityError(FittingError):
    """Raised when time scales are incompatible between model and data."""

    pass


class ParameterBoundsError(FittingError):
    """Raised when parameter bounds are invalid."""

    pass


class ConvergenceError(FittingError):
    """Raised when optimization does not converge."""

    def __init__(
        self, message: str, n_iterations: int | None = None, loss_history: list | None = None
    ):
        self.n_iterations = n_iterations
        self.loss_history = loss_history
        super().__init__(message)


__all__ = [
    "FittingError",
    "DataValidationError",
    "OptimizationError",
    "TimeCompatibilityError",
    "ParameterBoundsError",
    "ConvergenceError",
]

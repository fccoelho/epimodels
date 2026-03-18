"""
Model fitting module for epimodels.

This module provides tools for fitting epidemic models to observed data.

Main Classes:
    - Dataset: Container for observed data with validation
    - DataSeries: Single observed data series
    - ModelFitter: Main class for fitting models to data
    - ParameterSpec: Specification for a parameter to fit

Loss Functions:
    - SumOfSquaredErrors: Standard SSE loss
    - WeightedSSE: Weighted sum of squared errors
    - PoissonLikelihood: For count data
    - NegativeBinomialLikelihood: For overdispersed count data
    - NormalLikelihood: Gaussian likelihood
    - HuberLoss: Robust to outliers

Optimizers:
    - ScipyOptimizer: Scipy-based optimization
    - JAXOptimizer: GPU-accelerated with JAX/optimistix
    - NevergradOptimizer: Derivative-free optimization
    - MultiStartOptimizer: Multi-start wrapper

Convenience Functions:
    - fit_model: Simple fitting with minimal setup

Example:
    >>> from epimodels.continuous import SIR
    >>> from epimodels.fitting import Dataset, ModelFitter, ParameterSpec
    >>>
    >>> # Create model and data
    >>> model = SIR()
    >>> dataset = Dataset(model).register(
    ...     name="cases",
    ...     values=observed_values,
    ...     times=time_points,
    ...     state_variable="I",
    ... )
    >>>
    >>> # Set up fitter
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
    >>> # Fit
    >>> result = fitter.fit()
    >>> print(f"Fitted parameters: {result.best_params}")
"""

from epimodels.fitting.base import (
    FittingResult,
    InitialConditionSpec,
    ModelFitter,
    ParameterSpec,
    fit_model,
)
from epimodels.fitting.data import (
    DataSeries,
    Dataset,
    TimeCompatibility,
    ValidationResult,
)
from epimodels.fitting.exceptions import (
    ConvergenceError,
    DataValidationError,
    FittingError,
    OptimizationError,
    ParameterBoundsError,
    TimeCompatibilityError,
)
from epimodels.fitting.objectives import (
    CustomLoss,
    HuberLoss,
    LogLikelihood,
    LossFunction,
    LossResult,
    NegativeBinomialLikelihood,
    NormalLikelihood,
    PoissonLikelihood,
    SumOfSquaredErrors,
    WeightedSSE,
)
from epimodels.fitting.optimizers import (
    JAXOptimizer,
    MultiStartOptimizer,
    NevergradOptimizer,
    Optimizer,
    OptimizerResult,
    ScipyOptimizer,
)

__all__ = [
    # Exceptions
    "FittingError",
    "DataValidationError",
    "OptimizationError",
    "TimeCompatibilityError",
    "ParameterBoundsError",
    "ConvergenceError",
    # Data
    "Dataset",
    "DataSeries",
    "ValidationResult",
    "TimeCompatibility",
    # Loss functions
    "LossFunction",
    "LossResult",
    "SumOfSquaredErrors",
    "WeightedSSE",
    "PoissonLikelihood",
    "NegativeBinomialLikelihood",
    "NormalLikelihood",
    "LogLikelihood",
    "CustomLoss",
    "HuberLoss",
    # Optimizers
    "Optimizer",
    "OptimizerResult",
    "ScipyOptimizer",
    "JAXOptimizer",
    "NevergradOptimizer",
    "MultiStartOptimizer",
    # Main API
    "ParameterSpec",
    "InitialConditionSpec",
    "FittingResult",
    "ModelFitter",
    "fit_model",
]

"""
Loss functions for model fitting.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln


@dataclass
class LossResult:
    """Result of loss function computation."""

    value: float
    per_point: NDArray[np.floating] | None = None
    per_variable: dict[str, float] | None = None


class LossFunction(ABC):
    """Base class for loss functions."""

    @abstractmethod
    def compute(
        self,
        observed: dict[str, NDArray[np.floating]],
        predicted: dict[str, NDArray[np.floating]],
        weights: dict[str, NDArray[np.floating]] | None = None,
    ) -> LossResult:
        """
        Compute the loss between observed and predicted values.

        Args:
            observed: Dict mapping variable names to observed values
            predicted: Dict mapping variable names to predicted values
            weights: Optional dict mapping variable names to weights

        Returns:
            LossResult with total loss and optional breakdowns
        """
        pass

    def __call__(
        self,
        observed: dict[str, NDArray[np.floating]],
        predicted: dict[str, NDArray[np.floating]],
        weights: dict[str, NDArray[np.floating]] | None = None,
    ) -> float:
        """Call compute() and return only the total loss value."""
        result = self.compute(observed, predicted, weights)
        return result.value


class SumOfSquaredErrors(LossFunction):
    """
    Sum of squared errors loss function.

    L = Σ (y_obs - y_pred)²
    """

    def __init__(self, normalize: bool = False):
        """
        Args:
            normalize: If True, divide by number of observations
        """
        self.normalize = normalize

    def compute(
        self,
        observed: dict[str, NDArray[np.floating]],
        predicted: dict[str, NDArray[np.floating]],
        weights: dict[str, NDArray[np.floating]] | None = None,
    ) -> LossResult:
        per_variable = {}
        per_point_list = []
        total = 0.0
        n_total = 0

        for var_name, obs in observed.items():
            if var_name not in predicted:
                continue

            pred = predicted[var_name]

            if weights is not None and var_name in weights:
                w = weights[var_name]
                residuals_sq = w * (obs - pred) ** 2
            else:
                residuals_sq = (obs - pred) ** 2

            var_loss = float(np.sum(residuals_sq))
            per_variable[var_name] = var_loss
            total += var_loss
            n_total += len(obs)

            per_point_list.append(residuals_sq)

        if self.normalize and n_total > 0:
            total /= n_total

        per_point = np.concatenate(per_point_list) if per_point_list else np.array([])

        return LossResult(
            value=total,
            per_point=per_point,
            per_variable=per_variable,
        )


class WeightedSSE(LossFunction):
    """
    Weighted sum of squared errors.

    L = Σ w_i * (y_obs_i - y_pred_i)²
    """

    def __init__(
        self,
        variable_weights: dict[str, float] | None = None,
        point_weights: dict[str, NDArray[np.floating]] | None = None,
    ):
        """
        Args:
            variable_weights: Weight multiplier for each variable
            point_weights: Array of weights for each data point per variable
        """
        self.variable_weights = variable_weights or {}
        self.point_weights = point_weights or {}

    def compute(
        self,
        observed: dict[str, NDArray[np.floating]],
        predicted: dict[str, NDArray[np.floating]],
        weights: dict[str, NDArray[np.floating]] | None = None,
    ) -> LossResult:
        per_variable = {}
        per_point_list = []
        total = 0.0

        for var_name, obs in observed.items():
            if var_name not in predicted:
                continue

            pred = predicted[var_name]

            residuals_sq = (obs - pred) ** 2

            combined_weights = np.ones(len(obs), dtype=float)

            if weights is not None and var_name in weights:
                combined_weights *= weights[var_name]

            if var_name in self.point_weights:
                combined_weights *= self.point_weights[var_name]

            if var_name in self.variable_weights:
                combined_weights *= self.variable_weights[var_name]

            weighted_sq = combined_weights * residuals_sq
            var_loss = float(np.sum(weighted_sq))
            per_variable[var_name] = var_loss
            total += var_loss

            per_point_list.append(weighted_sq)

        per_point = np.concatenate(per_point_list) if per_point_list else np.array([])

        return LossResult(
            value=total,
            per_point=per_point,
            per_variable=per_variable,
        )


class PoissonLikelihood(LossFunction):
    """
    Negative Poisson log-likelihood for count data.

    L = -2 * Σ [y_obs * log(y_pred) - y_pred - log(y_obs!)]

    Note: Returns -2 * log-likelihood for compatibility with chi-squared statistics.
    """

    def __init__(self, epsilon: float = 1e-10):
        """
        Args:
            epsilon: Small value to avoid log(0)
        """
        self.epsilon = epsilon

    def compute(
        self,
        observed: dict[str, NDArray[np.floating]],
        predicted: dict[str, NDArray[np.floating]],
        weights: dict[str, NDArray[np.floating]] | None = None,
    ) -> LossResult:
        per_variable = {}
        per_point_list = []
        total = 0.0

        for var_name, obs in observed.items():
            if var_name not in predicted:
                continue

            pred = predicted[var_name]

            pred_safe = np.maximum(pred, self.epsilon)
            obs_safe = np.maximum(obs, self.epsilon)

            log_lik = obs * np.log(pred_safe) - pred_safe - gammaln(obs_safe + 1)

            neg_log_lik = -2.0 * log_lik

            if weights is not None and var_name in weights:
                neg_log_lik *= weights[var_name]

            var_loss = float(np.sum(neg_log_lik))
            per_variable[var_name] = var_loss
            total += var_loss

            per_point_list.append(neg_log_lik)

        per_point = np.concatenate(per_point_list) if per_point_list else np.array([])

        return LossResult(
            value=total,
            per_point=per_point,
            per_variable=per_variable,
        )


class NegativeBinomialLikelihood(LossFunction):
    """
    Negative binomial log-likelihood for overdispersed count data.

    L = -2 * Σ [log Γ(y + r) - log Γ(y + 1) - log Γ(r) + r*log(r/(r+μ)) + y*log(μ/(r+μ))]

    Where r is the dispersion parameter (smaller r = more overdispersion).
    """

    def __init__(self, dispersion: float = 1.0, epsilon: float = 1e-10):
        """
        Args:
            dispersion: Dispersion parameter r (inverse of overdispersion)
            epsilon: Small value to avoid numerical issues
        """
        if dispersion <= 0:
            raise ValueError("Dispersion parameter must be positive")
        self.dispersion = dispersion
        self.epsilon = epsilon

    def compute(
        self,
        observed: dict[str, NDArray[np.floating]],
        predicted: dict[str, NDArray[np.floating]],
        weights: dict[str, NDArray[np.floating]] | None = None,
    ) -> LossResult:
        per_variable = {}
        per_point_list = []
        total = 0.0
        r = self.dispersion

        for var_name, obs in observed.items():
            if var_name not in predicted:
                continue

            pred = predicted[var_name]
            mu = np.maximum(pred, self.epsilon)

            log_lik = (
                gammaln(obs + r)
                - gammaln(obs + 1)
                - gammaln(r)
                + r * np.log(r / (r + mu))
                + obs * np.log(mu / (r + mu))
            )

            neg_log_lik = -2.0 * log_lik

            if weights is not None and var_name in weights:
                neg_log_lik *= weights[var_name]

            var_loss = float(np.sum(neg_log_lik))
            per_variable[var_name] = var_loss
            total += var_loss

            per_point_list.append(neg_log_lik)

        per_point = np.concatenate(per_point_list) if per_point_list else np.array([])

        return LossResult(
            value=total,
            per_point=per_point,
            per_variable=per_variable,
        )


class NormalLikelihood(LossFunction):
    """
    Negative normal log-likelihood.

    L = Σ [(y_obs - y_pred)² / (2σ²) + log(σ) + 0.5*log(2π)]

    Can estimate σ from residuals or use provided values.
    """

    def __init__(
        self,
        sigma: float | None = None,
        estimate_sigma: bool = True,
    ):
        """
        Args:
            sigma: Fixed standard deviation. If None and estimate_sigma is True,
                   will be estimated from residuals.
            estimate_sigma: Whether to estimate sigma from residuals
        """
        self.sigma = sigma
        self.estimate_sigma = estimate_sigma

    def compute(
        self,
        observed: dict[str, NDArray[np.floating]],
        predicted: dict[str, NDArray[np.floating]],
        weights: dict[str, NDArray[np.floating]] | None = None,
    ) -> LossResult:
        per_variable = {}
        per_point_list = []
        total = 0.0

        all_residuals = []
        for var_name, obs in observed.items():
            if var_name not in predicted:
                continue
            pred = predicted[var_name]
            all_residuals.append(obs - pred)

        if self.estimate_sigma and self.sigma is None and all_residuals:
            all_res = np.concatenate(all_residuals)
            sigma = float(np.std(all_res, ddof=1))
            if sigma < 1e-10:
                sigma = 1e-10
        elif self.sigma is not None:
            sigma = self.sigma
        else:
            sigma = 1.0

        for var_name, obs in observed.items():
            if var_name not in predicted:
                continue

            pred = predicted[var_name]
            residuals = obs - pred

            nll = (residuals**2) / (2 * sigma**2) + np.log(sigma) + 0.5 * np.log(2 * np.pi)

            if weights is not None and var_name in weights:
                nll *= weights[var_name]

            var_loss = float(np.sum(nll))
            per_variable[var_name] = var_loss
            total += var_loss

            per_point_list.append(nll)

        per_point = np.concatenate(per_point_list) if per_point_list else np.array([])

        return LossResult(
            value=total,
            per_point=per_point,
            per_variable=per_variable,
        )


class LogLikelihood(LossFunction):
    """
    Generic log-likelihood wrapper using a user-provided distribution.

    The user provides a function that computes the log-likelihood
    for each observation given the prediction.
    """

    def __init__(
        self,
        log_likelihood_fn: Callable[
            [NDArray[np.floating], NDArray[np.floating]], NDArray[np.floating]
        ],
    ):
        """
        Args:
            log_likelihood_fn: Function(observed, predicted) -> log_likelihood per point
        """
        self.log_likelihood_fn = log_likelihood_fn

    def compute(
        self,
        observed: dict[str, NDArray[np.floating]],
        predicted: dict[str, NDArray[np.floating]],
        weights: dict[str, NDArray[np.floating]] | None = None,
    ) -> LossResult:
        per_variable = {}
        per_point_list = []
        total = 0.0

        for var_name, obs in observed.items():
            if var_name not in predicted:
                continue

            pred = predicted[var_name]

            log_lik = self.log_likelihood_fn(obs, pred)
            neg_log_lik = -2.0 * log_lik

            if weights is not None and var_name in weights:
                neg_log_lik *= weights[var_name]

            var_loss = float(np.sum(neg_log_lik))
            per_variable[var_name] = var_loss
            total += var_loss

            per_point_list.append(neg_log_lik)

        per_point = np.concatenate(per_point_list) if per_point_list else np.array([])

        return LossResult(
            value=total,
            per_point=per_point,
            per_variable=per_variable,
        )


class CustomLoss(LossFunction):
    """
    Custom loss function defined by a user-provided callable.
    """

    def __init__(
        self,
        loss_fn: Callable[
            [dict[str, NDArray[np.floating]], dict[str, NDArray[np.floating]]], float
        ],
    ):
        """
        Args:
            loss_fn: Function(observed_dict, predicted_dict) -> total_loss
        """
        self.loss_fn = loss_fn

    def compute(
        self,
        observed: dict[str, NDArray[np.floating]],
        predicted: dict[str, NDArray[np.floating]],
        weights: dict[str, NDArray[np.floating]] | None = None,
    ) -> LossResult:
        total = self.loss_fn(observed, predicted)
        return LossResult(value=total)


class HuberLoss(LossFunction):
    """
    Huber loss - robust to outliers.

    L = 0.5 * (y - ŷ)²  if |y - ŷ| ≤ δ
    L = δ * (|y - ŷ| - 0.5*δ)  otherwise
    """

    def __init__(self, delta: float = 1.0):
        """
        Args:
            delta: Threshold for switching between quadratic and linear
        """
        if delta <= 0:
            raise ValueError("delta must be positive")
        self.delta = delta

    def compute(
        self,
        observed: dict[str, NDArray[np.floating]],
        predicted: dict[str, NDArray[np.floating]],
        weights: dict[str, NDArray[np.floating]] | None = None,
    ) -> LossResult:
        per_variable = {}
        per_point_list = []
        total = 0.0
        delta = self.delta

        for var_name, obs in observed.items():
            if var_name not in predicted:
                continue

            pred = predicted[var_name]
            residuals = np.abs(obs - pred)

            huber = np.where(
                residuals <= delta,
                0.5 * residuals**2,
                delta * (residuals - 0.5 * delta),
            )

            if weights is not None and var_name in weights:
                huber *= weights[var_name]

            var_loss = float(np.sum(huber))
            per_variable[var_name] = var_loss
            total += var_loss

            per_point_list.append(huber)

        per_point = np.concatenate(per_point_list) if per_point_list else np.array([])

        return LossResult(
            value=total,
            per_point=per_point,
            per_variable=per_variable,
        )


__all__ = [
    "LossResult",
    "LossFunction",
    "SumOfSquaredErrors",
    "WeightedSSE",
    "PoissonLikelihood",
    "NegativeBinomialLikelihood",
    "NormalLikelihood",
    "LogLikelihood",
    "CustomLoss",
    "HuberLoss",
]

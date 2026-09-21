"""
Bayesian parameter inference for epidemic models.

Provides :func:`fit_model_bayesian`, a black-box Bayesian sampler based on
Differential Evolution Markov Chain (DE-MCMC; ter Braak, 2006). Because it
only requires likelihood *values* (no gradients), it works with any
simulation-based model — including models solved with scipy — without
requiring JAX-traceable code.

Example:
    >>> from epimodels.continuous import SIR
    >>> from epimodels.fitting import Dataset, ParameterSpec
    >>> from epimodels.fitting.bayes import fit_model_bayesian
    >>>
    >>> model = SIR()
    >>> dataset = Dataset(model).register(
    ...     name="cases", values=observed_I, times=times, state_variable="I")
    >>>
    >>> result = fit_model_bayesian(
    ...     model, dataset,
    ...     parameters_to_fit=[
    ...         ParameterSpec("beta", bounds=(0.1, 5.0)),
    ...         ParameterSpec("gamma", bounds=(0.01, 1.0)),
    ...     ],
    ...     total_population=10000,
    ...     likelihood="normal", num_samples=1000, num_warmup=500,
    ... )
    >>> result.samples["beta"]  # posterior draws
    >>> result.map_estimate()   # maximum a posteriori parameters
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from epimodels.fitting.base import ModelFitter, ParameterSpec
from epimodels.fitting.data import Dataset

if TYPE_CHECKING:
    from epimodels import BaseModel

__all__ = ["BayesianFitResult", "fit_model_bayesian"]


# ---------------------------------------------------------------------------
# Likelihoods
# ---------------------------------------------------------------------------

def _normal_logpdf_sum(
    observed: NDArray[np.floating],
    predicted: NDArray[np.floating],
    sigma: float,
) -> float:
    return float(
        np.sum(
            -0.5 * ((observed - predicted) / sigma) ** 2
            - np.log(sigma * np.sqrt(2.0 * np.pi))
        )
    )


def _poisson_loglik_sum(
    observed: NDArray[np.floating],
    predicted: NDArray[np.floating],
) -> float:
    pred = np.clip(predicted, 1e-10, None)
    from scipy.special import gammaln

    return float(
        np.sum(observed * np.log(pred) - pred - gammaln(observed + 1.0))
    )


def _negbin_loglik_sum(
    observed: NDArray[np.floating],
    predicted: NDArray[np.floating],
    phi: float,
) -> float:
    """Negative binomial (mean-dispersion parametrization), phi = 1/size."""
    from scipy.special import gammaln

    mean = np.clip(predicted, 1e-10, None)
    size = 1.0 / max(phi, 1e-10)
    p = size / (size + mean)
    return float(
        np.sum(
            gammaln(observed + size)
            - gammaln(size)
            - gammaln(observed + 1.0)
            + size * np.log(p)
            + observed * np.log1p(-p)
        )
    )


def _make_log_likelihood(
    likelihood: str,
    observed: dict[str, NDArray[np.floating]],
    sigma: float | dict[str, float],
) -> Callable[[dict[str, NDArray[np.floating]]], float]:
    """Build a log-likelihood over predicted/observed dicts keyed by variable."""
    if likelihood == "normal":

        def sigma_for(var: str) -> float:
            if isinstance(sigma, dict):
                return float(sigma.get(var, 1.0))
            return float(sigma)

        def loglik(pred: dict[str, NDArray[np.floating]]) -> float:
            return sum(
                _normal_logpdf_sum(observed[var], pred[var], sigma_for(var))
                for var in observed
            )

        return loglik

    if likelihood == "poisson":
        return lambda pred: sum(
            _poisson_loglik_sum(observed[var], pred[var]) for var in observed
        )

    if likelihood == "negative_binomial":
        phi_map = sigma if isinstance(sigma, dict) else {}
        return lambda pred: sum(
            _negbin_loglik_sum(observed[var], pred[var], phi_map.get(var, 0.1))
            for var in observed
        )

    raise ValueError(
        f"Unknown likelihood '{likelihood}'. "
        "Options: 'normal', 'poisson', 'negative_binomial'"
    )


# ---------------------------------------------------------------------------
# DE-MCMC sampler
# ---------------------------------------------------------------------------

def _reflect(value: float, lower: float, upper: float) -> float:
    """Reflect a value into [lower, upper]."""
    span = upper - lower
    if span <= 0:
        return lower
    v = value
    # Reflect iteratively until inside bounds (usually 0-2 iterations).
    for _ in range(100):
        if v < lower:
            v = lower + (lower - v)
        elif v > upper:
            v = upper - (v - upper)
        else:
            return float(v)
    return float(np.clip(v, lower, upper))


def _run_demcmc(
    log_posterior: Callable[[NDArray[np.floating]], float],
    initial_population: NDArray[np.floating],
    bounds: list[tuple[float, float]],
    num_warmup: int,
    num_samples: int,
    rng: np.random.Generator,
    thinning: int = 1,
) -> tuple[NDArray[np.floating], NDArray[np.floating], float]:
    """
    Differential Evolution Metropolis sampler with a population of chains.

    Returns:
        (samples, log_probs, acceptance_rate) where samples has shape
        ``(num_samples, n_chains, n_params)``.
    """
    population = initial_population.copy()
    n_chains, n_params = population.shape
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])

    log_probs = np.array([log_posterior(x) for x in population])
    n_evals = n_chains

    total_iterations = num_warmup + num_samples * thinning
    kept: list[NDArray[np.floating]] = []
    kept_logp: list[NDArray[np.floating]] = []
    accepted = 0
    proposed = 0

    for iteration in range(total_iterations):
        for i in range(n_chains):
            # Choose two distinct donor chains
            candidates = [j for j in range(n_chains) if j != i]
            a, b = rng.choice(candidates, size=2, replace=False)

            # Snooker-style gamma: mixture of large and small steps
            gamma = 1.0 if rng.random() < 0.5 else rng.uniform(0.2, 0.6)
            noise = rng.normal(0.0, 1e-6, size=n_params)

            proposal = population[i] + gamma * (population[a] - population[b]) + noise
            proposal = np.array(
                [_reflect(proposal[k], lower[k], upper[k]) for k in range(n_params)]
            )

            log_post = log_posterior(proposal)
            n_evals += 1
            proposed += 1

            log_alpha = log_post - log_probs[i]
            if np.log(rng.random()) < log_alpha:
                population[i] = proposal
                log_probs[i] = log_post
                accepted += 1

        if iteration >= num_warmup and (iteration - num_warmup) % thinning == 0:
            kept.append(population.copy())
            kept_logp.append(log_probs.copy())

    samples = np.array(kept)  # (num_samples, n_chains, n_params)
    logp = np.array(kept_logp)
    return samples, logp, accepted / max(proposed, 1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class BayesianFitResult:
    """Result of Bayesian inference."""

    samples: dict[str, NDArray[np.floating]]
    log_probs: NDArray[np.floating]
    acceptance_rate: float
    likelihood: str
    observed: dict[str, NDArray[np.floating]]
    model: BaseModel | None = None
    _param_order: list[str] = field(default_factory=list)

    @property
    def n_draws(self) -> int:
        """Total number of posterior draws (chains combined)."""
        return len(next(iter(self.samples.values())))

    def map_estimate(self) -> dict[str, float]:
        """Maximum a posteriori parameter values."""
        idx = int(np.argmax(self.log_probs))
        return {name: float(values[idx]) for name, values in self.samples.items()}

    def credible_intervals(
        self, ci: float = 0.95
    ) -> dict[str, tuple[float, float]]:
        """Equal-tailed credible intervals for each parameter."""
        alpha = (1 - ci) / 2
        return {
            name: (
                float(np.quantile(values, alpha)),
                float(np.quantile(values, 1 - alpha)),
            )
            for name, values in self.samples.items()
        }

    def summary(self) -> dict[str, dict[str, float]]:
        """Posterior summary statistics per parameter."""
        return {
            name: {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "q025": float(np.quantile(values, 0.025)),
                "median": float(np.quantile(values, 0.5)),
                "q975": float(np.quantile(values, 0.975)),
            }
            for name, values in self.samples.items()
        }

    def trace_plot(self, ax: Any = None, **kwargs) -> Any:
        """Plot posterior trace/density per parameter (requires matplotlib)."""
        import matplotlib.pyplot as plt

        names = list(self.samples.keys())
        n = len(names)
        if ax is None:
            fig, ax = plt.subplots(n, 2, figsize=(10, 2.5 * n), squeeze=False)

        for i, name in enumerate(names):
            draws_flat = self.samples[name].reshape(-1)
            ax[i][0].plot(draws_flat, linewidth=0.5)
            ax[i][0].set_ylabel(name)
            ax[i][1].hist(draws_flat, bins=40, **kwargs)
            ax[i][1].set_yticks([])
        return ax

    def to_inference_data(self) -> Any:
        """Convert to an ArviZ InferenceData (requires arviz)."""
        try:
            import arviz as az  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "arviz is required for to_inference_data(). "
                "Install with: pip install arviz"
            ) from e

        posterior = {
            name: values[:, None, :]  # (draws, chains=1, params)
            for name, values in self.samples.items()
        }
        return az.from_dict(posterior=posterior, sample_stats={"lp": self.log_probs[:, None, :]})


def fit_model_bayesian(
    model: BaseModel,
    dataset: Dataset,
    parameters_to_fit: list[ParameterSpec],
    total_population: float,
    likelihood: str = "normal",
    sigma: float | dict[str, float] = 1.0,
    num_samples: int = 1000,
    num_warmup: int = 500,
    thinning: int = 1,
    pop_size: int | None = None,
    seed: int | None = None,
    fixed_params: dict[str, float] | None = None,
    time_offset: float = 0.0,
    attach_fitted_model: bool = False,
) -> BayesianFitResult:
    """
    Bayesian parameter inference via DE-MCMC.

    Uses the same simulation machinery as :class:`ModelFitter` (scipy-based,
    black-box friendly) — no gradients or JAX-traceable code required.

    Args:
        model: Model instance to fit (not modified).
        dataset: Dataset with observed series.
        parameters_to_fit: ParameterSpecs defining names and bounds.
        total_population: Total population size.
        likelihood: Observation model — "normal", "poisson" or
            "negative_binomial".
        sigma: For "normal": standard deviation (scalar or per-variable dict).
            For "negative_binomial": dispersion phi = 1/size (scalar or dict).
            Ignored for "poisson".
        num_samples: Posterior draws per chain.
        num_warmup: Burn-in iterations.
        thinning: Keep every ``thinning``-th post-warmup draw.
        pop_size: Number of parallel chains (default: max(2 * n_params, 8)).
        seed: RNG seed for reproducibility.
        fixed_params: Fixed (non-fitted) parameter values.
        time_offset: Time offset applied to simulation times.
        attach_fitted_model: If True, simulate once at the MAP estimate and
            attach the resulting model to ``result.model``.

    Returns:
        BayesianFitResult with posterior samples and summaries.
    """
    if num_samples < 1 or num_warmup < 0:
        raise ValueError("num_samples must be >= 1 and num_warmup >= 0")
    if not parameters_to_fit:
        raise ValueError("parameters_to_fit must not be empty")

    # Reuse ModelFitter's simulation + interpolation machinery.
    fitter = ModelFitter(
        model=model,
        dataset=dataset,
        parameters_to_fit=parameters_to_fit,
        total_population=total_population,
        fixed_params=fixed_params,
        time_offset=time_offset,
    )
    param_names = list(fitter._param_order)
    bounds = [fitter.parameters_to_fit[name].bounds for name in param_names]

    observed: dict[str, NDArray[np.floating]] = {}
    for series in fitter.dataset.series.values():
        observed[series.state_variable] = np.asarray(series.values, dtype=float)

    log_likelihood = _make_log_likelihood(likelihood, observed, sigma)

    def log_posterior(theta: NDArray[np.floating]) -> float:
        params_dict, _ = fitter._params_array_to_dict(np.asarray(theta, dtype=float))
        try:
            predictions, model_times = fitter._run_model(params_dict)
        except Exception:
            return -np.inf
        interpolated = fitter._interpolate_predictions(predictions, model_times)
        # All observed variables must be present in predictions
        if not all(var in interpolated for var in observed):
            return -np.inf
        return log_likelihood(interpolated)

    rng = np.random.default_rng(seed)
    n_params = len(param_names)
    n_chains = pop_size if pop_size is not None else max(2 * n_params, 8)

    # Initialize population uniformly within bounds
    initial_population = np.empty((n_chains, n_params))
    for k, (low, high) in enumerate(bounds):
        initial_population[:, k] = rng.uniform(low, high, size=n_chains)

    samples, logp, acceptance = _run_demcmc(
        log_posterior,
        initial_population,
        bounds,
        num_warmup=num_warmup,
        num_samples=num_samples,
        rng=rng,
        thinning=thinning,
    )

    # samples: (n_draws, n_chains, n_params) -> per-param flat dict
    n_draws = samples.shape[0]
    posterior: dict[str, NDArray[np.floating]] = {
        name: samples[:, :, k].reshape(n_draws * n_chains)
        for k, name in enumerate(param_names)
    }

    result = BayesianFitResult(
        samples=posterior,
        log_probs=logp.reshape(n_draws * n_chains),
        acceptance_rate=float(acceptance),
        likelihood=likelihood,
        observed=observed,
        _param_order=param_names,
    )

    if attach_fitted_model:
        theta_map = np.array([result.map_estimate()[n] for n in param_names])
        params_dict, _ = fitter._params_array_to_dict(theta_map)
        time_range = fitter.dataset.time_range
        if time_range is None:
            raise FittingError("No time range available in dataset")
        fitted_model = model.copy()
        fitted_model(
            inits=fitter._estimate_initial_conditions(),
            trange=list(time_range),
            totpop=total_population,
            params={**(fixed_params or {}), **params_dict},
        )
        result.model = fitted_model

    return result

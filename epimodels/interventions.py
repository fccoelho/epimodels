"""
Intervention scenarios for continuous epidemic models.

An :class:`Intervention` modifies a parameter over a time interval
(e.g. reduce ``beta`` by 60% during a lockdown). A :class:`Scenario`
bundles a model with parameters, initial conditions and a set of
interventions. :class:`ScenarioComparison` runs a baseline and several
scenarios and plots them together.

Example:
    >>> from epimodels.continuous import SIR
    >>> from epimodels.interventions import Intervention, Scenario, ScenarioComparison
    >>>
    >>> model = SIR()
    >>> base = Scenario("no-intervention", model,
    ...     params={"beta": 2.0, "gamma": 0.1},
    ...     initial_conditions=[1000, 1, 0], trange=[0, 100], totpop=1001)
    >>> lockdown = Scenario("lockdown", model,
    ...     params={"beta": 2.0, "gamma": 0.1},
    ...     initial_conditions=[1000, 1, 0], trange=[0, 100], totpop=1001,
    ...     interventions=[Intervention("beta", start=10, end=40, factor=0.4)])
    >>> cmp = ScenarioComparison(base, [lockdown]).run()
    >>> cmp.plot("I")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from epimodels.continuous import ContinuousModel

import numpy as np

__all__ = ["Intervention", "Scenario", "ScenarioComparison"]


@dataclass
class Intervention:
    """
    Time-bounded modification of a model parameter.

    Exactly one of ``factor`` (multiplicative change applied to the base
    parameter value) or ``new_value`` (absolute override) must be given.

    Attributes:
        parameter: Name of the parameter to modify.
        start: Intervention start time (inclusive).
        end: Intervention end time (exclusive); None = until the end.
        factor: Multiplier applied to the base value during the window.
        new_value: Absolute value used during the window.
    """

    parameter: str
    start: float
    end: float | None = None
    factor: float | None = None
    new_value: float | None = None

    def __post_init__(self) -> None:
        if (self.factor is None) == (self.new_value is None):
            raise ValueError(
                f"Intervention for '{self.parameter}' must define exactly one of "
                "'factor' or 'new_value'"
            )
        if self.end is not None and self.end <= self.start:
            raise ValueError(
                f"Intervention for '{self.parameter}': end ({self.end}) must be "
                f"greater than start ({self.start})"
            )

    def applies_at(self, t: float) -> bool:
        """Whether the intervention is active at time t."""
        return t >= self.start and (self.end is None or t < self.end)


@dataclass
class Scenario:
    """
    A simulation scenario: model + parameters + interventions.

    Args:
        name: Scenario label.
        model: A ContinuousModel instance (not modified; a copy is used).
        params: Base parameter dict.
        initial_conditions: Initial state values.
        trange: Time range [t0, tf].
        totpop: Total population.
        interventions: List of Intervention objects applied on top of params.
    """

    name: str
    model: "ContinuousModel"
    params: dict[str, Any]
    initial_conditions: list[float]
    trange: list[float]
    totpop: float
    interventions: list[Intervention] = field(default_factory=list)

    def __post_init__(self) -> None:
        from epimodels.continuous import ContinuousModel

        if not isinstance(self.model, ContinuousModel):
            raise TypeError(
                "Interventions are only supported for ContinuousModel instances, "
                f"got {type(self.model).__name__}"
            )

        unknown = [iv.parameter for iv in self.interventions
                   if iv.parameter not in self.params]
        if unknown:
            raise KeyError(
                f"Interventions reference parameters not in params: {sorted(set(unknown))}"
            )

        self._traces: dict[str, Any] | None = None

    def effective_params(self, t: float) -> dict[str, Any]:
        """Base parameters with all interventions applied at time t."""
        effective = dict(self.params)
        for iv in self.interventions:
            if iv.applies_at(t) and iv.parameter in effective:
                base = effective[iv.parameter]
                if isinstance(base, (int, float, np.floating)):
                    effective[iv.parameter] = (
                        base * iv.factor if iv.factor is not None else iv.new_value
                    )
        return effective

    def run(self, validate: bool = True, **solver_kwargs) -> dict[str, Any]:
        """
        Run the scenario and return its traces.

        Time-dependence is injected by wrapping the model's ``_model`` so
        that parameters are re-evaluated at each integration step.
        """
        model = self.model.copy()

        if self.interventions:
            base_model_fn = model._model
            interventions = list(self.interventions)

            def _model_with_interventions(t, y, params):
                effective = dict(params)
                for iv in interventions:
                    if iv.applies_at(t) and iv.parameter in effective:
                        base = effective[iv.parameter]
                        if isinstance(base, (int, float, np.floating)):
                            effective[iv.parameter] = (
                                base * iv.factor
                                if iv.factor is not None
                                else iv.new_value
                            )
                return base_model_fn(t, y, effective)

            model._model = _model_with_interventions

        model(
            self.initial_conditions,
            self.trange,
            self.totpop,
            self.params,
            validate=validate,
            **solver_kwargs,
        )
        self._traces = model.traces
        return self._traces

    @property
    def traces(self) -> dict[str, Any]:
        """Traces from the last :meth:`run` (raises if not run yet)."""
        if self._traces is None:
            raise ValueError(f"Scenario '{self.name}' has not been run yet.")
        return self._traces


class ScenarioComparison:
    """
    Run a baseline scenario plus a set of alternative scenarios together.

    Example:
        >>> cmp = ScenarioComparison(baseline, [lockdown, masks]).run()
        >>> cmp.final_size("I")
        >>> cmp.plot("I")
    """

    def __init__(self, baseline: Scenario, scenarios: list[Scenario] | None = None):
        self.baseline = baseline
        self.scenarios: list[Scenario] = list(scenarios or [])
        self._ran = False

    def add(self, scenario: Scenario) -> ScenarioComparison:
        """Add a scenario to the comparison."""
        self.scenarios.append(scenario)
        self._ran = False
        return self

    def run(self, validate: bool = True, **solver_kwargs) -> ScenarioComparison:
        """Run baseline and all scenarios."""
        self.baseline.run(validate=validate, **solver_kwargs)
        for scenario in self.scenarios:
            scenario.run(validate=validate, **solver_kwargs)
        self._ran = True
        return self

    def _require_run(self) -> None:
        if not self._ran:
            raise ValueError("Call run() before extracting results.")

    def scenarios_traces(self) -> dict[str, dict[str, Any]]:
        """Dict mapping scenario name to its traces (baseline included)."""
        self._require_run()
        out = {self.baseline.name: self.baseline.traces}
        for scenario in self.scenarios:
            out[scenario.name] = scenario.traces
        return out

    def final_size(self, var: str = "R") -> dict[str, float]:
        """Final value of a variable per scenario."""
        self._require_run()
        return {
            name: float(traces[var][-1]) for name, traces in self.scenarios_traces().items()
        }

    def peak(self, var: str = "I") -> dict[str, float]:
        """Peak value of a variable per scenario."""
        self._require_run()
        return {
            name: float(np.max(traces[var]))
            for name, traces in self.scenarios_traces().items()
        }

    def plot(self, var: str = "I", ax: Any = None, **plot_kwargs) -> Any:
        """Plot a variable across all scenarios on one axes."""
        import matplotlib.pyplot as plt

        self._require_run()

        if ax is None:
            _, ax = plt.subplots(figsize=(9, 5))

        for name, traces in self.scenarios_traces().items():
            kwargs = dict(plot_kwargs)
            kwargs.setdefault("label", name)
            ax.plot(traces["time"], traces[var], **kwargs)

        ax.set_xlabel("Time")
        ax.set_ylabel(var)
        ax.set_title(f"{var}: scenario comparison")
        ax.legend(loc=0)
        return ax

"""Tests for intervention scenarios."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from epimodels.continuous import SIR
from epimodels.interventions import Intervention, Scenario, ScenarioComparison


@pytest.fixture
def sir():
    return SIR()


def make_scenario(model, name="scenario", **kwargs):
    return Scenario(
        name=name,
        model=model,
        params={"beta": 2.0, "gamma": 0.1},
        initial_conditions=[1000, 1, 0],
        trange=[0, 100],
        totpop=1001,
        **kwargs,
    )


class TestIntervention:
    def test_factor_and_new_value_mutually_exclusive(self):
        with pytest.raises(ValueError, match="exactly one"):
            Intervention("beta", start=0, factor=0.5, new_value=1.0)

    def test_requires_factor_or_new_value(self):
        with pytest.raises(ValueError, match="exactly one"):
            Intervention("beta", start=0)

    def test_end_after_start(self):
        with pytest.raises(ValueError, match="end"):
            Intervention("beta", start=10, end=5, factor=0.5)

    def test_applies_at(self):
        iv = Intervention("beta", start=10, end=40, factor=0.5)
        assert not iv.applies_at(9.9)
        assert iv.applies_at(10.0)
        assert iv.applies_at(39.9)
        assert not iv.applies_at(40.0)

        open_ended = Intervention("beta", start=10, factor=0.5)
        assert open_ended.applies_at(1e6)


class TestScenario:
    def test_unknown_parameter_rejected(self, sir):
        with pytest.raises(KeyError, match="not in params"):
            make_scenario(
                sir, interventions=[Intervention("kappa", start=0, factor=0.5)]
            )

    def test_effective_params(self, sir):
        scenario = make_scenario(
            sir, interventions=[Intervention("beta", start=10, end=40, factor=0.4)]
        )
        assert scenario.effective_params(5)["beta"] == 2.0
        assert scenario.effective_params(20)["beta"] == pytest.approx(0.8)
        assert scenario.effective_params(50)["beta"] == 2.0

    def test_run_returns_traces(self, sir):
        scenario = make_scenario(sir)
        traces = scenario.run()
        assert "I" in traces
        assert "time" in traces

    def test_intervention_flattens_peak(self, sir):
        baseline = make_scenario(sir, name="baseline")
        mitigated = make_scenario(
            sir,
            name="mitigated",
            interventions=[Intervention("beta", start=1, end=30, factor=0.4)],
        )
        baseline.run()
        mitigated.run()
        assert (
            np.max(mitigated.traces["I"]) < np.max(baseline.traces["I"]) * 0.9
        ), "60% contact reduction at early stage should flatten the peak"

    def test_traces_before_run_raises(self, sir):
        scenario = make_scenario(sir)
        with pytest.raises(ValueError, match="not been run"):
            scenario.traces

    def test_original_model_untouched(self, sir):
        scenario = make_scenario(
            sir, interventions=[Intervention("beta", start=10, factor=0.4)]
        )
        scenario.run()
        assert sir.traces == {}
        # The original _model must be intact for other instances
        fresh = SIR()
        fresh([1000, 1, 0], [0, 10], 1001, {"beta": 2, "gamma": 0.1})
        assert "I" in fresh.traces


class TestScenarioComparison:
    def test_run_and_metrics(self, sir):
        baseline = make_scenario(sir, name="baseline")
        lockdown = make_scenario(
            sir,
            name="lockdown",
            interventions=[Intervention("beta", start=1, end=60, factor=0.4)],
        )
        cmp = ScenarioComparison(baseline, [lockdown]).run()
        traces = cmp.scenarios_traces()
        assert set(traces) == {"baseline", "lockdown"}

        sizes = cmp.final_size("R")
        assert sizes["lockdown"] < sizes["baseline"]

        peaks = cmp.peak("I")
        assert peaks["lockdown"] < peaks["baseline"]

    def test_requires_run(self, sir):
        cmp = ScenarioComparison(make_scenario(sir))
        with pytest.raises(ValueError, match="run"):
            cmp.final_size()

    def test_plot(self, sir):
        baseline = make_scenario(sir, name="baseline")
        lockdown = make_scenario(
            sir,
            name="lockdown",
            interventions=[Intervention("beta", start=1, end=60, factor=0.4)],
        )
        cmp = ScenarioComparison(baseline, [lockdown]).run()
        ax = cmp.plot("I")
        assert ax is not None
        plt.close("all")

    def test_discrete_model_rejected(self):
        from epimodels.discrete import SIR as DiscreteSIR

        with pytest.raises(TypeError, match="ContinuousModel"):
            make_scenario(DiscreteSIR())

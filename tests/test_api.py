"""
Tests for new API methods: to_dataframe, to_dict, summary, copy, reset, R0.
"""

import copy
import pytest
import numpy as np
from epimodels import ValidationError
from epimodels.continuous import SIR, SIS, SEIR, SIRS, SIR1D
from epimodels.discrete import SIR as DiscreteSIR, SIS as DiscreteSIS, SEIR as DiscreteSEIR


class TestToDataframe:
    """Tests for to_dataframe method."""

    def test_sir_to_dataframe(self):
        """Should return DataFrame with correct columns."""
        model = SIR()
        model([1000, 1, 0], [0, 50], 1001, {"beta": 0.3, "gamma": 0.1})

        df = model.to_dataframe()
        assert "time" in df.columns
        assert "S" in df.columns
        assert "I" in df.columns
        assert "R" in df.columns
        assert len(df) > 0

    def test_empty_traces_raises(self):
        """Should raise ValueError when no simulation has been run."""
        model = SIR()
        with pytest.raises(ValueError, match="No simulation results"):
            model.to_dataframe()

    def test_discrete_sir_to_dataframe(self):
        """Should work with discrete models."""
        model = DiscreteSIR()
        model([1000, 1, 0], [0, 50], 1001, {"beta": 0.3, "gamma": 0.1})

        df = model.to_dataframe()
        assert len(df) == 50


class TestToDict:
    """Tests for to_dict method."""

    def test_returns_copy(self):
        """Should return a copy of traces."""
        model = SIR()
        model([1000, 1, 0], [0, 50], 1001, {"beta": 0.3, "gamma": 0.1})

        traces_copy = model.to_dict()
        assert traces_copy is not model.traces

    def test_original_unchanged(self):
        """Modifying copy should not affect original."""
        model = SIR()
        model([1000, 1, 0], [0, 50], 1001, {"beta": 0.3, "gamma": 0.1})

        traces_copy = model.to_dict()
        traces_copy["S"][0] = 9999

        assert model.traces["S"][0] != 9999

    def test_empty_model(self):
        """Should return empty dict for model without simulation."""
        model = SIR()
        traces = model.to_dict()
        assert traces == {}


class TestSummary:
    """Tests for summary method."""

    def test_sir_summary(self):
        """Should return summary statistics for SIR model."""
        model = SIR()
        model([1000, 1, 0], [0, 100], 1001, {"beta": 0.3, "gamma": 0.1})

        stats = model.summary()
        assert "peak_I" in stats
        assert "peak_time" in stats
        assert "final_S" in stats
        assert "final_R" in stats
        assert stats["peak_I"] > 1

    def test_contains_expected_keys(self):
        """Should contain model type and time bounds."""
        model = SIR()
        model([1000, 1, 0], [0, 50], 1001, {"beta": 0.3, "gamma": 0.1})

        stats = model.summary()
        assert "model" in stats
        assert "t_start" in stats
        assert "t_end" in stats

    def test_attack_rate(self):
        """Should calculate attack rate correctly."""
        model = SIR()
        model([1000, 1, 0], [0, 200], 1001, {"beta": 0.5, "gamma": 0.1})

        stats = model.summary()
        assert "attack_rate" in stats
        assert 0 < stats["attack_rate"] <= 1

    def test_empty_traces_raises(self):
        """Should raise ValueError when no simulation has been run."""
        model = SIR()
        with pytest.raises(ValueError, match="No simulation results"):
            model.summary()


class TestCopy:
    """Tests for copy method."""

    def test_copy_without_traces(self):
        """Should copy model without simulation results."""
        model = SIR()
        model([1000, 1, 0], [0, 50], 1001, {"beta": 0.3, "gamma": 0.1})

        new_model = model.copy(include_traces=False)
        assert new_model.traces == {}
        assert new_model.param_values == {}

    def test_copy_with_traces(self):
        """Should copy model with simulation results."""
        model = SIR()
        model([1000, 1, 0], [0, 50], 1001, {"beta": 0.3, "gamma": 0.1})

        new_model = model.copy(include_traces=True)
        assert len(new_model.traces) > 0
        assert len(new_model.param_values) > 0

    def test_independence(self):
        """Modifying copy should not affect original."""
        model = SIR()
        model([1000, 1, 0], [0, 50], 1001, {"beta": 0.3, "gamma": 0.1})

        new_model = model.copy(include_traces=True)
        new_model.traces["S"][0] = 9999

        assert model.traces["S"][0] != 9999

    def test_copy_preserves_model_type(self):
        """Should preserve model_type attribute."""
        model = SIR()
        new_model = model.copy()
        assert new_model.model_type == "SIR"


class TestReset:
    """Tests for reset method."""

    def test_clears_traces(self):
        """Should clear simulation results."""
        model = SIR()
        model([1000, 1, 0], [0, 50], 1001, {"beta": 0.3, "gamma": 0.1})

        model.reset()
        assert model.traces == {}

    def test_clears_param_values(self):
        """Should clear parameter values."""
        model = SIR()
        model([1000, 1, 0], [0, 50], 1001, {"beta": 0.3, "gamma": 0.1})

        model.reset()
        assert model.param_values == {}

    def test_can_run_again_after_reset(self):
        """Should be able to run simulation after reset."""
        model = SIR()
        model([1000, 1, 0], [0, 50], 1001, {"beta": 0.3, "gamma": 0.1})
        model.reset()
        model([500, 5, 0], [0, 30], 505, {"beta": 0.4, "gamma": 0.2})

        assert len(model.traces) > 0


class TestR0Continuous:
    """Tests for R0 property on continuous models."""

    def test_sir_r0(self):
        """Should calculate R0 correctly for SIR model."""
        model = SIR()
        model([1000, 1, 0], [0, 50], 1001, {"beta": 0.3, "gamma": 0.1})
        assert model.R0 == pytest.approx(3.0)

    def test_sis_r0(self):
        """Should calculate R0 correctly for SIS model."""
        model = SIS()
        model([1000, 1], [0, 50], 1001, {"beta": 0.5, "gamma": 0.25})
        assert model.R0 == pytest.approx(2.0)

    def test_seir_r0(self):
        """Should calculate R0 correctly for SEIR model."""
        model = SEIR()
        model([1000, 0, 1, 0], [0, 50], 1001, {"beta": 0.4, "gamma": 0.2, "epsilon": 0.1})
        assert model.R0 == pytest.approx(2.0)

    def test_sirs_r0(self):
        """Should calculate R0 correctly for SIRS model."""
        model = SIRS()
        model([1000, 1, 0], [0, 50], 1001, {"beta": 0.6, "gamma": 0.3, "xi": 0.05})
        assert model.R0 == pytest.approx(2.0)

    def test_sir1d_r0(self):
        """Should return R0 parameter for SIR1D model."""
        model = SIR1D()
        model([0], [0, 50], 100, {"R0": 2.5, "gamma": 0.1, "S0": 98})
        assert model.R0 == pytest.approx(2.5)

    def test_none_before_run(self):
        """Should return None before model is run."""
        model = SIR()
        assert model.R0 is None


class TestR0Discrete:
    """Tests for R0 property on discrete models."""

    def test_discrete_sir_r0(self):
        """Should calculate R0 correctly for discrete SIR model."""
        model = DiscreteSIR()
        model([1000, 1, 0], [0, 50], 1001, {"beta": 0.3, "gamma": 0.1})
        assert model.R0 == pytest.approx(3.0)

    def test_discrete_sis_r0(self):
        """Should calculate R0 correctly for discrete SIS model."""
        model = DiscreteSIS()
        model([0, 1, 1000], [0, 50], 1001, {"beta": 0.5, "gamma": 0.25})
        assert model.R0 == pytest.approx(2.0)

    def test_discrete_seir_r0(self):
        """Should calculate R0 correctly for discrete SEIR model."""
        model = DiscreteSEIR()
        model([1000, 1, 1, 0], [0, 50], 1002, {"beta": 0.4, "r": 0.2, "e": 0.1, "b": 0, "alpha": 1})
        assert model.R0 == pytest.approx(2.0)

    def test_discrete_none_before_run(self):
        """Should return None before model is run."""
        model = DiscreteSIR()
        assert model.R0 is None


class TestR0EpidemicBehavior:
    """Tests linking R0 to epidemic behavior."""

    def test_r0_above_one_epidemic_grows(self):
        """When R0 > 1, epidemic should initially grow."""
        model = SIR()
        model([1000, 1, 0], [0, 20], 1001, {"beta": 0.5, "gamma": 0.1})  # R0 = 5

        I = model.traces["I"]
        assert I[5] > I[0]  # I should increase initially

    def test_r0_below_one_epidemic_dies(self):
        """When R0 < 1, epidemic should die out."""
        model = SIR()
        model([1000, 100, 0], [0, 100], 1100, {"beta": 0.05, "gamma": 0.1})  # R0 = 0.5

        I = model.traces["I"]
        assert I[-1] < I[0]  # I should decrease

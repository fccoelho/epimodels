"""
Tests for parameter and initial condition validation.
"""

import pytest
from epimodels import ValidationError
from epimodels.continuous import SIR, SIS, SEIR
from epimodels.discrete import SIR as DiscreteSIR, SIS as DiscreteSIS


class TestParameterValidation:
    """Tests for parameter validation."""

    def test_sir_missing_parameter(self):
        """Should raise ValidationError when required parameter is missing."""
        model = SIR()
        with pytest.raises(ValidationError, match="Missing required parameters"):
            model([1000, 1, 0], [0, 50], 1001, {"beta": 0.3}, validate=True)

    def test_sir_negative_parameter(self):
        """Should raise ValidationError when parameter is negative."""
        model = SIR()
        with pytest.raises(ValidationError, match="must be non-negative"):
            model(
                [1000, 1, 0], [0, 50], 1001, {"beta": -0.3, "gamma": 0.1}, validate=True
            )

    def test_sir_valid_parameters(self):
        """Should accept valid parameters."""
        model = SIR()
        model([1000, 1, 0], [0, 50], 1001, {"beta": 0.3, "gamma": 0.1}, validate=True)
        assert len(model.traces) == 4

    def test_sir_zero_parameters(self):
        """Should accept zero parameters (edge case)."""
        model = SIR()
        model([1000, 1, 0], [0, 50], 1001, {"beta": 0, "gamma": 0}, validate=True)
        assert len(model.traces) == 4


class TestInitialConditionValidation:
    """Tests for initial condition validation."""

    def test_sir_wrong_number_of_inits(self):
        """Should raise ValidationError when too few initial conditions."""
        model = SIR()
        with pytest.raises(
            ValidationError, match="Expected at least 3 initial conditions"
        ):
            model([1000, 1], [0, 50], 1001, {"beta": 0.3, "gamma": 0.1}, validate=True)

    def test_sir_negative_initial_condition(self):
        """Should raise ValidationError when initial condition is negative."""
        model = SIR()
        with pytest.raises(ValidationError, match="must be non-negative"):
            model(
                [1000, -1, 0], [0, 50], 1001, {"beta": 0.3, "gamma": 0.1}, validate=True
            )

    def test_sir_inits_exceed_population(self):
        """Should raise ValidationError when initial conditions exceed population."""
        model = SIR()
        with pytest.raises(ValidationError, match="exceeds total population"):
            model(
                [1000, 100, 0],
                [0, 50],
                1001,
                {"beta": 0.3, "gamma": 0.1},
                validate=True,
            )

    def test_sir_valid_initial_conditions(self):
        """Should accept valid initial conditions."""
        model = SIR()
        model([1000, 1, 0], [0, 50], 1001, {"beta": 0.3, "gamma": 0.1}, validate=True)
        assert len(model.traces) == 4


class TestValidationCanBeDisabled:
    """Tests to verify validation can be disabled."""

    def test_sir_validation_disabled_missing_param(self):
        """Should not validate when validate=False."""
        model = SIR()
        # This will fail at a different point (in solve_ivp) but not at validation
        # We just check it doesn't raise ValidationError immediately
        try:
            model([1000, 1, 0], [0, 50], 1001, {"beta": 0.3}, validate=False)
        except ValidationError:
            pytest.fail("ValidationError raised when validation disabled")
        except Exception:
            pass  # Other exceptions are expected

    def test_discrete_sir_validation(self):
        """Should validate discrete model parameters."""
        model = DiscreteSIR()
        with pytest.raises(ValidationError, match="Missing required parameters"):
            model([1000, 1, 0], [0, 50], 1001, {"beta": 0.3}, validate=True)


class TestParameterBoundaryValues:
    """Tests for boundary values of parameters."""

    def test_sir_large_beta(self):
        """Should handle large beta values."""
        model = SIR()
        model([1000, 1, 0], [0, 50], 1001, {"beta": 100, "gamma": 0.1}, validate=True)
        assert len(model.traces) == 4

    def test_sir_small_gamma(self):
        """Should handle very small gamma values."""
        model = SIR()
        model(
            [1000, 1, 0], [0, 50], 1001, {"beta": 0.3, "gamma": 0.0001}, validate=True
        )
        assert len(model.traces) == 4

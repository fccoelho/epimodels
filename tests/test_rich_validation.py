"""
Tests for rich parameter and state variable validation.
"""

import pytest
from epimodels import ValidationError
from epimodels.validation import (
    ParameterSpec,
    VariableSpec,
    ModelConstraint,
    DomainType,
    validate_parameter_value,
    validate_initial_condition,
    evaluate_constraint,
)


class TestParameterSpec:
    """Tests for ParameterSpec dataclass."""

    def test_basic_spec(self):
        """Should create basic parameter spec."""
        spec = ParameterSpec(name="beta", symbol=r"$\beta$", description="Transmission rate")
        assert spec.name == "beta"
        assert spec.required is True
        assert spec.bounds is None

    def test_spec_with_bounds(self):
        """Should create spec with bounds."""
        spec = ParameterSpec(name="gamma", symbol=r"$\gamma$", bounds=(0, None))
        assert spec.bounds == (0, None)

    def test_invalid_bounds(self):
        """Should raise error for invalid bounds."""
        with pytest.raises(ValueError, match="Invalid bounds"):
            ParameterSpec(name="test", symbol="test", bounds=(10, 5))

    def test_spec_with_constraints(self):
        """Should create spec with constraints."""
        spec = ParameterSpec(name="R0", symbol="R_0", constraints=["value > 1"])
        assert "value > 1" in spec.constraints


class TestVariableSpec:
    """Tests for VariableSpec dataclass."""

    def test_basic_spec(self):
        """Should create basic variable spec."""
        spec = VariableSpec(name="S", symbol="S", description="Susceptible")
        assert spec.name == "S"
        assert spec.non_negative is True

    def test_non_negative_sets_bounds(self):
        """Non-negative flag should set bounds."""
        spec = VariableSpec(name="I", symbol="I", non_negative=True)
        assert spec.bounds == (0.0, None)

    def test_negative_bounds_conflict(self):
        """Should raise error if non_negative but bounds allow negative."""
        with pytest.raises(ValueError, match="is marked non_negative"):
            VariableSpec(name="test", symbol="test", non_negative=True, bounds=(-5, 10))


class TestModelConstraint:
    """Tests for ModelConstraint dataclass."""

    def test_basic_constraint(self):
        """Should create basic constraint."""
        constraint = ModelConstraint(expression="beta > gamma")
        assert constraint.expression == "beta > gamma"
        assert constraint.severity == "error"

    def test_warning_constraint(self):
        """Should create warning constraint."""
        constraint = ModelConstraint(expression="R0 > 1", severity="warning")
        assert constraint.severity == "warning"

    def test_invalid_severity(self):
        """Should raise error for invalid severity."""
        with pytest.raises(ValueError, match="Invalid severity"):
            ModelConstraint(expression="x > 0", severity="critical")


class TestValidateParameterValue:
    """Tests for validate_parameter_value function."""

    def test_valid_value(self):
        """Should accept valid value."""
        spec = ParameterSpec(name="beta", symbol="β", bounds=(0, None))
        errors = validate_parameter_value("beta", 0.5, spec)
        assert len(errors) == 0

    def test_value_below_min(self):
        """Should detect value below minimum."""
        spec = ParameterSpec(name="beta", symbol="β", bounds=(0, None))
        errors = validate_parameter_value("beta", -0.5, spec)
        assert len(errors) == 1
        assert "below minimum" in errors[0]

    def test_value_above_max(self):
        """Should detect value above maximum."""
        spec = ParameterSpec(name="p", symbol="p", bounds=(0, 1))
        errors = validate_parameter_value("p", 1.5, spec)
        assert len(errors) == 1
        assert "exceeds maximum" in errors[0]

    def test_wrong_type(self):
        """Should detect wrong type."""
        spec = ParameterSpec(name="count", symbol="n", dtype=int)
        errors = validate_parameter_value("count", "not a number", spec)
        assert len(errors) == 1
        assert "wrong type" in errors[0]

    def test_int_as_float(self):
        """Should accept int when float expected."""
        spec = ParameterSpec(name="rate", symbol="r", dtype=float)
        errors = validate_parameter_value("rate", 5, spec)
        assert len(errors) == 0

    def test_constraint_violation(self):
        """Should detect constraint violation."""
        spec = ParameterSpec(name="gamma", symbol="γ", constraints=["value > 0"])
        errors = validate_parameter_value("gamma", 0, spec)
        assert len(errors) == 1
        assert "violates constraint" in errors[0]


class TestValidateInitialCondition:
    """Tests for validate_initial_condition function."""

    def test_valid_value(self):
        """Should accept valid initial condition."""
        spec = VariableSpec(name="S", symbol="S", non_negative=True)
        errors = validate_initial_condition("S", 100, spec)
        assert len(errors) == 0

    def test_negative_value(self):
        """Should detect negative value."""
        spec = VariableSpec(name="I", symbol="I", non_negative=True)
        errors = validate_initial_condition("I", -10, spec)
        assert len(errors) == 2
        assert "must be non-negative" in errors[0]

    def test_value_outside_bounds(self):
        """Should detect value outside bounds."""
        spec = VariableSpec(name="proportion", symbol="p", bounds=(0, 1))
        errors = validate_initial_condition("proportion", 1.5, spec)
        assert len(errors) == 1
        assert "exceeds maximum" in errors[0]


class TestEvaluateConstraint:
    """Tests for evaluate_constraint function."""

    def test_simple_comparison(self):
        """Should evaluate simple comparison."""
        satisfied, msg = evaluate_constraint("x > 5", {"x": 10})
        assert satisfied is True

    def test_comparison_violation(self):
        """Should detect comparison violation."""
        satisfied, msg = evaluate_constraint("x > 5", {"x": 3})
        assert satisfied is False

    def test_equality(self):
        """Should evaluate equality."""
        satisfied, msg = evaluate_constraint("x == y", {"x": 5, "y": 5})
        assert satisfied is True

    def test_arithmetic_expression(self):
        """Should evaluate arithmetic in constraint."""
        satisfied, msg = evaluate_constraint("beta / gamma > 1", {"beta": 2, "gamma": 1})
        assert satisfied is True

    def test_boolean_and(self):
        """Should evaluate boolean AND."""
        satisfied, msg = evaluate_constraint("x > 0 and y > 0", {"x": 1, "y": 1})
        assert satisfied is True

        satisfied, msg = evaluate_constraint("x > 0 and y > 0", {"x": 1, "y": -1})
        assert satisfied is False

    def test_boolean_or(self):
        """Should evaluate boolean OR."""
        satisfied, msg = evaluate_constraint("x > 5 or y > 5", {"x": 3, "y": 6})
        assert satisfied is True

    def test_complex_expression(self):
        """Should evaluate complex expression."""
        satisfied, msg = evaluate_constraint("p + q <= 1", {"p": 0.3, "q": 0.6})
        assert satisfied is True

    def test_invalid_expression(self):
        """Should handle invalid expression."""
        satisfied, msg = evaluate_constraint("x >>> 5", {"x": 10})
        assert satisfied is False
        assert "Failed to evaluate" in msg

    def test_unknown_variable(self):
        """Should handle unknown variable."""
        satisfied, msg = evaluate_constraint("z > 0", {"x": 10})
        assert satisfied is False
        assert "Unknown variable" in msg


class TestDomainType:
    """Tests for DomainType enum."""

    def test_domain_types(self):
        """Should have expected domain types."""
        assert DomainType.CONTINUOUS.value == "continuous"
        assert DomainType.DISCRETE.value == "discrete"
        assert DomainType.CATEGORICAL.value == "categorical"
        assert DomainType.INTEGER.value == "integer"

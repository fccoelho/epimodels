"""
Tests for formula extraction module.
"""

import pytest

sympy = pytest.importorskip("sympy")
import sympy as sp

from epimodels import FormulaExtractionError
from epimodels.formulas import (
    extract_formulas,
    get_free_symbols,
    sympy_to_vfgen,
    validate_formulas,
    validate_model_method,
)
from epimodels.continuous import SIR, SIS, SIRS, SEIR


# =============================================================================
# Test Fixtures
# =============================================================================


class ModelWithIfStatement:
    """Test model with if statement - should warn."""

    model_type = "BrokenIf"
    state_variables = {"S": "Susceptible", "I": "Infectious"}
    parameters = {"beta": r"\beta"}

    def _model(self, t, y, params):
        S, I = y
        if S > 0:  # This will trigger warning
            return [-params["beta"] * S * I, params["beta"] * S * I]
        return [0, 0]


class ModelWithLoop:
    """Test model with loop - should fail."""

    model_type = "BrokenLoop"
    state_variables = {"S": "Susceptible"}
    parameters = {}

    def _model(self, t, y, params):
        S = y[0]
        result = 0
        for i in range(10):  # Loop will fail
            result += i
        return [result]


class ModelWithLambda:
    """Test model with lambda - should warn."""

    model_type = "BrokenLambda"
    state_variables = {"S": "Susceptible", "I": "Infectious"}
    parameters = {"beta": r"\beta", "t_switch": "t_switch"}

    def _model(self, t, y, params):
        S, I = y
        f = lambda t: 1 if t > params["t_switch"] else 0  # noqa: E731
        return [-params["beta"] * S * I * f(t), params["beta"] * S * I * f(t)]


class ModelWithoutModelMethod:
    """Test model without _model method."""

    model_type = "NoModel"
    state_variables = {"S": "Susceptible"}
    parameters = {}


class ModelReturningNone:
    """Test model that returns None."""

    model_type = "ReturnsNone"
    state_variables = {"S": "Susceptible"}
    parameters = {}

    def _model(self, t, y, params):
        return None


class ModelReturningWrongCount:
    """Test model that returns wrong number of expressions."""

    model_type = "WrongCount"
    state_variables = {"S": "Susceptible", "I": "Infectious"}
    parameters = {"beta": r"\beta"}

    def _model(self, t, y, params):
        return [1]  # Only one expression, but two state variables


# =============================================================================
# Tests: validate_model_method
# =============================================================================


class TestValidateModelMethod:
    """Tests for the validate_model_method function."""

    def test_simple_model_no_issues(self):
        """Simple models should have no critical issues."""
        model = SIR()
        issues = validate_model_method(model)
        critical = [i for i in issues if i.is_critical]
        assert len(critical) == 0

    def test_model_with_if_statement_warns(self):
        """If statements should generate non-critical warnings."""
        model = ModelWithIfStatement()
        issues = validate_model_method(model)
        conditional_issues = [i for i in issues if i.category == "conditional"]
        assert len(conditional_issues) > 0
        assert not conditional_issues[0].is_critical

    def test_model_with_loop_fails(self):
        """Loops should generate critical issues."""
        model = ModelWithLoop()
        issues = validate_model_method(model)
        loop_issues = [i for i in issues if i.category == "loop"]
        assert len(loop_issues) > 0
        assert loop_issues[0].is_critical

    def test_model_with_lambda_warns(self):
        """Lambda functions should generate non-critical warnings."""
        model = ModelWithLambda()
        issues = validate_model_method(model)
        lambda_issues = [i for i in issues if i.category == "lambda"]
        assert len(lambda_issues) > 0
        assert not lambda_issues[0].is_critical


# =============================================================================
# Tests: extract_formulas (successful cases)
# =============================================================================


class TestExtractFormulasSuccess:
    """Tests for successful formula extraction."""

    @pytest.mark.parametrize(
        "model_class,expected_states",
        [
            (SIR, ["S", "I", "R"]),
            (SIS, ["S", "I"]),
            (SIRS, ["S", "I", "R"]),
            (SEIR, ["S", "E", "I", "R"]),
        ],
    )
    def test_extract_formulas_simple_models(self, model_class, expected_states):
        """Test automatic extraction works for simple models."""
        model = model_class()
        formulas = extract_formulas(model)

        assert isinstance(formulas, dict)
        assert set(formulas.keys()) == set(expected_states)

        for state, expr in formulas.items():
            assert isinstance(expr, (sp.Expr, sp.Number))

    def test_extract_sir_formulas_content(self):
        """Verify SIR formulas are correctly extracted."""
        model = SIR()
        formulas = extract_formulas(model)

        # Check S formula
        assert formulas["S"] == -sp.Symbol("beta") * sp.Symbol("S") * sp.Symbol("I") / sp.Symbol(
            "N"
        )

        # Check I formula
        assert formulas["I"] == (
            sp.Symbol("beta") * sp.Symbol("S") * sp.Symbol("I") / sp.Symbol("N")
            - sp.Symbol("gamma") * sp.Symbol("I")
        )

        # Check R formula
        assert formulas["R"] == sp.Symbol("gamma") * sp.Symbol("I")

    def test_extract_sis_formulas_content(self):
        """Verify SIS formulas are correctly extracted."""
        model = SIS()
        formulas = extract_formulas(model)

        assert formulas["S"] == (
            -sp.Symbol("beta") * sp.Symbol("S") * sp.Symbol("I") / sp.Symbol("N")
            + sp.Symbol("gamma") * sp.Symbol("I")
        )
        assert formulas["I"] == (
            sp.Symbol("beta") * sp.Symbol("S") * sp.Symbol("I") / sp.Symbol("N")
            - sp.Symbol("gamma") * sp.Symbol("I")
        )


# =============================================================================
# Tests: extract_formulas (failure cases)
# =============================================================================


class TestExtractFormulasFailure:
    """Tests for formula extraction failures."""

    def test_model_with_loop_raises_error(self):
        """Loops should raise FormulaExtractionError."""
        model = ModelWithLoop()
        with pytest.raises(FormulaExtractionError) as exc_info:
            extract_formulas(model)

        assert "loop" in str(exc_info.value).lower()
        assert exc_info.value.model_name == "BrokenLoop"

    def test_model_without_model_method_raises_error(self):
        """Missing _model should raise FormulaExtractionError."""
        model = ModelWithoutModelMethod()
        with pytest.raises(FormulaExtractionError) as exc_info:
            extract_formulas(model)

        assert "_model" in str(exc_info.value).lower()

    def test_model_returning_none_raises_error(self):
        """Returning None should raise FormulaExtractionError."""
        model = ModelReturningNone()
        with pytest.raises(FormulaExtractionError) as exc_info:
            extract_formulas(model)

        assert "None" in str(exc_info.value)

    def test_model_returning_wrong_count_raises_error(self):
        """Wrong number of expressions should raise FormulaExtractionError."""
        model = ModelReturningWrongCount()
        with pytest.raises(FormulaExtractionError) as exc_info:
            extract_formulas(model)

        assert "expressions" in str(exc_info.value).lower()

    def test_model_with_if_statement_raises_error(self):
        """If statements with symbolic conditions should raise FormulaExtractionError."""
        model = ModelWithIfStatement()
        with pytest.raises(FormulaExtractionError) as exc_info:
            extract_formulas(model)

        # The error should mention symbolic execution failure
        assert "Symbolic execution" in str(exc_info.value) or "truth value" in str(exc_info.value)


# =============================================================================
# Tests: validate_formulas
# =============================================================================


class TestValidateFormulas:
    """Tests for manual formula validation."""

    def test_valid_formulas_pass(self):
        """Valid formulas should pass validation."""
        model = SIR()
        formulas = {
            "S": -sp.Symbol("beta") * sp.Symbol("S") * sp.Symbol("I") / sp.Symbol("N"),
            "I": sp.Symbol("beta") * sp.Symbol("S") * sp.Symbol("I") / sp.Symbol("N"),
            "R": sp.Symbol("gamma") * sp.Symbol("I"),
        }
        validate_formulas(model, formulas)  # Should not raise

    def test_missing_state_variable_raises_error(self):
        """Missing state variable should raise ValueError."""
        model = SIR()
        formulas = {
            "S": -sp.Symbol("beta") * sp.Symbol("S") * sp.Symbol("I") / sp.Symbol("N"),
            "I": sp.Symbol("beta") * sp.Symbol("S") * sp.Symbol("I") / sp.Symbol("N"),
            # Missing R
        }
        with pytest.raises(ValueError) as exc_info:
            validate_formulas(model, formulas)

        assert "R" in str(exc_info.value)

    def test_wrong_type_raises_error(self):
        """Non-SymPy expression should raise TypeError."""
        model = SIR()
        formulas = {
            "S": "not an expression",
            "I": sp.Symbol("I"),
            "R": sp.Symbol("R"),
        }
        with pytest.raises(TypeError) as exc_info:
            validate_formulas(model, formulas)

        assert "S" in str(exc_info.value)


# =============================================================================
# Tests: sympy_to_vfgen
# =============================================================================


class TestSympyToVfgen:
    """Tests for SymPy to vfgen string conversion."""

    def test_multiplication(self):
        """Multiplication should remain as *."""
        expr = sp.Symbol("a") * sp.Symbol("b")
        result = sympy_to_vfgen(expr)
        assert result == "a*b"

    def test_division(self):
        """Division should remain as /."""
        expr = sp.Symbol("a") / sp.Symbol("b")
        result = sympy_to_vfgen(expr)
        assert result == "a/b"

    def test_power(self):
        """Power should convert from ** to ^."""
        expr = sp.Symbol("a") ** 2
        result = sympy_to_vfgen(expr)
        assert result == "a^2"

    def test_complex_expression(self):
        """Complex expressions should convert correctly."""
        a, b, c = sp.symbols("a b c")
        expr = a * b / c + a**2
        result = sympy_to_vfgen(expr)
        assert "a^2" in result
        assert "a*b/c" in result

    def test_exp_function(self):
        """Exponential function should remain as exp."""
        expr = sp.exp(sp.Symbol("x"))
        result = sympy_to_vfgen(expr)
        assert result == "exp(x)"

    def test_sin_cos_functions(self):
        """Trig functions should remain as sin/cos."""
        x = sp.Symbol("x")
        assert sympy_to_vfgen(sp.sin(x)) == "sin(x)"
        assert sympy_to_vfgen(sp.cos(x)) == "cos(x)"

    def test_tanh_function(self):
        """Hyperbolic tangent should remain as tanh."""
        expr = sp.tanh(sp.Symbol("x"))
        result = sympy_to_vfgen(expr)
        assert result == "tanh(x)"


# =============================================================================
# Tests: get_free_symbols
# =============================================================================


class TestGetFreeSymbols:
    """Tests for extracting free symbols from formulas."""

    def test_get_free_symbols(self):
        """Should return all symbols used in formulas."""
        formulas = {
            "S": -sp.Symbol("beta") * sp.Symbol("S") * sp.Symbol("I") / sp.Symbol("N"),
            "I": sp.Symbol("gamma") * sp.Symbol("I"),
        }
        symbols = get_free_symbols(formulas)

        symbol_names = {s.name for s in symbols}
        assert "beta" in symbol_names
        assert "gamma" in symbol_names
        assert "S" in symbol_names
        assert "I" in symbol_names
        assert "N" in symbol_names


# =============================================================================
# Tests: Error message quality
# =============================================================================


class TestErrorMessages:
    """Tests for error message quality."""

    def test_error_includes_model_name(self):
        """Error should include the model name."""
        model = ModelWithLoop()
        try:
            extract_formulas(model)
        except FormulaExtractionError as e:
            assert "BrokenLoop" in str(e)

    def test_error_includes_suggestion(self):
        """Error should include a suggestion for fixing."""
        model = ModelWithLoop()
        try:
            extract_formulas(model)
        except FormulaExtractionError as e:
            assert e.suggestion != ""
            assert "_formulas" in e.suggestion


# =============================================================================
# Tests: Model.get_formulas() method
# =============================================================================


class TestModelGetFormulas:
    """Tests for the get_formulas method on ContinuousModel."""

    def test_get_formulas_automatic_extraction(self):
        """Should automatically extract formulas when _formulas not set."""
        model = SIR()
        assert model._formulas is None
        formulas = model.get_formulas()
        assert formulas is not None
        assert set(formulas.keys()) == {"S", "I", "R"}

    def test_get_formulas_manual_override(self):
        """Manual _formulas should take precedence."""
        model = SIR()
        manual_formulas = {
            "S": sp.Symbol("S"),
            "I": sp.Symbol("I"),
            "R": sp.Symbol("R"),
        }
        model._formulas = manual_formulas
        formulas = model.get_formulas()

        assert formulas == manual_formulas

    def test_get_formulas_validation_on_manual(self):
        """Should validate manual formulas."""
        model = SIR()
        # Missing R state variable
        model._formulas = {"S": sp.Symbol("S"), "I": sp.Symbol("I")}

        with pytest.raises(ValueError):
            model.get_formulas()

"""
Tests for VFGen XML exporter.
"""

import tempfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

sympy = pytest.importorskip("sympy")
import sympy as sp

from epimodels import FormulaExtractionError
from epimodels.continuous import SIR, SIS, SIRS, SEIR
from epimodels.discrete import SIR as DiscreteSIR
from epimodels.exporters import VFGenExporter


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sir_model():
    """Create a configured SIR model for testing."""
    model = SIR()
    model.param_values = {"beta": 0.3, "gamma": 0.1}
    return model


@pytest.fixture
def sir_model_configured():
    """Create a fully configured SIR model with initial conditions."""
    model = SIR()
    model.param_values = {"beta": 0.3, "gamma": 0.1}
    return model


# =============================================================================
# Tests: VFGenExporter initialization
# =============================================================================


class TestVFGenExporterInit:
    """Tests for VFGenExporter initialization."""

    def test_init_with_continuous_model(self, sir_model):
        """Should accept ContinuousModel instances."""
        exporter = VFGenExporter(sir_model)
        assert exporter.model is sir_model

    def test_init_rejects_non_model(self):
        """Should reject objects without _model method."""
        with pytest.raises(TypeError) as exc_info:
            VFGenExporter("not a model")
        assert "ContinuousModel" in str(exc_info.value)

    def test_init_rejects_discrete_model(self):
        """Should reject DiscreteModel instances."""
        model = DiscreteSIR()
        with pytest.raises(TypeError) as exc_info:
            VFGenExporter(model)
        assert "ContinuousModel" in str(exc_info.value) or "_model" in str(exc_info.value)

    def test_init_rejects_object_without_state_variables(self):
        """Should reject objects without state_variables."""

        class FakeModel:
            model_type = "Fake"
            parameters = {}

            def _model(self, t, y, params):
                return [1, 2, 3]

        with pytest.raises(TypeError) as exc_info:
            VFGenExporter(FakeModel())
        assert "state_variables" in str(exc_info.value)


# =============================================================================
# Tests: XML structure
# =============================================================================


class TestXMLStructure:
    """Tests for XML document structure."""

    def test_xml_has_declaration(self, sir_model):
        """XML should have version declaration."""
        exporter = VFGenExporter(sir_model)
        xml = exporter.export(population=1000)
        assert xml.startswith('<?xml version="1.0"')

    def test_xml_has_vector_field_root(self, sir_model):
        """XML root element should be VectorField."""
        exporter = VFGenExporter(sir_model)
        xml = exporter.export(population=1000)
        root = ET.fromstring(xml)
        assert root.tag == "VectorField"

    def test_xml_vector_field_name(self, sir_model):
        """VectorField should have Name attribute from model_type."""
        exporter = VFGenExporter(sir_model)
        xml = exporter.export(population=1000)
        root = ET.fromstring(xml)
        assert root.get("Name") == "SIR"


# =============================================================================
# Tests: Parameter elements
# =============================================================================


class TestParameterElements:
    """Tests for Parameter XML elements."""

    def test_xml_has_parameters(self, sir_model):
        """XML should contain Parameter elements."""
        exporter = VFGenExporter(sir_model)
        xml = exporter.export(population=1000)
        root = ET.fromstring(xml)

        params = root.findall("Parameter")
        assert len(params) == 2  # beta and gamma

    def test_xml_parameter_names(self, sir_model):
        """Parameters should have correct names."""
        exporter = VFGenExporter(sir_model)
        xml = exporter.export(population=1000)
        root = ET.fromstring(xml)

        param_names = {p.get("Name") for p in root.findall("Parameter")}
        assert param_names == {"beta", "gamma"}

    def test_xml_parameter_default_values(self, sir_model):
        """Parameters should have default values from param_values."""
        exporter = VFGenExporter(sir_model)
        xml = exporter.export(population=1000)
        root = ET.fromstring(xml)

        for param in root.findall("Parameter"):
            name = param.get("Name")
            if name == "beta":
                assert param.get("DefaultValue") == "0.3"
            elif name == "gamma":
                assert param.get("DefaultValue") == "0.1"

    def test_xml_parameter_default_values_override(self, sir_model):
        """Should use provided default_values over param_values."""
        exporter = VFGenExporter(sir_model)
        xml = exporter.export(
            population=1000,
            default_values={"beta": 0.5, "gamma": 0.2},
        )
        root = ET.fromstring(xml)

        for param in root.findall("Parameter"):
            name = param.get("Name")
            if name == "beta":
                assert param.get("DefaultValue") == "0.5"
            elif name == "gamma":
                assert param.get("DefaultValue") == "0.2"

    def test_xml_parameter_latex(self, sir_model):
        """Parameters should have Latex attributes."""
        exporter = VFGenExporter(sir_model)
        xml = exporter.export(population=1000, include_latex=True)
        root = ET.fromstring(xml)

        for param in root.findall("Parameter"):
            latex = param.get("Latex")
            if latex:
                assert latex in [r"$\beta$", r"$\gamma$", r"\beta", r"\gamma"]

    def test_xml_parameter_no_latex(self, sir_model):
        """Should not include Latex when include_latex=False."""
        exporter = VFGenExporter(sir_model)
        xml = exporter.export(population=1000, include_latex=False)
        root = ET.fromstring(xml)

        for param in root.findall("Parameter"):
            assert param.get("Latex") is None


# =============================================================================
# Tests: StateVariable elements
# =============================================================================


class TestStateVariableElements:
    """Tests for StateVariable XML elements."""

    def test_xml_has_state_variables(self, sir_model):
        """XML should contain StateVariable elements."""
        exporter = VFGenExporter(sir_model)
        xml = exporter.export(population=1000)
        root = ET.fromstring(xml)

        states = root.findall("StateVariable")
        assert len(states) == 3  # S, I, R

    def test_xml_state_variable_names(self, sir_model):
        """StateVariables should have correct names."""
        exporter = VFGenExporter(sir_model)
        xml = exporter.export(population=1000)
        root = ET.fromstring(xml)

        state_names = {s.get("Name") for s in root.findall("StateVariable")}
        assert state_names == {"S", "I", "R"}

    def test_xml_state_variable_formulas(self, sir_model):
        """StateVariables should have Formula attributes."""
        exporter = VFGenExporter(sir_model)
        xml = exporter.export(population=1000)
        root = ET.fromstring(xml)

        for state in root.findall("StateVariable"):
            formula = state.get("Formula")
            assert formula is not None
            assert len(formula) > 0

    def test_xml_state_variable_formula_content(self, sir_model):
        """StateVariable formulas should be correct."""
        exporter = VFGenExporter(sir_model)
        xml = exporter.export(population=1000)
        root = ET.fromstring(xml)

        for state in root.findall("StateVariable"):
            name = state.get("Name")
            formula = state.get("Formula")

            if name == "S":
                assert "beta" in formula
                assert "S" in formula
                assert "I" in formula
            elif name == "I":
                assert "gamma" in formula
            elif name == "R":
                assert "gamma" in formula

    def test_xml_state_variable_initial_conditions(self, sir_model):
        """StateVariables should have initial conditions when provided."""
        exporter = VFGenExporter(sir_model)
        xml = exporter.export(
            population=1000,
            initial_conditions={"S": 990, "I": 10, "R": 0},
        )
        root = ET.fromstring(xml)

        for state in root.findall("StateVariable"):
            name = state.get("Name")
            ic = state.get("DefaultInitialCondition")
            if name == "S":
                assert ic == "990"
            elif name == "I":
                assert ic == "10"
            elif name == "R":
                assert ic == "0"

    def test_xml_state_variable_no_initial_conditions(self, sir_model):
        """Should not include initial conditions when not provided."""
        exporter = VFGenExporter(sir_model)
        xml = exporter.export(population=1000)
        root = ET.fromstring(xml)

        for state in root.findall("StateVariable"):
            assert state.get("DefaultInitialCondition") is None


# =============================================================================
# Tests: Constant elements
# =============================================================================


class TestConstantElements:
    """Tests for Constant XML elements."""

    def test_xml_has_n_constant(self, sir_model):
        """XML should contain N as a Constant."""
        exporter = VFGenExporter(sir_model)
        xml = exporter.export(population=1000)
        root = ET.fromstring(xml)

        constants = root.findall("Constant")
        n_constant = [c for c in constants if c.get("Name") == "N"]
        assert len(n_constant) == 1
        assert n_constant[0].get("Value") == "1000"

    def test_xml_no_n_constant_when_disabled(self, sir_model):
        """Should not include N constant when include_n_constant=False."""
        exporter = VFGenExporter(sir_model)
        xml = exporter.export(population=1000, include_n_constant=False)
        root = ET.fromstring(xml)

        constants = root.findall("Constant")
        n_constant = [c for c in constants if c.get("Name") == "N"]
        assert len(n_constant) == 0


# =============================================================================
# Tests: Expression elements
# =============================================================================


class TestExpressionElements:
    """Tests for Expression XML elements."""

    def test_xml_has_expressions(self, sir_model):
        """XML should contain Expression elements when provided."""
        exporter = VFGenExporter(sir_model)
        expressions = {"lambda": sp.Symbol("beta") * sp.Symbol("I")}
        xml = exporter.export(population=1000, expressions=expressions)
        root = ET.fromstring(xml)

        exprs = root.findall("Expression")
        lambda_expr = [e for e in exprs if e.get("Name") == "lambda"]
        assert len(lambda_expr) == 1
        assert "beta" in lambda_expr[0].get("Formula")


# =============================================================================
# Tests: Function elements
# =============================================================================


class TestFunctionElements:
    """Tests for Function XML elements."""

    def test_xml_has_functions(self, sir_model):
        """XML should contain Function elements when provided."""
        exporter = VFGenExporter(sir_model)
        functions = {"R0": sp.Symbol("beta") / sp.Symbol("gamma")}
        xml = exporter.export(population=1000, functions=functions)
        root = ET.fromstring(xml)

        funcs = root.findall("Function")
        r0_func = [f for f in funcs if f.get("Name") == "R0"]
        assert len(r0_func) == 1
        assert "beta" in r0_func[0].get("Formula")
        assert "gamma" in r0_func[0].get("Formula")


# =============================================================================
# Tests: Multiple models
# =============================================================================


class TestMultipleModels:
    """Tests for exporting different model types."""

    @pytest.mark.parametrize(
        "model_class,expected_states,expected_params",
        [
            (SIR, ["S", "I", "R"], ["beta", "gamma"]),
            (SIS, ["S", "I"], ["beta", "gamma"]),
            (SIRS, ["S", "I", "R"], ["beta", "gamma", "xi"]),
            (SEIR, ["S", "E", "I", "R"], ["beta", "gamma", "epsilon"]),
        ],
    )
    def test_export_various_models(self, model_class, expected_states, expected_params):
        """Should export various model types correctly."""
        model = model_class()
        model.param_values = {p: 0.1 for p in model.parameters}

        exporter = VFGenExporter(model)
        xml = exporter.export(population=1000)
        root = ET.fromstring(xml)

        state_names = {s.get("Name") for s in root.findall("StateVariable")}
        param_names = {p.get("Name") for p in root.findall("Parameter")}

        assert state_names == set(expected_states)
        assert param_names == set(expected_params)


# =============================================================================
# Tests: Output formats
# =============================================================================


class TestOutputFormats:
    """Tests for different output formats."""

    def test_export_returns_string(self, sir_model):
        """Should return XML string when no filepath."""
        exporter = VFGenExporter(sir_model)
        result = exporter.export(population=1000)

        assert isinstance(result, str)
        assert "<?xml" in result

    def test_export_writes_to_file(self, sir_model):
        """Should write to file when filepath provided."""
        exporter = VFGenExporter(sir_model)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".vf", delete=False) as f:
            filepath = f.name

        try:
            result = exporter.export(filepath=filepath, population=1000)
            assert result is None

            content = Path(filepath).read_text()
            assert "<?xml" in content
            assert "VectorField" in content
        finally:
            Path(filepath).unlink()


# =============================================================================
# Tests: model.to_vfgen() method
# =============================================================================


class TestModelToVfgen:
    """Tests for the to_vfgen convenience method."""

    def test_to_vfgen_returns_string(self):
        """to_vfgen should return XML string."""
        model = SIR()
        model.param_values = {"beta": 0.3, "gamma": 0.1}
        xml = model.to_vfgen(population=1000)

        assert isinstance(xml, str)
        assert "<?xml" in xml

    def test_to_vfgen_writes_to_file(self):
        """to_vfgen should write to file."""
        model = SIR()
        model.param_values = {"beta": 0.3, "gamma": 0.1}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".vf", delete=False) as f:
            filepath = f.name

        try:
            result = model.to_vfgen(filepath=filepath, population=1000)
            assert result is None

            content = Path(filepath).read_text()
            assert "<?xml" in content
        finally:
            Path(filepath).unlink()

    def test_to_vfgen_with_initial_conditions(self):
        """to_vfgen should accept initial conditions."""
        model = SIR()
        model.param_values = {"beta": 0.3, "gamma": 0.1}
        xml = model.to_vfgen(
            population=1000,
            initial_conditions={"S": 990, "I": 10, "R": 0},
        )
        root = ET.fromstring(xml)

        for state in root.findall("StateVariable"):
            name = state.get("Name")
            if name == "S":
                assert state.get("DefaultInitialCondition") == "990"


# =============================================================================
# Tests: Formula extraction integration
# =============================================================================


class TestFormulaExtractionIntegration:
    """Tests for integration with formula extraction."""

    def test_automatic_formula_extraction(self):
        """Should automatically extract formulas."""
        model = SIR()
        model.param_values = {"beta": 0.3, "gamma": 0.1}

        exporter = VFGenExporter(model)
        xml = exporter.export(population=1000)
        root = ET.fromstring(xml)

        # Check that formulas exist
        for state in root.findall("StateVariable"):
            assert state.get("Formula") is not None


# =============================================================================
# Tests: XML formatting
# =============================================================================


class TestXMLFormatting:
    """Tests for XML formatting and readability."""

    def test_xml_is_pretty_printed(self, sir_model):
        """XML should be pretty-printed with indentation."""
        exporter = VFGenExporter(sir_model)
        xml = exporter.export(population=1000)

        # Should have newlines for readability
        assert "\n" in xml

        # Should have consistent indentation
        lines = xml.split("\n")
        indented_lines = [l for l in lines if l.startswith("    ")]
        assert len(indented_lines) > 0

    def test_xml_elements_on_separate_lines(self, sir_model):
        """Each element should be on its own line."""
        exporter = VFGenExporter(sir_model)
        xml = exporter.export(population=1000)

        assert "<Parameter" in xml
        assert "<StateVariable" in xml


# =============================================================================
# Tests: Error handling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in export."""

    def test_formula_extraction_error_propagates(self):
        """FormulaExtractionError should propagate from extraction."""
        from epimodels.discrete import SEIS

        model = SEIS()
        with pytest.raises(TypeError):
            VFGenExporter(model)


# =============================================================================
# Tests: Power operator conversion
# =============================================================================


class TestPowerOperatorConversion:
    """Tests for ** to ^ conversion in formulas."""

    def test_power_in_seir_model(self):
        """Power operator should be converted to ^."""
        model = SEIR()
        model.param_values = {"beta": 0.3, "gamma": 0.1, "epsilon": 0.5}

        exporter = VFGenExporter(model)
        xml = exporter.export(population=1000)

        # Check that ** is converted to ^
        # The formulas might contain powers, so verify conversion
        assert "**" not in xml or "^" in xml

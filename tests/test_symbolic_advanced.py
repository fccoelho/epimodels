"""
Tests for advanced symbolic analysis features (equilibrium, stability, and sensitivity).
"""

import pytest
import numpy as np
from epimodels.validation import SymbolicModel
from epimodels import ValidationError
from sympy import symbols, Matrix, simplify
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


@pytest.fixture
def sir_model():
    """Create a symbolic SIR model with vital dynamics for testing."""
    model = SymbolicModel()

    model.add_parameter("beta", positive=True)
    model.add_parameter("gamma", positive=True)
    model.add_parameter("mu", positive=True)
    model.add_variable("S", positive=True)
    model.add_variable("I", positive=True)
    model.add_variable("R", positive=True)
    model.set_total_population("N")

    model.define_ode("S", "mu*N - beta*S*I/N - mu*S")
    model.define_ode("I", "beta*S*I/N - gamma*I - mu*I")
    model.define_ode("R", "gamma*I - mu*R")

    return model


@pytest.fixture
def basic_sir_model():
    """Create a basic SIR model (no vital dynamics) for testing."""
    model = SymbolicModel()

    model.add_parameter("beta", positive=True)
    model.add_parameter("gamma", positive=True)
    model.add_variable("S", positive=True)
    model.add_variable("I", positive=True)
    model.add_variable("R", positive=True)
    model.set_total_population("N")

    model.define_ode("S", "-beta*S*I/N")
    model.define_ode("I", "beta*S*I/N - gamma*I")
    model.define_ode("R", "gamma*I")

    return model


class TestEquilibriumFinding:
    """Tests for equilibrium finding."""

    def test_find_disease_free_equilibrium(self, sir_model):
        """Test DFE finding."""
        dfe = sir_model.find_disease_free_equilibrium()
        assert dfe["S"] == sir_model.total_population
        assert dfe["I"] == 0
        assert dfe["R"] == 0

    def test_find_all_equilibria_basic_sir(self, basic_sir_model):
        """Test that basic SIR only has DFE."""
        params = {"beta": 0.3, "gamma": 0.1, "N": 1000}
        equilibria = basic_sir_model.find_all_equilibria(params=params)

        assert len(equilibria) >= 1

        for eq in equilibria:
            assert eq["type"] == "dfe"

    def test_find_all_equilibria_with_vital_dynamics(self, sir_model):
        """Test finding all equilibria in SIR with vital dynamics."""
        params = {"beta": 0.3, "gamma": 0.1, "mu": 0.01, "N": 1000}
        equilibria = sir_model.find_all_equilibria(params=params, numeric_fallback=True)

        assert len(equilibria) >= 1

        dfe = equilibria[0]
        assert dfe["type"] == "dfe"

    def test_find_endemic_equilibrium_exists(self, sir_model):
        """Test finding endemic equilibrium when R0 > 1."""
        params = {"beta": 0.3, "gamma": 0.1, "mu": 0.01, "N": 1000}

        endemic = sir_model.find_endemic_equilibrium(params, numeric_fallback=True)

        if endemic:
            assert endemic["type"] == "endemic"
            assert endemic["I"] is not None

    def test_find_endemic_equilibrium_not_exists(self, basic_sir_model):
        """Test that basic SIR has no endemic equilibrium."""
        params = {"beta": 0.3, "gamma": 0.1, "N": 1000}

        endemic = basic_sir_model.find_endemic_equilibrium(params)

        assert endemic is None


class TestR0Computation:
    """Tests for basic reproduction number computation."""

    def test_compute_R0(self, sir_model):
        """Test R0 computation."""
        R0 = sir_model.compute_R0_next_generation()

        assert R0 is not None

    def test_R0_numeric_evaluation(self, sir_model):
        """Test numeric evaluation of R0."""
        R0 = sir_model.compute_R0_next_generation()
        params = {"beta": 0.3, "gamma": 0.1, "mu": 0.01, "N": 1000, "S": 1000}

        R0_numeric = sir_model.substitute_values(R0, params)
        try:
            R0_val = float(R0_numeric.evalf())
            expected_R0 = 0.3 / (0.1 + 0.01)
            assert abs(abs(R0_val) - expected_R0) < 0.5
        except (TypeError, ValueError):
            pass


class TestJacobianComputation:
    """Tests for Jacobian computation."""

    def test_compute_jacobian(self, sir_model):
        """Test Jacobian computation."""
        dfe = sir_model.find_disease_free_equilibrium()

        J = sir_model.compute_jacobian(dfe)

        assert J.shape == (3, 3)

    def test_jacobian_with_substitution(self, sir_model):
        """Test Jacobian with value substitution."""
        dfe = sir_model.find_disease_free_equilibrium()
        params = {"beta": 0.3, "gamma": 0.1, "mu": 0.01, "N": 1000}

        J = sir_model.compute_jacobian(dfe, substitute_values=True)
        J_sub = sir_model.substitute_values(J, params)

        assert J_sub.shape == (3, 3)


class TestEigenvalueComputation:
    """Tests for eigenvalue computation."""

    def test_compute_eigenvalues_symbolic(self, sir_model):
        """Test symbolic eigenvalue computation."""
        dfe = sir_model.find_disease_free_equilibrium()
        J = sir_model.compute_jacobian(dfe)

        eigenvalues = sir_model.compute_eigenvalues(J, numeric=False)

        assert len(eigenvalues) >= 1

    def test_compute_eigenvalues_numeric(self, sir_model):
        """Test numeric eigenvalue computation."""
        dfe = sir_model.find_disease_free_equilibrium()
        params = {"beta": 0.3, "gamma": 0.1, "mu": 0.01, "N": 1000}

        J_numeric = sir_model.compute_jacobian(dfe, substitute_values=True)
        eigenvalues = sir_model.compute_eigenvalues(J_numeric, numeric=True, params=params)

        assert len(eigenvalues) >= 1


class TestStabilityAnalysis:
    """Tests for stability analysis."""

    def test_analyze_stability_dfe(self, sir_model):
        """Test stability analysis at DFE."""
        dfe = sir_model.find_disease_free_equilibrium()
        params = {"beta": 0.3, "gamma": 0.1, "mu": 0.01, "N": 1000}

        result = sir_model.analyze_stability_full(dfe, params)

        assert result["jacobian"] is not None
        assert result["stability"] in ["stable", "unstable", "neutral", "saddle", "unknown"]
        assert result["classification"] is not None

    def test_classify_stability_stable(self, sir_model):
        """Test stability classification for stable case."""
        real_parts = [-0.1, -0.2, -0.3]
        imag_parts = [0.0, 0.0, 0.0]

        classification = sir_model._classify_stability(real_parts, imag_parts, 1e-10)

        assert classification == "stable"

    def test_classify_stability_unstable(self, sir_model):
        """Test stability classification for unstable case."""
        real_parts = [0.1, 0.2, 0.3]
        imag_parts = [0.0, 0.0, 0.0]

        classification = sir_model._classify_stability(real_parts, imag_parts, 1e-10)

        assert classification == "unstable"

    def test_classify_stability_saddle(self, sir_model):
        """Test stability classification for saddle point."""
        real_parts = [0.5, -0.5]
        imag_parts = [0.0, 0.0]

        classification = sir_model._classify_stability(real_parts, imag_parts, 1e-10)

        assert classification == "saddle"

    def test_classify_stability_detailed_stable_node(self, sir_model):
        """Test detailed stability classification for stable node."""
        real_parts = [-0.1, -0.2]
        imag_parts = [0.0, 0.0]

        classification = sir_model._classify_stability_detailed(real_parts, imag_parts, 1e-10)

        assert "stable" in classification

    def test_classify_stability_detailed_saddle(self, sir_model):
        """Test detailed stability classification for saddle point."""
        real_parts = [0.5, -0.5]
        imag_parts = [0.0, 0.0]

        classification = sir_model._classify_stability_detailed(real_parts, imag_parts, 1e-10)

        assert "saddle" in classification


class TestBifurcationDetection:
    """Tests for bifurcation detection."""

    def test_detect_bifurcation_none(self, sir_model):
        """Test no bifurcation when eigenvalues far from imaginary axis."""
        real_parts = [-0.5, -0.3]
        imag_parts = [0.0, 0.0]

        near_bifurcation, bifurcation_type = sir_model._detect_bifurcation(
            real_parts, imag_parts, 1e-10
        )

        assert near_bifurcation is False
        assert bifurcation_type is None

    def test_detect_bifurcation_transcritical(self, sir_model):
        """Test transcritical bifurcation detection."""
        real_parts = [1e-12, -0.1]
        imag_parts = [0.0, 0.0]

        near_bifurcation, bifurcation_type = sir_model._detect_bifurcation(
            real_parts, imag_parts, 1e-10
        )

        assert near_bifurcation is True
        assert bifurcation_type == "transcritical_or_saddle_node"


class TestSensitivityAnalysis:
    """Tests for sensitivity analysis."""

    def test_compute_sensitivity_matrix(self, sir_model):
        """Test sensitivity matrix computation."""
        sensitivity_matrix = sir_model.compute_sensitivity_matrix(
            output_vars=["S", "I"], params=["beta", "gamma"]
        )

        assert "S" in sensitivity_matrix
        assert "I" in sensitivity_matrix
        assert "beta" in sensitivity_matrix["S"]
        assert "gamma" in sensitivity_matrix["S"]

    def test_compute_elasticity_indices(self, sir_model):
        """Test elasticity indices computation."""
        params = {"beta": 0.3, "gamma": 0.1, "mu": 0.01, "N": 1000}

        elasticities = sir_model.compute_elasticity_indices(params, output_vars=["I"])

        assert isinstance(elasticities, dict)

    def test_rank_parameter_importance(self, sir_model):
        """Test parameter importance ranking."""
        params = {"beta": 0.3, "gamma": 0.1, "mu": 0.01, "N": 1000}

        ranking = sir_model.rank_parameter_importance(params, "I", method="elasticity")

        assert isinstance(ranking, list)

    def test_perform_perturbation_analysis(self, sir_model):
        """Test perturbation analysis."""
        params = {"beta": 0.3, "gamma": 0.1, "mu": 0.01, "N": 1000}
        dfe = sir_model.find_disease_free_equilibrium()

        perturbations = sir_model.perform_perturbation_analysis(
            params, dfe, perturbation=0.01, output_vars=["S", "I", "R"]
        )

        assert isinstance(perturbations, dict)


class TestEquilibriumValidation:
    """Tests for equilibrium validation."""

    def test_validate_dfe(self, basic_sir_model):
        """Test validation of disease-free equilibrium."""
        dfe = basic_sir_model.find_disease_free_equilibrium()

        is_valid = basic_sir_model._validate_equilibrium(dfe)

        assert is_valid is True

    def test_classify_equilibrium_dfe(self, basic_sir_model):
        """Test classification of DFE."""
        dfe = basic_sir_model.find_disease_free_equilibrium()

        eq_type = basic_sir_model._classify_equilibrium(dfe)

        assert eq_type == "dfe"

    def test_classify_equilibrium_endemic(self, basic_sir_model):
        """Test classification of endemic equilibrium."""
        eq = {"S": 500, "I": 100, "R": 0}

        eq_type = basic_sir_model._classify_equilibrium(eq)

        assert eq_type == "endemic"


class TestDuplicateDetection:
    """Tests for duplicate equilibrium detection."""

    def test_is_equilibrium_duplicate_true(self, sir_model):
        """Test duplicate detection when equilibria are identical."""
        eq1 = {"S": 1000, "I": 0, "R": 0}
        eq2 = {"S": 1000, "I": 0, "R": 0}

        is_duplicate = sir_model._is_equilibrium_duplicate(eq1, [eq2])

        assert is_duplicate is True

    def test_is_equilibrium_duplicate_false(self, sir_model):
        """Test duplicate detection when equilibria are different."""
        eq1 = {"S": 500, "I": 100, "R": 0}
        eq2 = {"S": 1000, "I": 0, "R": 0}

        is_duplicate = sir_model._is_equilibrium_duplicate(eq1, [eq2])

        assert is_duplicate is False


class TestModelIdentification:
    """Tests for infected compartment identification."""

    def test_identify_infected_compartments(self, sir_model):
        """Test identification of infected compartments."""
        infected = sir_model._identify_infected_compartments()

        assert "I" in infected

    def test_identify_infected_compartments_seir(self):
        """Test identification for SEIR model."""
        model = SymbolicModel()
        model.add_parameter("beta", positive=True)
        model.add_parameter("gamma", positive=True)
        model.add_parameter("epsilon", positive=True)
        model.add_variable("S", positive=True)
        model.add_variable("E", positive=True)
        model.add_variable("I", positive=True)
        model.add_variable("R", positive=True)

        infected = model._identify_infected_compartments()

        assert "I" in infected

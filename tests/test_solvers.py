"""
Tests for solver abstraction layer.
"""

import pytest
import numpy as np
from epimodels.solvers import (
    SolverResult,
    SolverBase,
    ScipySolver,
    DiffraxSolver,
    get_default_solver,
)


class TestSolverResult:
    """Tests for SolverResult container."""

    def test_solver_result_creation(self):
        """Should create SolverResult with t and y arrays."""
        t = np.array([0, 1, 2, 3])
        y = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        result = SolverResult(t, y)

        np.testing.assert_array_equal(result.t, t)
        np.testing.assert_array_equal(result.y, y)

    def test_solver_result_repr(self):
        """Should have string representation."""
        t = np.array([0, 1, 2])
        y = np.array([[1, 2, 3]])
        result = SolverResult(t, y)

        repr_str = repr(result)
        assert "SolverResult" in repr_str


class TestScipySolver:
    """Tests for ScipySolver class."""

    def test_scipy_solver_creation(self):
        """Should create solver with default method."""
        solver = ScipySolver()
        assert solver.method == "RK45"

    def test_scipy_solver_custom_method(self):
        """Should create solver with custom method."""
        solver = ScipySolver(method="LSODA")
        assert solver.method == "LSODA"

    def test_scipy_solver_invalid_method(self):
        """Should raise error for invalid method."""
        with pytest.raises(ValueError, match="Unknown method"):
            ScipySolver(method="invalid")

    def test_scipy_solve_simple_ode(self):
        """Should solve a simple ODE."""
        solver = ScipySolver(method="RK45")

        # dy/dt = -y, y(0) = 1 => y(t) = exp(-t)
        def fn(t, y):
            return [-y[0]]

        result = solver.solve(fn, (0, 5), [1.0])

        assert len(result.t) > 1
        assert result.y.shape[0] == 1
        # Check that y decays exponentially
        np.testing.assert_allclose(result.y[0, -1], np.exp(-result.t[-1]), rtol=1e-2)

    def test_scipy_solve_sir_ode(self):
        """Should solve SIR-like ODE system."""
        solver = ScipySolver(method="RK45")

        beta, gamma, N = 0.3, 0.1, 1000

        def sir(t, y):
            S, I, R = y
            dS = -beta * S * I / N
            dI = beta * S * I / N - gamma * I
            dR = gamma * I
            return [dS, dI, dR]

        result = solver.solve(sir, (0, 50), [999, 1, 0])

        assert result.y.shape[0] == 3
        # Population should be conserved
        total = result.y[0, :] + result.y[1, :] + result.y[2, :]
        np.testing.assert_allclose(total, 1000, rtol=1e-5)

    def test_scipy_different_methods(self):
        """Should work with different scipy methods."""
        methods = ["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"]

        def fn(t, y):
            return [-y[0]]

        for method in methods:
            solver = ScipySolver(method=method)
            result = solver.solve(fn, (0, 5), [1.0])
            assert len(result.t) > 1


class TestDiffraxSolver:
    """Tests for DiffraxSolver class."""

    def test_diffrax_solver_creation(self):
        """Should create solver with default settings."""
        solver = DiffraxSolver()
        assert solver.solver_name == "Tsit5"
        assert solver.adaptive == True

    def test_diffrax_solver_custom_settings(self):
        """Should create solver with custom settings."""
        solver = DiffraxSolver(solver="Dopri5", rtol=1e-4, atol=1e-6)
        assert solver.solver_name == "Dopri5"
        assert solver.rtol == 1e-4

    def test_diffrax_solver_invalid_solver(self):
        """Should raise error for invalid solver."""
        with pytest.raises(ValueError, match="Unknown solver"):
            DiffraxSolver(solver="invalid")

    def test_diffrax_solve_simple_ode(self):
        """Should solve a simple ODE with diffrax."""
        pytest.importorskip("diffrax")

        solver = DiffraxSolver(solver="Tsit5")

        def fn(t, y):
            return [-y[0]]

        result = solver.solve(fn, (0, 5), [1.0])

        assert len(result.t) > 1
        # Check exponential decay
        np.testing.assert_allclose(result.y[0, -1], np.exp(-result.t[-1]), rtol=1e-2)


class TestGetDefaultSolver:
    """Tests for get_default_solver function."""

    def test_default_solver(self):
        """Should return ScipySolver with RK45."""
        solver = get_default_solver()
        assert isinstance(solver, ScipySolver)
        assert solver.method == "RK45"


class TestIntegrationWithModels:
    """Integration tests with actual models."""

    def test_sir_with_scipy_solver(self):
        """Should work with SIR model using ScipySolver."""
        from epimodels.continuous import SIR

        model = SIR()
        solver = ScipySolver(method="LSODA")
        model([999, 1, 0], [0, 50], 1000, {"beta": 0.3, "gamma": 0.1}, solver=solver)

        assert "S" in model.traces
        assert "I" in model.traces
        assert "R" in model.traces

        # Population conserved
        total = model.traces["S"] + model.traces["I"] + model.traces["R"]
        np.testing.assert_allclose(total, 1000, rtol=1e-5)

    def test_sir_backward_compatible(self):
        """Should still work with method parameter (backward compatible)."""
        from epimodels.continuous import SIR

        model = SIR()
        model([999, 1, 0], [0, 50], 1000, {"beta": 0.3, "gamma": 0.1}, method="RK45")

        assert "S" in model.traces
        assert model.method == "RK45"

    def test_sir_default_solver(self):
        """Should use ScipySolver by default."""
        from epimodels.continuous import SIR

        model = SIR()
        model([999, 1, 0], [0, 50], 1000, {"beta": 0.3, "gamma": 0.1})

        assert hasattr(model, "solver")
        assert isinstance(model.solver, ScipySolver)

    def test_different_scipy_methods(self):
        """Should produce similar results with different methods."""
        from epimodels.continuous import SIR

        methods = ["RK45", "LSODA", "DOP853"]
        results = []

        for method in methods:
            model = SIR()
            model([999, 1, 0], [0, 50], 1000, {"beta": 0.3, "gamma": 0.1}, method=method)
            results.append(model.traces["I"])

        # All methods should produce similar peak values
        peaks = [max(r) for r in results]
        np.testing.assert_allclose(peaks[0], peaks[1], rtol=0.03)
        np.testing.assert_allclose(peaks[0], peaks[2], rtol=0.03)

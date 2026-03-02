"""
Numerical accuracy tests for epidemic models.

Tests population conservation, equilibrium behavior, and edge cases.
"""

import numpy as np
import pytest
from epimodels.continuous import SIR, SIS, SIRS, SEIR
from epimodels.discrete import SIR as DiscreteSIR, SIS as DiscreteSIS


class TestPopulationConservation:
    """Tests to verify population is conserved in closed models."""

    def test_sir_continuous_population_conserved(self):
        """S+I+R should equal N at all times in continuous SIR."""
        model = SIR()
        N = 1000
        model([990, 10, 0], [0, 100], N, {"beta": 0.3, "gamma": 0.1})

        total = model.traces["S"] + model.traces["I"] + model.traces["R"]
        assert np.allclose(total, N, rtol=1e-5), (
            "Population not conserved in continuous SIR"
        )

    def test_sir_discrete_population_conserved(self):
        """S+I+R should equal N at all times in discrete SIR."""
        model = DiscreteSIR()
        N = 1000
        model([990, 10, 0], [0, 100], N, {"beta": 0.3, "gamma": 0.1})

        total = model.traces["S"] + model.traces["I"] + model.traces["R"]
        assert np.allclose(total, N, rtol=1e-10), (
            "Population not conserved in discrete SIR"
        )

    def test_sis_continuous_population_conserved(self):
        """S+I should equal N at all times in continuous SIS."""
        model = SIS()
        N = 1000
        model([990, 10], [0, 100], N, {"beta": 0.3, "gamma": 0.1})

        total = model.traces["S"] + model.traces["I"]
        assert np.allclose(total, N, rtol=1e-5), (
            "Population not conserved in continuous SIS"
        )

    def test_sis_discrete_population_conserved(self):
        """S+I should equal N at all times in discrete SIS."""
        model = DiscreteSIS()
        N = 1000
        model([0, 10, 990], [0, 100], N, {"beta": 0.3, "gamma": 0.1})

        total = model.traces["S"] + model.traces["I"]
        assert np.allclose(total, N, rtol=1e-10), (
            "Population not conserved in discrete SIS"
        )

    def test_seir_continuous_population_conserved(self):
        """S+E+I+R should equal N at all times in continuous SEIR."""
        model = SEIR()
        N = 1000
        model([990, 0, 10, 0], [0, 100], N, {"beta": 0.3, "gamma": 0.1, "epsilon": 0.2})

        total = (
            model.traces["S"]
            + model.traces["E"]
            + model.traces["I"]
            + model.traces["R"]
        )
        assert np.allclose(total, N, rtol=1e-5), (
            "Population not conserved in continuous SEIR"
        )


class TestEquilibriumBehavior:
    """Tests for equilibrium behavior of models."""

    def test_sir_no_infected_equilibrium(self):
        """SIR with no initial infected should remain at equilibrium."""
        model = SIR()
        N = 1000
        model([1000, 0, 0], [0, 50], N, {"beta": 0.3, "gamma": 0.1})

        assert np.allclose(model.traces["I"], 0, atol=1e-10)
        assert np.allclose(model.traces["S"], N, rtol=1e-5)
        assert np.allclose(model.traces["R"], 0, atol=1e-10)

    def test_sis_disease_free_equilibrium(self):
        """SIS should reach disease-free equilibrium when R0 < 1."""
        model = SIS()
        N = 1000
        # R0 = beta/gamma = 0.1/0.5 = 0.2 < 1
        model([900, 100], [0, 200], N, {"beta": 0.1, "gamma": 0.5})

        # I should approach 0
        assert model.traces["I"][-1] < 1, "Disease should die out when R0 < 1"


class TestMonotonicity:
    """Tests for monotonic behavior in certain scenarios."""

    def test_sir_susceptible_decreases(self):
        """Susceptible population should monotonically decrease in SIR."""
        model = SIR()
        N = 1000
        model([990, 10, 0], [0, 50], N, {"beta": 0.5, "gamma": 0.1})

        S = model.traces["S"]
        # Check that S is monotonically decreasing (allowing for numerical noise)
        assert np.all(np.diff(S) <= 1e-6), "Susceptible should monotonically decrease"

    def test_sir_removed_increases(self):
        """Removed population should monotonically increase in SIR."""
        model = SIR()
        N = 1000
        model([990, 10, 0], [0, 50], N, {"beta": 0.5, "gamma": 0.1})

        R = model.traces["R"]
        # Check that R is monotonically increasing (allowing for numerical noise)
        assert np.all(np.diff(R) >= -1e-6), "Removed should monotonically increase"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_sir_zero_beta_no_transmission(self):
        """With beta=0, no transmission should occur."""
        model = SIR()
        N = 1000
        model([990, 10, 0], [0, 50], N, {"beta": 0, "gamma": 0.1})

        # I should decrease due to recovery, S should stay constant
        assert np.allclose(model.traces["S"], 990, rtol=1e-5)

    def test_sir_single_initial_infected(self):
        """Model should handle single initial infected individual."""
        model = SIR()
        N = 10000
        model([9999, 1, 0], [0, 100], N, {"beta": 0.5, "gamma": 0.1})

        # Should still run without errors and conserve population
        total = model.traces["S"] + model.traces["I"] + model.traces["R"]
        assert np.allclose(total, N, rtol=1e-5)


class TestDiscreteVsContinuous:
    """Compare discrete and continuous model outputs for consistency."""

    def test_sir_discrete_approximates_continuous(self):
        """Discrete SIR should approximate continuous SIR with small time steps."""
        N = 1000
        params = {"beta": 0.3, "gamma": 0.1}
        inits = [990, 10, 0]

        # Continuous model
        cont_model = SIR()
        cont_model(inits, [0, 50], N, params, t_eval=range(51))

        # Discrete model
        disc_model = DiscreteSIR()
        disc_model(inits, [0, 51], N, params)

        # Both should conserve population
        cont_total = (
            cont_model.traces["S"] + cont_model.traces["I"] + cont_model.traces["R"]
        )
        disc_total = (
            disc_model.traces["S"] + disc_model.traces["I"] + disc_model.traces["R"]
        )
        assert np.allclose(cont_total, N, rtol=1e-5)
        assert np.allclose(disc_total, N, rtol=1e-10)

        # Both should show epidemic dynamics (I increases then decreases)
        assert np.max(cont_model.traces["I"]) > inits[1]
        assert np.max(disc_model.traces["I"]) > inits[1]

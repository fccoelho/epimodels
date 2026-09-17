"""Tests for the SDE layer (requires jax/diffrax)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

jax = pytest.importorskip("jax")
diffrax = pytest.importorskip("diffrax")

from epimodels.continuous import SIR  # noqa: E402
from epimodels.exceptions import ValidationError  # noqa: E402
from epimodels.sde import SDEModel  # noqa: E402

PARAMS = {"beta": 2.0, "gamma": 0.5}


@pytest.fixture
def sde():
    return SDEModel(SIR())


class TestSDEModel:
    def test_requires_continuous_model(self):
        from epimodels.discrete import SIR as DiscreteSIR

        with pytest.raises(TypeError, match="ContinuousModel"):
            SDEModel(DiscreteSIR())

    def test_negative_noise_scale_rejected(self):
        with pytest.raises(ValueError, match="noise_scale"):
            SDEModel(SIR(), noise_scale=-1)

    def test_single_run_shapes(self, sde):
        sde([989, 10, 0], [0, 50], 1000, PARAMS, n_sims=1, seed=0, n_points=101)
        assert sde.traces["I"].shape == (101,)
        assert sde.traces["time"].shape == (101,)

    def test_multi_run_shapes(self, sde):
        sde([989, 10, 0], [0, 50], 1000, PARAMS, n_sims=10, seed=0, n_points=51)
        assert sde.traces["I"].shape == (10, 51)
        assert len(sde.get_quantiles(0.95)["I"][0]) == 51

    def test_stochasticity(self, sde):
        """Two identical-seed-free runs differ; zero noise reproduces ODE."""
        sde([989, 10, 0], [0, 40], 1000, PARAMS, n_sims=5, seed=1, n_points=41)
        traces_a = np.asarray(sde.traces["I"]).copy()

        sde2 = SDEModel(SIR(), noise_scale=0.0)
        sde2([989, 10, 0], [0, 40], 1000, PARAMS, n_sims=2, seed=1, n_points=41)
        traces_b = np.asarray(sde2.traces["I"])
        assert np.allclose(traces_b[0], traces_b[1]), "noise_scale=0 must be deterministic"
        assert not np.allclose(traces_a[0], traces_b[0]), "noisy vs noiseless must differ"

    def test_seed_reproducibility(self, sde):
        sde([989, 10, 0], [0, 30], 1000, PARAMS, n_sims=3, seed=42, n_points=31)
        first = np.asarray(sde.traces["I"]).copy()
        sde([989, 10, 0], [0, 30], 1000, PARAMS, n_sims=3, seed=42, n_points=31)
        np.testing.assert_allclose(first, np.asarray(sde.traces["I"]), rtol=1e-5)

    def test_mean_close_to_ode(self, sde):
        """Ensemble mean should track the deterministic SIR solution."""
        from scipy.integrate import solve_ivp

        sde([989, 10, 0], [0, 40], 1000, PARAMS, n_sims=30, seed=3, n_points=81)

        def ode(t, y):
            S, Inf, R = y
            return [
                -PARAMS["beta"] * S * Inf / 1000,
                PARAMS["beta"] * S * Inf / 1000 - PARAMS["gamma"] * Inf,
                PARAMS["gamma"] * Inf,
            ]

        t_eval = sde.traces["time"]
        sol = solve_ivp(ode, [0, 40], [989, 10, 0], t_eval=t_eval)
        mean_I = np.asarray(sde.get_mean()["I"])
        # Loose tolerance: demographic noise on 10 initial infected shifts the wave
        assert np.sqrt(np.mean((mean_I - sol.y[1]) ** 2)) < 25.0

    def test_custom_diffusion(self, sde):
        """User-supplied diffusion function is used."""
        calls = []

        def diffusion(t, y, params):
            calls.append(1)
            import jax.numpy as jnp

            return jnp.diag(jnp.full(len(y), 1e-6))

        sde.custom_diffusion = diffusion
        sde([989, 10, 0], [0, 20], 1000, PARAMS, n_sims=2, seed=0, n_points=21)
        assert len(calls) > 0

    def test_invalid_args(self, sde):
        with pytest.raises(ValueError, match="n_sims"):
            sde([989, 10, 0], [0, 10], 1000, PARAMS, n_sims=0)
        with pytest.raises(ValueError, match="dt"):
            sde([989, 10, 0], [0, 10], 1000, PARAMS, dt=0)
        with pytest.raises(ValidationError):
            sde([989, 10, 0], [0, 10], 1000, {"beta": 2.0})  # missing gamma

    def test_methods_before_run_raise(self, sde):
        with pytest.raises(ValueError, match="Run the model"):
            sde.get_mean()
        with pytest.raises(ValueError, match="Run the model"):
            sde.get_quantiles()

    def test_plot_traces(self, sde):
        sde([989, 10, 0], [0, 30], 1000, PARAMS, n_sims=5, seed=2, n_points=31)
        ax = sde.plot_traces("I")
        assert ax is not None
        with pytest.raises(KeyError):
            sde.plot_traces("Z")
        plt.close("all")

"""
Tests for CTMC stochastic models.

Covers:
- Solver infrastructure (CTMCTrajectory, GillespieSolver)
- CTMCModel base class validation and accessors
- Concrete models (SIR, SIS, SIRS, SEIR)
- Multi-replicate runs and statistics
- Comparison with deterministic ODE models
"""


import numpy as np
import pytest

from epimodels.exceptions import ValidationError
from epimodels.stochastic.CTMC.models import (
    SEIR,
    SIR,
    SIRS,
    SIS,
)
from epimodels.stochastic.CTMC.solvers import (
    CTMCTrajectory,
    GillespieSolver,
)

# ============================================================
# Solver tests
# ============================================================


class TestCTMCTrajectory:
    """Tests for the CTMCTrajectory dataclass."""

    def test_basic_construction(self):
        traj = CTMCTrajectory(
            times=np.array([0.0, 0.5, 1.2, 2.0]),
            states=np.array([[100, 1, 0], [99, 2, 0], [99, 1, 1], [98, 1, 1]]),
            event_indices=np.array([0, 1, 0]),
            steps=3,
        )
        assert traj.steps == 3
        assert traj.duration == pytest.approx(2.0)

    def test_interpolate_to_grid(self):
        traj = CTMCTrajectory(
            times=np.array([0.0, 0.5, 1.5, 3.0]),
            states=np.array(
                [[100, 1], [99, 2], [98, 1], [97, 1]]
            ),
            event_indices=np.array([0, 1, 0]),
            steps=3,
        )
        t_grid = np.array([0.0, 0.3, 0.7, 1.0, 2.0, 3.0])
        result = traj.interpolate_to_grid(t_grid)

        assert result.shape == (6, 2)
        np.testing.assert_array_equal(result[0], [100, 1])  # t=0
        np.testing.assert_array_equal(result[1], [100, 1])  # t=0.3 (before first event)
        np.testing.assert_array_equal(result[2], [99, 2])  # t=0.7 (after event at 0.5)
        np.testing.assert_array_equal(result[3], [99, 2])  # t=1.0 (after event at 0.5)
        np.testing.assert_array_equal(result[4], [98, 1])  # t=2.0 (after event at 1.5)
        np.testing.assert_array_equal(result[5], [97, 1])  # t=3.0

    def test_event_times_by_index(self):
        traj = CTMCTrajectory(
            times=np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
            states=np.zeros((5, 2)),
            event_indices=np.array([0, 1, 0, 1]),
            steps=4,
        )
        et = traj.event_times_by_index()
        assert et[0] == [pytest.approx(0.5), pytest.approx(1.5)]
        assert et[1] == [pytest.approx(1.0), pytest.approx(2.0)]


class TestGillespieSolver:
    """Tests for the GillespieSolver."""

    def test_basic_sir_run(self):
        def prop_fn(params, state):
            S, I, R = state
            beta, gamma = params["beta"], params["gamma"]
            N = params["N"]
            return np.array([beta * S * I / N, gamma * I])

        tmat = np.array([[-1, 0], [1, -1], [0, 1]], dtype=np.int64)
        initial = np.array([99, 1, 0], dtype=np.int64)
        params = {"beta": 0.5, "gamma": 0.1, "N": 100}
        rng = np.random.default_rng(42)

        solver = GillespieSolver()
        traj = solver.solve(prop_fn, tmat, initial, (0.0, 50.0), params, rng)

        assert traj.steps > 0
        assert traj.times[0] == 0.0
        assert traj.times[-1] <= 50.0
        assert traj.states.shape[1] == 3
        assert traj.states[0, 0] == 99
        assert traj.states[0, 1] == 1
        assert traj.states[0, 2] == 0

        final = traj.states[-1]
        assert final.sum() == 100

    def test_reproducibility_with_seed(self):
        def prop_fn(params, state):
            S, I = state
            return np.array([0.01 * S * I, 0.1 * I])

        tmat = np.array([[-1, 0], [1, -1]], dtype=np.int64)
        initial = np.array([99, 1], dtype=np.int64)

        solver = GillespieSolver()
        params = {}

        rng1 = np.random.default_rng(123)
        traj1 = solver.solve(prop_fn, tmat, initial, (0.0, 30.0), params, rng1)

        rng2 = np.random.default_rng(123)
        traj2 = solver.solve(prop_fn, tmat, initial, (0.0, 30.0), params, rng2)

        np.testing.assert_array_equal(traj1.times, traj2.times)
        np.testing.assert_array_equal(traj1.states, traj2.states)

    def test_absorbing_state(self):
        def prop_fn(params, state):
            S, I, R = state
            return np.array([0.5 * S * I / 100, 0.1 * I])

        tmat = np.array([[-1, 0], [1, -1], [0, 1]], dtype=np.int64)
        initial = np.array([100, 0, 0], dtype=np.int64)
        params = {}
        rng = np.random.default_rng(42)

        solver = GillespieSolver()
        traj = solver.solve(prop_fn, tmat, initial, (0.0, 50.0), params, rng)

        assert traj.steps == 0
        assert traj.states[-1, 0] == 100
        assert traj.states[-1, 1] == 0

    def test_conservation_of_population(self):
        def prop_fn(params, state):
            S, I, R = state
            beta, gamma = params["beta"], params["gamma"]
            N = params["N"]
            return np.array([beta * S * I / N, gamma * I])

        tmat = np.array([[-1, 0], [1, -1], [0, 1]], dtype=np.int64)
        initial = np.array([990, 10, 0], dtype=np.int64)
        params = {"beta": 0.3, "gamma": 0.1, "N": 1000}
        rng = np.random.default_rng(42)

        solver = GillespieSolver()
        traj = solver.solve(prop_fn, tmat, initial, (0.0, 100.0), params, rng)

        for i in range(traj.states.shape[0]):
            assert traj.states[i].sum() == 1000


# ============================================================
# CTMCModel tests
# ============================================================


class TestCTMCModelValidation:
    """Tests for CTMCModel input validation."""

    def test_valid_inputs(self):
        model = SIR()
        model(
            inits=[990, 10, 0],
            trange=[0, 100],
            totpop=1000,
            params={"beta": 0.3, "gamma": 0.1},
            reps=1,
            seed=42,
        )
        assert model.traces

    def test_missing_parameters(self):
        model = SIR()
        with pytest.raises(ValidationError, match="Missing"):
            model(
                inits=[990, 10, 0],
                trange=[0, 100],
                totpop=1000,
                params={"beta": 0.3},
                validate=True,
            )

    def test_negative_initial_condition(self):
        model = SIR()
        with pytest.raises(ValidationError, match="non-negative"):
            model(
                inits=[990, -1, 0],
                trange=[0, 100],
                totpop=1000,
                params={"beta": 0.3, "gamma": 0.1},
                validate=True,
            )

    def test_float_initial_condition(self):
        model = SIR()
        with pytest.raises(ValidationError, match="integer"):
            model(
                inits=[990, 10.5, 0],
                trange=[0, 100],
                totpop=1000,
                params={"beta": 0.3, "gamma": 0.1},
                validate=True,
            )

    def test_invalid_reps(self):
        model = SIR()
        with pytest.raises(ValidationError, match="reps"):
            model(
                inits=[990, 10, 0],
                trange=[0, 100],
                totpop=1000,
                params={"beta": 0.3, "gamma": 0.1},
                reps=0,
            )

    def test_invalid_trange(self):
        model = SIR()
        with pytest.raises(ValidationError, match="time range"):
            model(
                inits=[990, 10, 0],
                trange=[100, 0],
                totpop=1000,
                params={"beta": 0.3, "gamma": 0.1},
            )

    def test_negative_parameter(self):
        model = SIR()
        with pytest.raises(ValidationError, match="non-negative"):
            model(
                inits=[990, 10, 0],
                trange=[0, 100],
                totpop=1000,
                params={"beta": -0.3, "gamma": 0.1},
                validate=True,
            )


# ============================================================
# Concrete model tests
# ============================================================


class TestSIRModel:
    """Tests for the stochastic SIR model."""

    def test_single_replicate(self):
        model = SIR()
        model(
            inits=[990, 10, 0],
            trange=[0, 100],
            totpop=1000,
            params={"beta": 0.3, "gamma": 0.1},
            reps=1,
            seed=42,
            n_points=50,
        )
        assert "S" in model.traces
        assert "I" in model.traces
        assert "R" in model.traces
        assert "time" in model.traces
        assert model.traces["time"].shape == (50,)
        assert model.traces["S"].shape == (50,)
        assert model.traces["S"][0] == 990
        assert model.traces["I"][0] == 10
        assert model.traces["R"][0] == 0

    def test_multi_replicate_traces_shape(self):
        model = SIR()
        model(
            inits=[990, 10, 0],
            trange=[0, 100],
            totpop=1000,
            params={"beta": 0.3, "gamma": 0.1},
            reps=50,
            seed=42,
            n_points=100,
        )
        assert model.traces["S"].shape == (50, 100)
        assert model.traces["I"].shape == (50, 100)
        assert model.traces["time"].shape == (100,)

    def test_conservation_across_replicates(self):
        model = SIR()
        model(
            inits=[990, 10, 0],
            trange=[0, 100],
            totpop=1000,
            params={"beta": 0.3, "gamma": 0.1},
            reps=10,
            seed=42,
            n_points=50,
        )
        for rep_i in range(10):
            S = model.traces["S"][rep_i]
            I = model.traces["I"][rep_i]
            R = model.traces["R"][rep_i]
            total = S + I + R
            np.testing.assert_array_equal(total, 1000)

    def test_reproducibility(self):
        model1 = SIR()
        model1(
            inits=[990, 10, 0],
            trange=[0, 50],
            totpop=1000,
            params={"beta": 0.3, "gamma": 0.1},
            reps=5,
            seed=123,
        )
        model2 = SIR()
        model2(
            inits=[990, 10, 0],
            trange=[0, 50],
            totpop=1000,
            params={"beta": 0.3, "gamma": 0.1},
            reps=5,
            seed=123,
        )
        np.testing.assert_array_equal(model1.traces["S"], model2.traces["S"])

    def test_r0_property(self):
        model = SIR()
        assert model.R0 is None
        model(
            inits=[990, 10, 0],
            trange=[0, 10],
            totpop=1000,
            params={"beta": 0.3, "gamma": 0.1},
            reps=1,
            seed=42,
        )
        assert pytest.approx(3.0) == model.R0

    def test_transitions_matrix(self):
        model = SIR()
        tmat = model.transitions()
        assert tmat.shape == (3, 2)
        assert tmat.dtype == np.int64
        np.testing.assert_array_equal(tmat, [[-1, 0], [1, -1], [0, 1]])

    def test_propensity_function(self):
        model = SIR()
        state = np.array([990, 10, 0])
        params = {"beta": 0.3, "gamma": 0.1, "N": 1000}
        a = model.propensity(params, state)
        assert a.shape == (2,)
        assert a[0] == pytest.approx(0.3 * 990 * 10 / 1000)
        assert a[1] == pytest.approx(0.1 * 10)

    def test_events_defined(self):
        model = SIR()
        assert "infection" in model.events
        assert "recovery" in model.events
        assert len(model.events) == 2


class TestSISModel:
    """Tests for the stochastic SIS model."""

    def test_single_replicate(self):
        model = SIS()
        model(
            inits=[490, 10],
            trange=[0, 100],
            totpop=500,
            params={"beta": 0.3, "gamma": 0.1},
            reps=1,
            seed=42,
        )
        assert model.traces["S"].shape[0] == 100
        assert model.traces["I"].shape[0] == 100

    def test_conservation(self):
        model = SIS()
        model(
            inits=[490, 10],
            trange=[0, 100],
            totpop=500,
            params={"beta": 0.3, "gamma": 0.1},
            reps=10,
            seed=42,
        )
        for i in range(10):
            total = model.traces["S"][i] + model.traces["I"][i]
            np.testing.assert_array_equal(total, 500)

    def test_transitions_matrix(self):
        model = SIS()
        tmat = model.transitions()
        assert tmat.shape == (2, 2)
        np.testing.assert_array_equal(tmat, [[-1, 1], [1, -1]])


class TestSIRSModel:
    """Tests for the stochastic SIRS model."""

    def test_single_replicate(self):
        model = SIRS()
        model(
            inits=[990, 10, 0],
            trange=[0, 100],
            totpop=1000,
            params={"beta": 0.3, "gamma": 0.1, "xi": 0.01},
            reps=1,
            seed=42,
        )
        assert model.traces["S"].shape[0] == 100

    def test_conservation(self):
        model = SIRS()
        model(
            inits=[990, 10, 0],
            trange=[0, 100],
            totpop=1000,
            params={"beta": 0.3, "gamma": 0.1, "xi": 0.01},
            reps=5,
            seed=42,
        )
        for i in range(5):
            total = model.traces["S"][i] + model.traces["I"][i] + model.traces["R"][i]
            np.testing.assert_array_equal(total, 1000)

    def test_three_events(self):
        model = SIRS()
        assert len(model.events) == 3
        assert "waning" in model.events


class TestSEIRModel:
    """Tests for the stochastic SEIR model."""

    def test_single_replicate(self):
        model = SEIR()
        model(
            inits=[990, 0, 10, 0],
            trange=[0, 100],
            totpop=1000,
            params={"beta": 0.3, "gamma": 0.1, "epsilon": 0.5},
            reps=1,
            seed=42,
        )
        assert model.traces["S"].shape[0] == 100
        assert model.traces["E"].shape[0] == 100
        assert model.traces["I"].shape[0] == 100
        assert model.traces["R"].shape[0] == 100

    def test_conservation(self):
        model = SEIR()
        model(
            inits=[990, 0, 10, 0],
            trange=[0, 100],
            totpop=1000,
            params={"beta": 0.3, "gamma": 0.1, "epsilon": 0.5},
            reps=5,
            seed=42,
        )
        for i in range(5):
            total = (
                model.traces["S"][i]
                + model.traces["E"][i]
                + model.traces["I"][i]
                + model.traces["R"][i]
            )
            np.testing.assert_array_equal(total, 1000)


# ============================================================
# Accessor method tests
# ============================================================


class TestAccessors:
    """Tests for CTMCModel accessor methods."""

    @pytest.fixture
    def sir_model_multi(self):
        model = SIR()
        model(
            inits=[990, 10, 0],
            trange=[0, 100],
            totpop=1000,
            params={"beta": 0.3, "gamma": 0.1},
            reps=20,
            seed=42,
            n_points=50,
        )
        return model

    def test_get_replicate(self, sir_model_multi):
        rep = sir_model_multi.get_replicate(0)
        assert "S" in rep
        assert "I" in rep
        assert "R" in rep
        assert "time" in rep
        assert rep["S"].shape == (50,)

    def test_get_replicate_out_of_range(self, sir_model_multi):
        with pytest.raises(IndexError):
            sir_model_multi.get_replicate(20)

    def test_get_mean(self, sir_model_multi):
        mean = sir_model_multi.get_mean()
        assert mean["S"].shape == (50,)
        assert mean["I"].shape == (50,)
        assert mean["R"].shape == (50,)

    def test_get_variance(self, sir_model_multi):
        var = sir_model_multi.get_variance()
        assert var["S"].shape == (50,)
        assert (var["S"] >= 0).all()

    def test_get_quantiles(self, sir_model_multi):
        q = sir_model_multi.get_quantiles([0.025, 0.975])
        assert 0.025 in q
        assert 0.975 in q
        assert q[0.025]["I"].shape == (50,)

    def test_get_event_times_all(self, sir_model_multi):
        et = sir_model_multi.get_event_times()
        assert "infection" in et
        assert "recovery" in et
        assert len(et["infection"]) > 0
        assert len(et["recovery"]) > 0

    def test_get_event_times_specific(self, sir_model_multi):
        infection_times = sir_model_multi.get_event_times("infection")
        assert isinstance(infection_times, list)
        assert len(infection_times) > 0

    def test_get_event_times_unknown(self, sir_model_multi):
        with pytest.raises(ValueError, match="Unknown event"):
            sir_model_multi.get_event_times("nonexistent")

    def test_summary(self, sir_model_multi):
        s = sir_model_multi.summary()
        assert s["model"] == "SIR CTMC"
        assert "peak_I_mean" in s
        assert "extinction_probability" in s
        assert "reps" in s
        assert s["reps"] == 20

    def test_to_dataframe_mean(self, sir_model_multi):
        df = sir_model_multi.to_dataframe()
        assert "S" in df.columns
        assert "I" in df.columns
        assert "time" in df.columns
        assert len(df) == 50

    def test_to_dataframe_replicate(self, sir_model_multi):
        df = sir_model_multi.to_dataframe(replicate=0)
        assert len(df) == 50

    def test_reset(self, sir_model_multi):
        sir_model_multi.reset()
        assert sir_model_multi.traces == {}
        assert sir_model_multi._trajectories == []


# ============================================================
# Single replicate convenience tests
# ============================================================


class TestSingleReplicateConvenience:
    """Tests that single-replicate runs return squeezed (1D) arrays."""

    def test_squeezed_traces(self):
        model = SIR()
        model(
            inits=[990, 10, 0],
            trange=[0, 50],
            totpop=1000,
            params={"beta": 0.3, "gamma": 0.1},
            reps=1,
            seed=42,
            n_points=50,
        )
        assert model.traces["S"].ndim == 1
        assert model.traces["I"].ndim == 1
        assert model.traces["S"].shape == (50,)

    def test_variance_zero_for_single_rep(self):
        model = SIR()
        model(
            inits=[990, 10, 0],
            trange=[0, 50],
            totpop=1000,
            params={"beta": 0.3, "gamma": 0.1},
            reps=1,
            seed=42,
        )
        var = model.get_variance()
        np.testing.assert_array_equal(var["S"], np.zeros(100))


# ============================================================
# Model without prior run tests
# ============================================================


class TestNoSimulation:
    """Tests for calling accessors before simulation."""

    def test_get_mean_no_sim(self):
        model = SIR()
        with pytest.raises(ValueError, match="No simulation"):
            model.get_mean()

    def test_get_replicate_no_sim(self):
        model = SIR()
        with pytest.raises(ValueError, match="No simulation"):
            model.get_replicate(0)

    def test_summary_no_sim(self):
        model = SIR()
        with pytest.raises(ValueError, match="No simulation"):
            model.summary()

    def test_get_event_times_no_sim(self):
        model = SIR()
        with pytest.raises(ValueError, match="No simulation"):
            model.get_event_times()


# ============================================================
# Integration: comparison with deterministic model
# ============================================================


class TestDeterministicComparison:
    """Verify stochastic mean converges toward deterministic for large N."""

    def test_stochastic_mean_approximates_ode(self):
        from epimodels.continuous.models import SIR as ODE_SIR

        N = 10000
        I0 = 100
        params = {"beta": 0.3, "gamma": 0.1}

        ode = ODE_SIR()
        ode(
            inits=[N - I0, I0, 0],
            trange=[0, 80],
            totpop=N,
            params=params,
        )

        sto = SIR()
        sto(
            inits=[N - I0, I0, 0],
            trange=[0, 80],
            totpop=N,
            params=params,
            reps=200,
            seed=42,
            n_points=100,
        )

        sto_mean = sto.get_mean()

        ode_t = ode.traces["time"]
        sto_t = sto_mean["time"]

        ode_I = np.interp(sto_t, ode_t, ode.traces["I"])
        sto_I = sto_mean["I"]

        diff = np.abs(ode_I - sto_I)
        rel_diff = diff / (np.maximum(ode_I, 1))

        assert np.median(rel_diff) < 0.15


# ============================================================
# Import tests
# ============================================================


class TestImports:
    """Verify public API is accessible from expected locations."""

    def test_import_from_ctmc(self):
        pass

    def test_import_from_stochastic(self):
        pass

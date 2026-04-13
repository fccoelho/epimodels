import numpy as np
import pytest
from epimodels.continuous.models import SISLogistic, SIRSNonAutonomous, NeipelHeterogeneousSIR


def test_SISLogistic():
    """Test SIS model with logistic growth."""
    model = SISLogistic()
    model([900, 100], [0, 100], 1000, {"R0": 3.0, "gamma": 0.1, "r": 0.01, "k": 10000})
    assert len(model.traces) == 3  # S, I, time
    assert "S" in model.traces
    assert "I" in model.traces
    assert model.traces["S"][0] == 900
    assert model.traces["I"][0] == 100


def test_SISLogistic_R0():
    """Test SISLogistic R0 computation."""
    model = SISLogistic()
    model([900, 100], [0, 100], 1000, {"R0": 4.0, "gamma": 0.1, "r": 0.01, "k": 10000})
    assert model.R0 == 4.0


def test_SISLogistic_parameters():
    """Test SISLogistic parameters are set correctly after running model."""
    model = SISLogistic()
    model([900, 100], [0, 100], 1000, {"R0": 4.0, "gamma": 0.1, "r": 0.01, "k": 10000})
    assert model.param_values["R0"] == 4.0
    assert model.param_values["gamma"] == 0.1
    assert model.param_values["r"] == 0.01
    assert model.param_values["k"] == 10000


def test_SISLogistic_simulation():
    """Test SISLogistic simulation produces reasonable results."""
    model = SISLogistic()
    model([950, 50], [0, 200], 1000, {"R0": 3.0, "gamma": 0.1, "r": 0.02, "k": 10000})
    assert len(model.traces["time"]) > 0
    assert len(model.traces["S"]) == len(model.traces["time"])
    assert len(model.traces["I"]) == len(model.traces["time"])


def test_SIRSNonAutonomous():
    """Test SIRS model with time-dependent parameters."""
    model = SIRSNonAutonomous()

    # Time-dependent parameters
    def alpha(t):
        return 0.05

    def beta(t):
        return 0.3 * (1 + 0.1 * np.sin(t / 10))

    def gamma(t):
        return 0.1

    model([950, 50, 0], [0, 100], 1000, {"alpha": alpha, "beta": beta, "gamma": gamma})
    assert len(model.traces) == 4  # S, I, R, time
    assert "S" in model.traces
    assert "I" in model.traces
    assert "R" in model.traces


def test_SIRSNonAutonomous_parameters():
    """Test SIRSNonAutonomous parameters are set correctly after running model."""
    model = SIRSNonAutonomous()

    def alpha(t):
        return 0.05

    def beta(t):
        return 0.3

    def gamma(t):
        return 0.1

    model([950, 50, 0], [0, 100], 1000, {"alpha": alpha, "beta": beta, "gamma": gamma})
    assert callable(model.param_values["alpha"])
    assert callable(model.param_values["beta"])
    assert callable(model.param_values["gamma"])


def test_SIRSNonAutonomous_time_dependent():
    """Test SIRSNonAutonomous with fully time-dependent parameters."""
    model = SIRSNonAutonomous()

    model(
        [950, 50, 0],
        [0, 100],
        1000,
        {
            "alpha": lambda t: 0.05 + 0.01 * t / 100,
            "beta": lambda t: 0.4 - 0.1 * t / 100,
            "gamma": lambda t: 0.1 + 0.02 * t / 100,
        },
    )

    assert len(model.traces["time"]) > 0
    # Check that traces exist
    assert "S" in model.traces
    assert "I" in model.traces


def test_NeipelHeterogeneousSIR():
    """Test Neipel Heterogeneous SIR model."""
    model = NeipelHeterogeneousSIR()
    model([10, 0], [0, 100], 1000, {"beta": 0.3, "gamma": 0.1, "alpha": 0.5, "I0": 10, "N": 1000})
    assert len(model.traces) == 3  # I, tau, time
    assert "I" in model.traces
    assert "tau" in model.traces


def test_NeipelHeterogeneousSIR_parameters():
    """Test NeipelHeterogeneousSIR parameters are set correctly after running model."""
    model = NeipelHeterogeneousSIR()
    model([10, 0], [0, 100], 1000, {"beta": 0.3, "gamma": 0.1, "alpha": 0.5, "I0": 10, "N": 1000})
    assert model.param_values["beta"] == 0.3
    assert model.param_values["gamma"] == 0.1
    assert model.param_values["alpha"] == 0.5
    assert model.param_values["I0"] == 10
    # N is passed to model call but not stored in param_values


def test_NeipelHeterogeneousSIR_R0():
    """Test NeipelHeterogeneousSIR R0 computation."""
    model = NeipelHeterogeneousSIR()
    model([10, 0], [0, 100], 1000, {"beta": 0.4, "gamma": 0.1, "alpha": 0.5, "I0": 10, "N": 1000})
    assert model.R0 == 4.0


def test_NeipelHeterogeneousSIR_susceptible():
    """Test NeipelHeterogeneousSIR susceptible formula."""
    model = NeipelHeterogeneousSIR()
    N = 1000
    I0 = 10
    alpha = 0.5
    tau = 0.0

    S = model.susceptible(tau, N, I0, alpha)
    assert S == pytest.approx(N - I0)  # At tau=0, S = N - I0


def test_NeipelHeterogeneousSIR_removed():
    """Test NeipelHeterogeneousSIR removed computation."""
    model = NeipelHeterogeneousSIR()
    N = 1000
    I0 = 10
    alpha = 0.5
    tau = 1.0
    I = 50

    S = model.susceptible(tau, N, I0, alpha)
    R = model.removed(I, tau, N, I0, alpha)

    assert S + I + R == pytest.approx(N)


def test_NeipelHeterogeneousSIR_simulation():
    """Test NeipelHeterogeneousSIR simulation produces reasonable results."""
    model = NeipelHeterogeneousSIR()
    model([50, 0], [0, 100], 1000, {"beta": 0.3, "gamma": 0.1, "alpha": 0.5, "I0": 50, "N": 1000})

    assert len(model.traces["time"]) > 0
    # I should grow initially then decrease
    assert model.traces["I"][0] == 50
    # tau should always increase
    assert model.traces["tau"][-1] > model.traces["tau"][0]


def test_NeipelHeterogeneousSIR_diagram():
    """Test NeipelHeterogeneousSIR has a diagram."""
    model = NeipelHeterogeneousSIR()
    assert hasattr(model, "diagram")
    assert "flowchart" in model.diagram


def test_NeipelHeterogeneousSIR_model_type():
    """Test NeipelHeterogeneousSIR has correct model type."""
    model = NeipelHeterogeneousSIR()
    assert model.model_type == "NeipelHeterogeneousSIR"


def test_SISLogistic_model_type():
    """Test SISLogistic has correct model type."""
    model = SISLogistic()
    assert model.model_type == "SIS_logistic"


def test_SIRSNonAutonomous_model_type():
    """Test SIRSNonAutonomous has correct model type."""
    model = SIRSNonAutonomous()
    assert model.model_type == "SIRS Non-Autonomous"

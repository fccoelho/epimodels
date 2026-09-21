"""Tests for model serialization (epimodels.io)."""

import json

import numpy as np
import pytest

from epimodels.continuous import SIR as ContinuousSIR
from epimodels.discrete import SIR as DiscreteSIR
from epimodels.io import (
    load_model,
    model_from_dict,
    model_to_dict,
    save_model,
)
from epimodels.stochastic.CTMC import SIR as CTMCSIR

PARAMS = {"beta": 2.0, "gamma": 0.1}


def run_continuous():
    m = ContinuousSIR()
    m([1000, 1, 0], [0, 50], 1001, PARAMS, t_eval=np.linspace(0, 50, 51))
    return m


class TestModelToDict:
    def test_spec_fields(self):
        m = run_continuous()
        d = model_to_dict(m)
        assert d["model"] == "SIR"
        assert d["family"] == "continuous"
        assert set(d["state_variables"]) == {"S", "I", "R"}
        assert "beta" in d["parameters"]
        assert d["param_values"]["beta"] == 2.0
        assert "traces" not in d

    def test_family_detection(self):
        assert model_to_dict(run_continuous())["family"] == "continuous"
        assert model_to_dict(DiscreteSIR())["family"] == "discrete"
        assert model_to_dict(CTMCSIR())["family"] == "stochastic"

    def test_json_serializable(self):
        m = run_continuous()
        d = model_to_dict(m, include_traces=True)
        roundtrip = json.loads(json.dumps(d))  # must not raise
        assert roundtrip["traces"]["I"][0] == pytest.approx(1.0)

    def test_numpy_scalars_plain(self):
        m = run_continuous()
        m.param_values["beta"] = np.float64(3.5)
        d = model_to_dict(m)
        assert isinstance(d["param_values"]["beta"], float)


class TestModelFromDict:
    def test_roundtrip_continuous(self):
        m = run_continuous()
        clone = model_from_dict(model_to_dict(m))
        assert isinstance(clone, ContinuousSIR)
        assert clone.param_values == m.param_values
        assert set(clone.state_variables) == set(m.state_variables)
        assert clone.traces == {}

    def test_roundtrip_with_traces(self):
        m = run_continuous()
        clone = model_from_dict(model_to_dict(m, include_traces=True))
        np.testing.assert_allclose(clone.traces["I"], m.traces["I"])
        np.testing.assert_allclose(clone.traces["time"], m.traces["time"])

    def test_roundtrip_discrete(self):
        m = DiscreteSIR()
        m([1000, 1, 0], [0, 50], 1001, PARAMS)
        clone = model_from_dict(model_to_dict(m))
        assert isinstance(clone, DiscreteSIR)
        assert clone.param_values == m.param_values

    def test_roundtrip_stochastic(self):
        m = CTMCSIR()
        m([990, 10, 0], [0, 50], 1000, PARAMS, reps=2, seed=1)
        clone = model_from_dict(model_to_dict(m, include_traces=True))
        assert isinstance(clone, CTMCSIR)
        np.testing.assert_allclose(clone.traces["I"], m.traces["I"])

    def test_restored_clone_is_runnable(self):
        m = run_continuous()
        clone = model_from_dict(model_to_dict(m))
        clone([1000, 1, 0], [0, 50], 1001, PARAMS)
        assert "I" in clone.traces

    def test_missing_schema_flag_rejected(self):
        with pytest.raises(ValueError, match="epimodels_model"):
            model_from_dict({"model": "SIR"})


class TestSaveLoad:
    def test_json_roundtrip(self, tmp_path):
        m = run_continuous()
        path = save_model(m, tmp_path / "sir.json", include_traces=True)
        clone = load_model(path)
        np.testing.assert_allclose(clone.traces["I"], m.traces["I"])

    def test_yaml_roundtrip(self, tmp_path):
        m = run_continuous()
        path = save_model(m, tmp_path / "sir.yaml", include_traces=True)
        clone = load_model(path)
        np.testing.assert_allclose(clone.traces["I"], m.traces["I"])

    def test_yaml_snake(self, tmp_path):
        m = run_continuous()
        path = save_model(m, tmp_path / "sir.yml")
        assert path.exists()

    def test_unsupported_extension(self, tmp_path):
        m = run_continuous()
        with pytest.raises(ValueError, match="extension"):
            save_model(m, tmp_path / "sir.toml")

    def test_load_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_model(tmp_path / "ghost.json")

    def test_saved_file_is_valid_json(self, tmp_path):
        m = run_continuous()
        path = save_model(m, tmp_path / "sir.json")
        data = json.loads(path.read_text())
        assert data["model"] == "SIR"

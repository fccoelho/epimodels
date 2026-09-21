"""Tests for the model registry."""

import pytest

from epimodels import get_model as top_level_get_model
from epimodels import list_models as top_level_list_models
from epimodels.continuous import ContinuousModel
from epimodels.continuous.models import SIR as ContinuousSIR
from epimodels.discrete.models import SIR as DiscreteSIR
from epimodels.registry import (
    get_model,
    list_models,
    register_model,
    unregister_model,
)


class TestGetModel:
    def test_get_continuous_sir(self):
        assert get_model("SIR", family="continuous") is ContinuousSIR

    def test_get_discrete_sir(self):
        assert get_model("SIR", family="discrete") is DiscreteSIR

    def test_ambiguity_resolved_by_precedence(self):
        """family='any' should resolve deterministically (continuous first)."""
        assert get_model("SIR") is ContinuousSIR

    def test_case_sensitive_lookup(self):
        with pytest.raises(KeyError):
            get_model("sir", family="continuous")

    def test_unknown_model_raises_with_suggestions(self):
        with pytest.raises(KeyError, match="Available models"):
            get_model("NotAModel")

    def test_unknown_family_raises(self):
        with pytest.raises(ValueError, match="Unknown family"):
            get_model("SIR", family="quantum")

    def test_top_level_wrapper(self):
        assert top_level_get_model("SIR", family="continuous") is ContinuousSIR

    def test_stochastic_family(self):
        from epimodels.stochastic.CTMC.models import SIR as CTMCSIR

        assert get_model("SIR", family="stochastic") is CTMCSIR


class TestListModels:
    def test_lists_all_families(self):
        listing = list_models()
        for family in ("continuous", "discrete", "stochastic"):
            assert family in listing
            assert "SIR" in listing[family]

    def test_single_family(self):
        listing = list_models(family="discrete")
        assert set(listing) == {"discrete"}
        assert "SEQIAHR" in listing["discrete"]

    def test_unknown_family_raises(self):
        with pytest.raises(ValueError):
            list_models(family="quantum")

    def test_top_level_wrapper(self):
        listing = top_level_list_models("continuous")
        assert "Dengue4Strain" in listing["continuous"]


class TestCustomRegistration:
    def teardown_method(self):
        unregister_model("MyTestModel", "custom")

    def test_register_and_get(self):
        @register_model("MyTestModel", family="custom")
        class MyTestModel(ContinuousModel):
            pass

        assert get_model("MyTestModel", family="custom") is MyTestModel
        assert "MyTestModel" in list_models(family="custom")["custom"]

    def test_register_defaults_to_class_name(self):
        @register_model()
        class AnotherTestModel(ContinuousModel):
            pass

        try:
            assert get_model("AnotherTestModel", family="custom") is AnotherTestModel
        finally:
            unregister_model("AnotherTestModel", "custom")


class TestInstantiation:
    def test_get_and_run(self):
        """End-to-end: registry lookup produces a runnable model."""
        SIR = get_model("SIR", family="continuous")
        model = SIR()
        model([1000, 1, 0], [0, 50], 1001, {"beta": 2, "gamma": 0.1})
        assert "I" in model.traces
        assert pytest.approx(20.0) == model.R0

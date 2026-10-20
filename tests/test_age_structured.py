"""Tests for the age-structured MSEIR model and the contact-matrix helpers."""

import numpy as np
import pytest

from epimodels import get_model
from epimodels.continuous import AgeStructuredSEIR
from epimodels.tools.contact_matrix import (
    DEFAULT_AGE_BOUNDS,
    age_group_labels,
    aggregate_contact_matrix,
    check_reciprocity,
    enforce_reciprocity,
    polymod_like_contact_matrix,
)

# A crude Brazilian age structure on the default 21-group grid: small birth
# cohorts, a large working-age block and a small elderly block.
POP = np.array([2.5e6] + [2.6e6] * 4 + [13e6] * 15 + [3e6])


def _simple_model(**kwargs):
    """Single-age-group model with unit contact rate (R0 = beta/(gamma+mu))."""
    return AgeStructuredSEIR(age_bounds=[0], contact_matrix=[[1.0]], **kwargs)


# ----------------------------------------------------------------------
# Contact matrices
# ----------------------------------------------------------------------
def test_polymod_matrix_shape_and_peak():
    matrix = polymod_like_contact_matrix(population=POP)
    assert matrix.shape == (len(DEFAULT_AGE_BOUNDS),) * 2
    assert np.all(matrix >= 0)
    # Busiest age group has exactly `peak_contacts` contacts
    assert matrix.sum(axis=1).max() == pytest.approx(18.0)


def test_polymod_matrix_is_reciprocal_by_construction():
    matrix = polymod_like_contact_matrix(population=POP)
    assert check_reciprocity(matrix, POP)
    counts = POP[:, None] * matrix
    # machine-precision agreement, not just within tolerance
    assert np.abs(counts - counts.T).max() / np.abs(counts).max() < 1e-15


def test_polymod_matrix_peak_is_in_school_ages():
    matrix = polymod_like_contact_matrix(population=POP)
    rows = matrix.sum(axis=1)
    labels = age_group_labels(DEFAULT_AGE_BOUNDS)
    busiest = labels[int(rows.argmax())]
    assert busiest in {"5-10", "10-15"}
    # infants and the elderly mix less than school-age children
    assert rows[0] < rows[6]
    assert rows[-1] < rows[0]


def test_polymod_matrix_validation():
    with pytest.raises(ValueError):
        polymod_like_contact_matrix(age_bounds=[5, 1])
    with pytest.raises(ValueError):
        polymod_like_contact_matrix(age_bounds=[0, 1], population=[1.0])
    with pytest.raises(ValueError):
        polymod_like_contact_matrix(age_bounds=[0, 1], population=[0.0, 1.0])


def test_enforce_reciprocity_is_exact():
    matrix = polymod_like_contact_matrix(population=np.ones(len(DEFAULT_AGE_BOUNDS)))
    assert check_reciprocity(matrix, POP) is False  # only reciprocal for uniform
    fixed = enforce_reciprocity(matrix, POP)
    assert check_reciprocity(fixed, POP)


def test_enforce_reciprocity_rejects_empty_group():
    matrix = np.ones((2, 2))
    with pytest.raises(ValueError, match="non-positive population"):
        enforce_reciprocity(matrix, [1.0, 0.0])


def test_check_reciprocity_detects_asymmetry():
    matrix = np.array([[1.0, 1.0], [1.0, 1.0]])
    assert check_reciprocity(matrix, [1.0, 1.0])
    matrix[0, 1] = 2.0
    assert not check_reciprocity(matrix, [1.0, 1.0])


def test_aggregate_contact_matrix_preserves_reciprocity():
    matrix = polymod_like_contact_matrix(population=POP)
    mapping = np.array([0] * 5 + [1] + [2] * 5 + [3] * 10)  # 0-5, 5-10, 10-35, 35+
    coarsened = aggregate_contact_matrix(matrix, POP, mapping)
    assert coarsened.shape == (4, 4)
    pooled = np.bincount(mapping, weights=POP)
    assert check_reciprocity(coarsened, pooled)


def test_aggregate_contact_matrix_validation():
    matrix = np.ones((3, 3))
    with pytest.raises(ValueError):
        aggregate_contact_matrix(matrix, [1.0, 1.0], [0, 0, 0])
    with pytest.raises(ValueError):
        aggregate_contact_matrix(matrix, [1.0, 1.0, 1.0], [0, 1, 5], n_targets=2)


# ----------------------------------------------------------------------
# Structure of the model
# ----------------------------------------------------------------------
def test_model_is_registered():
    assert get_model("AgeStructuredSEIR", family="continuous") is AgeStructuredSEIR


def test_state_variables_and_parameters():
    model = AgeStructuredSEIR()
    assert model.n_ages == len(DEFAULT_AGE_BOUNDS)
    assert model.dimension == 5 * model.n_ages == len(model.state_variables)
    assert list(model.state_variables)[0] == "M_0"
    assert list(model.state_variables)[-1] == f"R_{model.n_ages - 1}"
    assert set(model.parameters) == {
        "beta",
        "epsilon",
        "gamma",
        "omega",
        "births",
        "mu",
        "vaccination",
    }
    assert "diagram" in dir(model)


def test_ageing_rates_match_group_widths():
    model = AgeStructuredSEIR(age_bounds=[0, 1, 5], time_per_year=365.25)
    assert model.ageing_rates[0] == pytest.approx(1.0 / 365.25)
    assert model.ageing_rates[1] == pytest.approx(1.0 / (4 * 365.25))
    # the open-ended group has no outflow
    assert model.ageing_rates[-1] == 0.0


def test_invalid_configuration():
    with pytest.raises(ValueError):
        AgeStructuredSEIR(age_bounds=[0, 2, 1])
    with pytest.raises(ValueError):
        AgeStructuredSEIR(age_bounds=[0, 1], contact_matrix=[[1.0]])
    with pytest.raises(ValueError):
        AgeStructuredSEIR(age_bounds=[0, 1], contact_matrix=[[-1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError):
        AgeStructuredSEIR(time_per_year=0.0)
    with pytest.raises(ValueError):
        AgeStructuredSEIR(age_bounds=[0, 1], vaccination_age_index=5)


def test_compartment_labels():
    model = AgeStructuredSEIR(age_bounds=[0, 1, 2])
    assert model.compartment_labels("I") == ["I_0", "I_1", "I_2"]
    with pytest.raises(ValueError):
        model.compartment_labels("Z")


# ----------------------------------------------------------------------
# Initial conditions
# ----------------------------------------------------------------------
def test_initial_conditions_conserve_population():
    model = AgeStructuredSEIR()
    inits = model.initial_conditions(POP, infectious=[10.0] + [0.0] * 20)
    assert sum(inits) == pytest.approx(POP.sum())
    assert np.all(np.asarray(inits) >= 0)


def test_initial_conditions_requires_population_argument():
    model = AgeStructuredSEIR()  # no population at construction
    with pytest.raises(ValueError, match="reference population"):
        model._contact_operator()


def test_initial_conditions_reject_oversubscription():
    model = AgeStructuredSEIR(age_bounds=[0, 1], contact_matrix=[[1.0, 1.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="more infectious"):
        model.initial_conditions([100.0, 100.0], infectious=[200.0, 0.0])


# ----------------------------------------------------------------------
# R0 and the next-generation matrix
# ----------------------------------------------------------------------
def test_R0_none_before_simulation():
    assert AgeStructuredSEIR().R0 is None


def test_R0_reduces_to_beta_over_gamma_for_one_group():
    model = _simple_model()
    model([0, 99, 1, 0, 0], [0, 10], 100, {"beta": 0.3, "gamma": 0.1, "epsilon": 0.2})
    assert model.R0 == pytest.approx(3.0)


def test_R0_includes_mortality_in_infectious_period():
    model = _simple_model()
    model(
        [0, 99, 1, 0, 0],
        [0, 10],
        100,
        {"beta": 0.3, "gamma": 0.1, "epsilon": 0.2, "mu": 0.05},
    )
    assert model.R0 == pytest.approx(0.3 / 0.15)


def test_scale_beta_for_r0_roundtrip():
    model = AgeStructuredSEIR(population=POP)
    beta = model.scale_beta_for_r0(12.0, gamma=0.1)
    inits = model.initial_conditions(POP, infectious=[1e-3] + [0.0] * 20)
    model(inits, [0, 30], float(POP.sum()), {"beta": beta, "epsilon": 0.125, "gamma": 0.1})
    assert model.R0 == pytest.approx(12.0)


def test_next_generation_matrix_equals_transposed_contacts_when_reciprocal():
    model = AgeStructuredSEIR(population=POP)
    inits = model.initial_conditions(POP, infectious=[1.0] + [0.0] * 20)
    model(inits, [0, 1], float(POP.sum()), {"beta": 0.5, "epsilon": 0.2, "gamma": 0.1})
    operator = model.contact_operator
    # detailed balance means N_i C_ij / N_j == C_ji
    assert np.allclose(operator, model._resolved_matrix.T)


def test_R0_uses_eigvals_not_eigvalsh():
    """A non-reciprocal matrix makes the NGM non-symmetric.

    ``eigvalsh`` reads a single triangle and returns a different spectral
    radius, so the model must not use it.
    """
    matrix = polymod_like_contact_matrix(population=np.ones(len(DEFAULT_AGE_BOUNDS)))
    matrix = matrix.copy()
    matrix[0, 6] *= 4.0  # break detailed balance for this population
    model = AgeStructuredSEIR(contact_matrix=matrix, population=POP, enforce_reciprocity=False)
    inits = model.initial_conditions(POP, infectious=[1.0] + [0.0] * 20)
    with pytest.warns(UserWarning, match="detailed balance"):
        model(inits, [0, 1], float(POP.sum()), {"beta": 0.5, "epsilon": 0.2, "gamma": 0.1})

    operator = model.contact_operator
    assert not np.allclose(operator, operator.T)
    correct = np.abs(np.linalg.eigvals(operator)).max()
    wrong = np.abs(np.linalg.eigvalsh(operator)).max()
    assert model.R0 == pytest.approx(0.5 / 0.1 * correct)
    assert not np.isclose(correct, wrong)


def test_below_threshold_epidemic_dies_out():
    model = AgeStructuredSEIR(population=POP)
    beta = model.scale_beta_for_r0(0.7, gamma=0.1)
    inits = model.initial_conditions(POP, infectious=[100.0] + [0.0] * 20)
    model(inits, [0, 2000], float(POP.sum()), {"beta": beta, "epsilon": 0.125, "gamma": 0.1})
    assert model.summary()["attack_rate"] < 1e-6


def test_above_threshold_epidemic_spreads():
    model = AgeStructuredSEIR(population=POP)
    beta = model.scale_beta_for_r0(6.0, gamma=0.1)
    inits = model.initial_conditions(POP, infectious=[100.0] + [0.0] * 20)
    model(inits, [0, 1000], float(POP.sum()), {"beta": beta, "epsilon": 0.125, "gamma": 0.1})
    stats = model.summary()
    assert stats["attack_rate"] > 0.5
    assert stats["peak_I"] > 100.0


# ----------------------------------------------------------------------
# Conservation and demography
# ----------------------------------------------------------------------
def test_total_population_is_conserved_without_demography():
    model = AgeStructuredSEIR()
    inits = [1.0] * model.dimension
    model(inits, [0, 100], float(model.dimension), {"beta": 0.5, "epsilon": 0.2, "gamma": 0.2})
    total = sum(model.aggregate(c) for c in "MSEIR")
    assert np.abs(total - model.dimension).max() < 1e-9


def test_demography_follows_births_minus_deaths():
    model = AgeStructuredSEIR()
    inits = [1.0] * model.dimension
    n0 = float(model.dimension)
    mu, births = 0.001, 0.5
    model(
        inits,
        [0, 100],
        n0,
        {"beta": 0.0, "epsilon": 0.2, "gamma": 0.2, "mu": mu, "births": births},
    )
    total = sum(model.aggregate(c) for c in "MSEIR")
    time = model.traces["time"]
    expected = (n0 - births / mu) * np.exp(-mu * time) + births / mu
    assert np.abs(total - expected).max() < 1e-6


def test_births_enter_maternal_compartment():
    model = AgeStructuredSEIR(population=np.full(len(DEFAULT_AGE_BOUNDS), 100.0))
    inits = [0.0] * model.dimension
    model(
        inits,
        [0, 10],
        1000.0,
        {"beta": 0.0, "epsilon": 0.2, "gamma": 0.2, "births": 1.0},
    )
    assert model.aggregate("M")[-1] > 0
    assert model.aggregate("S")[-1] > 0  # maternal immunity wanes into S


# ----------------------------------------------------------------------
# Vaccination
# ----------------------------------------------------------------------
def test_vaccination_rate_for_coverage():
    model = AgeStructuredSEIR(age_bounds=[0, 1, 2], time_per_year=1.0)
    ageing = model.ageing_rates[0]
    assert model.vaccination_rate_for_coverage(0.5) == pytest.approx(ageing)
    assert model.vaccination_rate_for_coverage(0.9) == pytest.approx(9 * ageing)
    with pytest.raises(ValueError):
        model.vaccination_rate_for_coverage(1.0)
    with pytest.raises(ValueError):
        model.vaccination_rate_for_coverage(0.5, age_index=2)  # open-ended group


def test_vaccination_achieved_coverage():
    """Vaccinated + aged-out fractions of a cohort match the requested coverage."""
    for coverage in (0.0, 0.5, 0.9):
        model = AgeStructuredSEIR(
            age_bounds=[0, 1],
            contact_matrix=np.ones((2, 2)),
            population=[1e6, 1e-3],
            vaccination_age_index=0,
        )
        inits = model.initial_conditions([1e6, 1e-3], infectious=[0.0, 0.0])
        rate = model.vaccination_rate_for_coverage(coverage) if coverage else 0.0
        # Ten years leaves a fraction p * exp(-(v + a) * t) of the cohort still
        # in the vaccinated group, which is below 1e-8 for every p tested.
        model(
            inits,
            [0, 10 * 365.25],
            1e6 + 1e-3,
            {
                "beta": 0.0,
                "epsilon": 0.2,
                "gamma": 0.1,
                "omega": 0.0,
                "vaccination": rate,
            },
        )
        assert model.aggregate("R")[-1] / 1e6 == pytest.approx(coverage, abs=1e-6)
        assert model.aggregate("S")[-1] / 1e6 == pytest.approx(1 - coverage, abs=1e-6)


# ----------------------------------------------------------------------
# Post-processing
# ----------------------------------------------------------------------
def test_aggregate_and_summary():
    model = AgeStructuredSEIR(age_bounds=[0, 1, 2])
    model([0, 0, 0, 4, 4, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 20], 15.0, {"beta": 2.0, "epsilon": 0.2, "gamma": 0.2})
    assert model.aggregate("I").shape == model.traces["time"].shape
    stats = model.summary()
    for key in ("peak_I", "peak_time", "final_S", "final_R", "attack_rate"):
        assert key in stats
    with pytest.raises(ValueError):
        model.aggregate("Z")


def test_summary_requires_simulation():
    with pytest.raises(ValueError, match="No simulation results"):
        AgeStructuredSEIR().summary()


def test_plot_traces_aggregates_compartments():
    pytest.importorskip("matplotlib")
    model = _simple_model()
    model([0, 99, 1, 0, 0], [0, 30], 100, {"beta": 0.3, "gamma": 0.1, "epsilon": 0.2})
    plotted = model.plot_traces(["S", "I"])
    assert set(plotted) == {"S", "I"}
    assert len(plotted["I"]) == len(model.traces["time"])
    with pytest.raises(ValueError):
        model.plot_traces(["nope"])


def test_get_formulas_is_unsupported():
    from epimodels import FormulaExtractionError

    model = _simple_model()
    with pytest.raises(FormulaExtractionError):
        model.get_formulas()

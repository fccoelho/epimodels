"""Tests for network epidemic models."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from epimodels.network import NetworkModel, NetworkSIR, NetworkSIS

nx = pytest.importorskip("networkx")


@pytest.fixture
def graph():
    return nx.barabasi_albert_graph(200, 3, seed=0)


PARAMS = {"beta": 0.4, "gamma": 0.1}


class TestGraphInputs:
    def test_from_networkx(self, graph):
        m = NetworkSIR(graph)
        assert m.n_nodes == 200

    def test_from_adjacency_dict(self, graph):
        m = NetworkSIR({n: list(graph.neighbors(n)) for n in graph.nodes()})
        assert m.n_nodes == 200

    def test_from_adjacency_matrix(self, graph):
        m = NetworkSIR(nx.to_numpy_array(graph))
        assert m.n_nodes == 200

    def test_asymmetric_matrix_rejected(self):
        with pytest.raises(ValueError, match="symmetric"):
            NetworkSIR(np.array([[0, 1], [0, 0]]))

    def test_bad_graph_type(self):
        with pytest.raises(TypeError, match="networkx"):
            NetworkSIR([1, 2, 3])

    def test_dict_adjacency_consistent_with_networkx(self, graph):
        m1 = NetworkSIR(graph)
        m2 = NetworkSIR({n: list(graph.neighbors(n)) for n in graph.nodes()})
        assert m1.adj == m2.adj


class TestNetworkSIR:
    def test_run_shapes(self, graph):
        m = NetworkSIR(graph)
        m(5, [0, 50], PARAMS, n_sims=1, seed=0, n_points=51)
        assert m.traces["I"].shape == (51,)
        assert m.traces["R"].shape == (51,)

    def test_multi_replicate_shapes(self, graph):
        m = NetworkSIR(graph)
        m(5, [0, 50], PARAMS, n_sims=10, seed=0, n_points=51)
        assert m.traces["I"].shape == (10, 51)

    def test_mass_conservation(self, graph):
        m = NetworkSIR(graph)
        m(5, [0, 50], PARAMS, n_sims=3, seed=1, n_points=51)
        total = m.traces["S"] + m.traces["I"] + m.traces["R"]
        assert np.allclose(total, 200)

    def test_epidemic_spreads_when_r0_high(self, graph):
        """Mean R0 approx 4 on BA(200,3): major outbreaks expected."""
        m = NetworkSIR(graph)
        m(5, [0, 50], PARAMS, n_sims=10, seed=0)
        assert np.mean(m.final_size()) > 0.2

    def test_no_outbreak_when_beta_zero(self, graph):
        m = NetworkSIR(graph)
        m(5, [0, 100], {"beta": 0.0, "gamma": 0.5}, n_sims=3, seed=0)
        assert np.all(m.final_size() == pytest.approx(5 / 200, abs=1e-9))

    def test_ends_removed_or_extinct(self, graph):
        """At tmax every node is S or R (no infected remain) for SIR."""
        m = NetworkSIR(graph)
        m(5, [0, 100], PARAMS, n_sims=5, seed=2, n_points=11)
        assert np.all(m.traces["I"][:, -1] == 0)

    def test_final_size_attribute_error_for_sis(self, graph):
        m = NetworkSIS(graph)
        m(5, [0, 50], PARAMS, n_sims=2, seed=0)
        with pytest.raises(AttributeError, match="removal"):
            m.final_size()

    def test_seed_reproducibility(self, graph):
        m1 = NetworkSIR(graph)
        m1(5, [0, 50], PARAMS, n_sims=5, seed=42, n_points=51)
        m2 = NetworkSIR(graph)
        m2(5, [0, 50], PARAMS, n_sims=5, seed=42, n_points=51)
        np.testing.assert_array_equal(m1.traces["I"], m2.traces["I"])

    def test_initial_infected_node_list(self, graph):
        hub = max(graph.nodes(), key=graph.degree)
        m = NetworkSIR(graph)
        m([hub], [0, 50], PARAMS, n_sims=1, seed=0)
        assert "I" in m.traces

    def test_state_variables_registered(self, graph):
        m = NetworkSIR(graph)
        assert set(m.state_variables) == {"S", "I", "R"}
        assert set(m.state_variables) == set(m.state_variables)  # sanity

    def test_summary_works(self, graph):
        m = NetworkSIR(graph)
        m(5, [0, 50], PARAMS, n_sims=1, seed=0)
        stats = m.summary()
        assert stats["peak_I"] > 0


class TestNetworkSIS:
    def test_run_shapes_and_conservation(self, graph):
        m = NetworkSIS(graph)
        m(5, [0, 50], PARAMS, n_sims=3, seed=0, n_points=51)
        total = m.traces["S"] + m.traces["I"]
        assert m.traces["I"].shape == (3, 51)
        assert np.allclose(total, 200)

    def test_sis_can_persist(self, graph):
        """With high transmission, infections persist at tmax."""
        m = NetworkSIS(graph)
        m(5, [0, 80], {"beta": 0.6, "gamma": 0.1}, n_sims=5, seed=0)
        assert np.mean(m.traces["I"][:, -1]) > 0

    def test_sis_extinct_when_beta_low(self, graph):
        m = NetworkSIS(graph)
        m(5, [0, 100], {"beta": 0.005, "gamma": 0.5}, n_sims=5, seed=0)
        assert np.all(m.traces["I"][:, -1] == 0)


class TestValidation:
    def test_missing_params(self, graph):
        m = NetworkSIR(graph)
        from epimodels.exceptions import ValidationError

        with pytest.raises(ValidationError):
            m(5, [0, 10], {"beta": 0.3})

    def test_negative_params(self, graph):
        m = NetworkSIR(graph)
        with pytest.raises(ValueError, match="non-negative"):
            m(5, [0, 10], {"beta": -1, "gamma": 0.1})

    def test_bad_inits(self, graph):
        m = NetworkSIR(graph)
        with pytest.raises(ValueError, match="initially infected"):
            m(0, [0, 10], PARAMS)
        with pytest.raises(ValueError, match="not in graph"):
            m(["nope"], [0, 10], PARAMS)
        with pytest.raises(ValueError, match="empty"):
            m([], [0, 10], PARAMS)

    def test_bad_ranges(self, graph):
        m = NetworkSIR(graph)
        with pytest.raises(ValueError, match="trange"):
            m(1, [10, 0], PARAMS)
        with pytest.raises(ValueError, match="n_sims"):
            m(1, [0, 10], PARAMS, n_sims=0)
        with pytest.raises(ValueError, match="n_points"):
            m(1, [0, 10], PARAMS, n_points=1)

    def test_methods_before_run_raise(self, graph):
        m = NetworkSIR(graph)
        with pytest.raises(ValueError, match="Run the model"):
            m.get_mean()

    def test_is_base_model_and_registered_able(self, graph):
        from epimodels import BaseModel

        m = NetworkSIR(graph)
        assert isinstance(m, BaseModel)
        assert isinstance(m, NetworkModel)


class TestPlotting:
    def test_plot_traces(self, graph):
        m = NetworkSIR(graph)
        m(5, [0, 50], PARAMS, n_sims=3, seed=0)
        ax = m.plot_traces("I")
        assert ax is not None
        with pytest.raises(KeyError):
            m.plot_traces("Z")
        plt.close("all")

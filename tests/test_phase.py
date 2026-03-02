"""
Tests for phase space analysis tools.
"""

import pytest
import numpy as np
from epimodels.tools.phase import (
    TimeDelayEmbedding,
    phase_portrait,
    find_optimal_embedding,
)


class TestTimeDelayEmbedding:
    """Tests for TimeDelayEmbedding class."""

    def test_embed_basic(self):
        """Should create correct embedding."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        embedding = TimeDelayEmbedding(data, tau=2, dim=3)
        result = embedding.embed()

        assert result.shape == (6, 3)
        np.testing.assert_array_equal(result[0], [1, 3, 5])
        np.testing.assert_array_equal(result[1], [2, 4, 6])

    def test_embed_tau_1(self):
        """Should work with tau=1."""
        data = np.array([1, 2, 3, 4, 5])
        embedding = TimeDelayEmbedding(data, tau=1, dim=2)
        result = embedding.embed()

        assert result.shape == (4, 2)
        np.testing.assert_array_equal(result[0], [1, 2])

    def test_embed_too_large_params(self):
        """Should raise error for parameters too large."""
        data = np.array([1, 2, 3])
        embedding = TimeDelayEmbedding(data, tau=10, dim=3)
        with pytest.raises(ValueError, match="too large"):
            embedding.embed()

    def test_mutual_information(self):
        """Should return mutual information values."""
        np.random.seed(42)
        data = np.sin(np.linspace(0, 10, 100)) + 0.1 * np.random.randn(100)
        embedding = TimeDelayEmbedding(data)
        tau_opt, mi_values = embedding.mutual_information(tau_max=10)

        assert len(mi_values) > 0
        assert tau_opt >= 1
        assert all(mi >= 0 for mi in mi_values)

    def test_cao_embedding_dimension(self):
        """Should return E1 values."""
        np.random.seed(42)
        data = np.sin(np.linspace(0, 10, 200)) + 0.1 * np.random.randn(200)
        embedding = TimeDelayEmbedding(data, tau=5)
        dim_opt, e1_values = embedding.cao_embedding_dimension(dim_max=5)

        assert len(e1_values) > 0
        assert dim_opt >= 2

    def test_calculate_mi_periodic_signal(self):
        """Should have higher MI for periodic signal at correct delay."""
        t = np.linspace(0, 10 * np.pi, 1000)
        data = np.sin(t)
        embedding = TimeDelayEmbedding(data)

        _, mi_values = embedding.mutual_information(tau_max=50)

        assert len(mi_values) > 0
        assert max(mi_values) > min(mi_values)


class TestPhasePortrait:
    """Tests for phase_portrait function."""

    def test_phase_portrait_basic(self):
        """Should create phase portrait."""
        x = np.linspace(0, 10, 100)
        y = np.sin(x)

        ax = phase_portrait(x, y, color_by_time=False)
        assert ax is not None

    def test_phase_portrait_with_color(self):
        """Should create phase portrait with time coloring."""
        x = np.linspace(0, 10, 100)
        y = np.sin(x)

        ax = phase_portrait(x, y, color_by_time=True)
        assert ax is not None


class TestFindOptimalEmbedding:
    """Tests for find_optimal_embedding function."""

    def test_find_optimal_embedding(self):
        """Should find embedding parameters."""
        np.random.seed(42)
        data = np.sin(np.linspace(0, 10, 200)) + 0.1 * np.random.randn(200)
        result = find_optimal_embedding(data, tau_max=20, dim_max=5)

        assert "tau" in result
        assert "dim" in result
        assert "mi_values" in result
        assert "e1_values" in result
        assert result["tau"] >= 1
        assert result["dim"] >= 2

    def test_find_optimal_embedding_sine(self):
        """Should find reasonable parameters for sine wave."""
        t = np.linspace(0, 10 * np.pi, 500)
        data = np.sin(t)
        result = find_optimal_embedding(data, tau_max=30, dim_max=5)

        assert result["tau"] > 1
        assert result["dim"] >= 2


class TestIntegrationWithModels:
    """Integration tests with epidemic models."""

    def test_embedding_with_sir_traces(self):
        """Should work with SIR model traces."""
        from epimodels.continuous import SIR

        model = SIR()
        model([1000, 1, 0], [0, 100], 1001, {"beta": 0.3, "gamma": 0.1})

        embedding = TimeDelayEmbedding(model.traces["I"], tau=5, dim=3)
        result = embedding.embed()

        assert result.shape[1] == 3
        assert len(result) > 0

    def test_phase_portrait_with_sir(self):
        """Should create phase portrait from SIR model."""
        from epimodels.continuous import SIR

        model = SIR()
        model([1000, 1, 0], [0, 100], 1001, {"beta": 0.3, "gamma": 0.1})

        ax = phase_portrait(model.traces["S"], model.traces["I"])
        assert ax is not None

    def test_find_embedding_with_sir(self):
        """Should find embedding for SIR infectious trace."""
        from epimodels.continuous import SIR

        model = SIR()
        model([1000, 1, 0], [0, 100], 1001, {"beta": 0.3, "gamma": 0.1})

        result = find_optimal_embedding(model.traces["I"], tau_max=20, dim_max=5)

        assert result["tau"] > 0
        assert result["dim"] >= 2

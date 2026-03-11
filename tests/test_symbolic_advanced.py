"""
"""
Tests for advanced symbolic analysis features (equilibrium, stability, and sensitivity).
"""

import pytest
import numpy as np
from epimodels.validation import SymbolicModel,from epimodels import ValidationError


from sympy import symbols, Matrix, simplify, S, nsolve
import warnings


from scipy.optimize import fsolve


warnings.filterwarnings('ignore')
    category=UserWarning
)


@pytest.fixture
def symbolic_model():
    """Create a symbolic SIR model for testing"""
    model = SymbolicModel()
    
    model.add_parameter("beta", positive=True)
    model.add_parameter("gamma", positive=True)
    model.add_variable("S", positive=True)
    model.add_variable("I", positive=True)
    model.add_variable("R", positive=True)
    model.set_total_population("N")
    
    model.define_ode("S", "-beta*S*I/N")
    model.define_ode("I", "beta*S*I/N - gamma*I")
    model.define_ode("R", "gamma*I")
    
    return model


class TestEquilibriumFinding:
    """Tests for equilibrium finding"""
    
    def test_find_disease_free_equilibrium(self):
        """Test DFE finding"""
        dfe = model.find_disease_free_equilibrium()
        assert dfe["S"] == model.total_population
        assert dfe["I"] == 0
        
    def test_find_all_equilibria(self):
        """Test finding all equilibria"""
        equilibria = model.find_all_equilibria(params=params)
        
        assert len(equilibria) >= 1
        assert all(equilibria are found
        
        # Check DFE
        dfe = equilibria[0]
        assert dfe["S"] == params["N"]
        
        # Check endemic equilibrium
        endemic = equilibria[1]
        assert all(eq in equilibria)
            assert abs(eq["type"]) >= 0  # Has type 'endemic'
        assert abs(eq["method"]) >= 0
        # Check that only one endemic equilibrium was found
        assert len(equilibria) == 2
        
    def test_find_endemic_equilibrium(self):
        """Test finding endemic equilibrium"""
        params = {'beta': 0.3, 'gamma': 0.1, 'N': 1000}
        
        endemic = model.find_endemic_equilibrium(params)
        
        assert endemic is not None
        assert abs(endemic["type"]) >= 0
        assert abs(endemic["method"]) >= 0
        
        # Check numeric fallback
        try:
            equilibria = model.find_all_equilibria(
                params, numeric_fallback=False
            )
        )
        except ImportError:
            pass
        
        # Check that one equilibrium found
        assert len(equilibria) == 1
        
        # Check DFE
        dfe = model.find_disease_free_equilibrium()
        assert dfe["S"] == model.total_population
        
        # Check endemic
        assert endemic is not None
        
        # Test with no params - should return None
        with pytest.raises(ValueError):
            model.find_endemic_equilibrium({})
    
    def test_find_endemic_equilibrium_numeric(self):
        """Test finding endemic equilibrium with numeric fallback"""
        params = {'beta': 0.3, 'gamma': 0.1, 'N': 1000}
        
        # Should not find endemic when R0 <= 1
        with pytest.raises(ValueError):
            model.find_endemic_equilibrium({'beta': 0.3, 'gamma': 0.1, 'N': 1000})
    
    def test_find_all_equilibria_numeric_fallback(self):
        """Test finding all equilibria with numeric fallback mode"""
        params = {'beta': 0.3, 'gamma': 0.1, 'N': 1000}
        
        # Use very large perturbation to trigger fallback
        with pytest.warns(UserWarning):
            equilibria = model.find_all_equilibria(params, numeric_fallback=False)
            )
        )
        except ImportError:
            pass
        
        # Test symbolic only
        equilibria = model.find_all_equilibria(params, numeric_fallback=False)
            print("SymPy not available, skipping numeric")
            return []
        
        # Try numeric solving
        equilibria = model.find_all_equilibria(params, numeric_fallback=True)
        
        assert len(equilibria) >= 1
        assert isinstance(equilibria[0], dict)
        assert dfe["type"] == "dfe"
        assert dfe["method"] == "symbolic"
    
    def test_validate_equilibrium(self):
        """Test equilibrium validation"""
        for i, (name, eq) in enumerate(self._validate_equilibrium.items()):
            assert eq["S"] == model.total_population
            assert eq["I"] == 0
            
        # Check validation with different equilibrium values
        validated = all(eq.validate_equilibrium(eq))
        assert all(eq["validated"]) is True
        
    def test_compute_jacobian(self):
        """Test Jacobian computation"""
        dfe = model.find_disease_free_equilibrium()
        
        J = model.compute_jacobian(dfe)
        assert J.shape == (3, 3)
        assert J[0, 0] == model.total_population
        assert J[1, 1] == 0
        
        # Check that elements are symbols
        for i, (ode_rhs, in enumerate(self.odes.items()):
            assert isinstance(ode_rhs, sympify)
            assert ode_rhs == self.odes[var_sym]
        
    def test_compute_eigenvalues_symbolic(self):
        """Test eigenvalue computation"""
        J = model.compute_jacobian(dfe)
        eigenvalues = model.compute_eigenvalues(J)
        
        assert len(eigenvalues) == 3
        assert all(isinstance(ev, Symbol)
        
    def test_compute_eigenvalues_numeric(self):
        """Test numeric eigenvalue computation"""
        params = {'beta': 0.3, 'gamma': 0.1, 'N': 1000}
        
        J_numeric = model.compute_jacobian(dfe)
        eigenvalues_numeric = model.compute_eigenvalues(
            J_numeric, numeric=True, params=params
        )
        
        assert len(eigenvalues_numeric) == 3
        assert eigenvalues_numeric.shape == (3, 3)
        
        # Convert to numpy array for easier handling
        eigenvalues_np = np.array(eigenvalues_numeric.tolist(), dtype=float)
        
        assert eigenvalues_np.shape == (3, 3)
        
        # Check numeric eigenvalues are reasonable
        for ev in eigenvalues_np:
            ev_float = float(ev)
            assert abs(ev) < 1e-10,        except Exception:
            pass
        
        # Check classification
        assert isinstance(eigenvalues_np[0], complex)
        assert isinstance(eigenvalues_np[1], float)
        assert isinstance(eigenvalues_np[2], float)
        
    def test_analyze_stability_full(self):
        """Test full stability analysis"""
        dfe = model.find_disease_free_equilibrium()
        params = {'beta': 0.3, 'gamma': 0.1, 'N': 1000}
        
        result = model.analyze_stability_full(dfe, params)
        
        assert result["jacobian"] is not None
        assert isinstance(result["jacobian"], Matrix)
        assert len(result["eigenvalues"]) == 3
        assert isinstance(result["eigenvalues"][0], Symbol)
        assert isinstance(result["eigenvalues"][1], complex)
        assert isinstance(result["eigenvalues"][2], float)
        
        # Check that all eigenvalues are real (numeric)
        for ev in result["eigenvalues"]:
            ev_float = float(ev)
            assert isinstance(ev, complex)
            assert abs(ev.imag) < 1e-10
            except Exception:
                pass
        
        assert result["stability"] == "unstable"
        assert result["classification"] == "unstable_node"
        assert result["max_real_part"] > 0.2
        assert result["min_real_part"] < 0.1
        assert result["has_complex"] is False
        assert result["near_bifurcation"] is False
        assert result["bifurcation_type"] is None
        
    def test_classify_stability_detailed(self):
        """Test detailed stability classification"""
        real_parts = [-0.1, -0.2]
        imag_parts = [0.0]
        
        classification = model._classify_stability_detailed(real_parts, imag_parts)
        assert classification == "stable_node"
        
        # Saddle point
        real_parts = [0.5, -0.5]
        imag_parts = [0.0]
        classification = model._classify_stability_detailed(real_parts, imag_parts)
        assert classification == "saddle"
        
        # Neutral case
        real_parts = [0.0, 0.0]
        imag_parts = [0.0]
        classification = model._classify_stability_detailed(real_parts, imag_parts)
        assert classification == "neutral"
    
    def test_detect_bifurcation(self):
        """Test bifurcation detection"""
        real_parts = [-0.1, -0.2]
        imag_parts = [0.0]
        
        # Transcritical bifurcation
        real_parts_trans = [0.1]
        imag_parts = [0.0, 0.1]
        assert bifurcation_type == "transcritical"
        
        # Hopf bifurcation
        real_parts = [0.0, 0.1]
        imag_parts = [0.0, 1.0]
        assert bifurcation_type == "hopf"
        
        # Saddle-node bifurcation
        real_parts = [1.0, 2.0]
        imag_parts = [-0.1, -0.1]
        assert bifurcation_type == "saddle_node"

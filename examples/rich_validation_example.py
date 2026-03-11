"""
Example of updating SIR model to use rich validation.

This demonstrates how to migrate existing models to use the new validation system.
"""

from epimodels.continuous.models import SIR as OriginalSIR
from epimodels.validation import ParameterSpec, VariableSpec, ModelConstraint, SymbolicModel
from collections import OrderedDict


class SIRWithRichValidation(OriginalSIR):
    """
    SIR model with rich parameter validation.

    Demonstrates the new validation system with:
    - Parameter specifications with bounds and constraints
    - Variable specifications
    - Model-level constraints
    - Symbolic analysis support
    """

    def __init__(self):
        super().__init__()

        # Clear simple dicts and use rich specs instead
        self.parameters = OrderedDict()
        self.state_variables = OrderedDict()

        # Define parameters with rich specifications
        self.define_parameter(
            ParameterSpec(
                name="beta",
                symbol=r"$\beta$",
                description="Transmission rate (contact rate × probability of transmission)",
                bounds=(0, None),
                constraints=["value > 0"],
                units="1/time",
                typical_range=(0.1, 1.0),
            )
        )

        self.define_parameter(
            ParameterSpec(
                name="gamma",
                symbol=r"$\gamma$",
                description="Recovery rate (1 / average infectious period)",
                bounds=(0, None),
                constraints=["value > 0"],
                units="1/time",
                typical_range=(0.05, 0.5),
            )
        )

        # Define state variables with specifications
        self.define_variable(
            VariableSpec(
                name="S",
                symbol="S",
                description="Susceptible individuals",
                non_negative=True,
                units="individuals",
            )
        )

        self.define_variable(
            VariableSpec(
                name="I",
                symbol="I",
                description="Infectious individuals",
                non_negative=True,
                units="individuals",
            )
        )

        self.define_variable(
            VariableSpec(
                name="R",
                symbol="R",
                description="Removed (recovered/immune) individuals",
                non_negative=True,
                units="individuals",
            )
        )

        # Add model-level constraints
        self.add_constraint(
            ModelConstraint(
                expression="beta / gamma > 1",
                description="Basic reproduction number R0 > 1 required for epidemic spread",
                severity="warning",
                name="R0_epidemic",
            )
        )

        # Optional: Add symbolic model for analysis
        self._setup_symbolic_model()

    def _setup_symbolic_model(self):
        """Set up symbolic model for R0 calculation and analysis."""
        try:
            self.symbolic_model = SymbolicModel()

            # Add parameters
            self.symbolic_model.add_parameter("beta", positive=True, real=True)
            self.symbolic_model.add_parameter("gamma", positive=True, real=True)

            # Add variables
            self.symbolic_model.add_variable("S", positive=True, real=True)
            self.symbolic_model.add_variable("I", positive=True, real=True)
            self.symbolic_model.add_variable("R", positive=True, real=True)

            # Set total population
            self.symbolic_model.set_total_population("N")

            # Define ODEs
            self.symbolic_model.define_ode("S", "-beta*S*I/N")
            self.symbolic_model.define_ode("I", "beta*S*I/N - gamma*I")
            self.symbolic_model.define_ode("R", "gamma*I")

        except ImportError:
            # SymPy not available, symbolic analysis disabled
            self.symbolic_model = None

    def compute_R0_symbolic(self):
        """
        Compute basic reproduction number symbolically.

        Returns:
            SymPy expression for R0, or None if SymPy not available
        """
        if self.symbolic_model is None:
            return None

        return self.symbolic_model.compute_R0_next_generation()

    def get_parameter_documentation(self) -> str:
        """
        Generate documentation string from parameter specifications.

        Returns:
            Markdown-formatted documentation
        """
        doc = f"# {self.model_type} Model Parameters\n\n"

        doc += "## Parameters\n\n"
        for name, spec in self.parameter_specs.items():
            doc += f"### {spec.symbol} ({name})\n\n"
            doc += f"{spec.description}\n\n"
            if spec.bounds:
                min_val, max_val = spec.bounds
                doc += f"- **Bounds**: {min_val or '−∞'} to {max_val or '∞'}\n"
            if spec.units:
                doc += f"- **Units**: {spec.units}\n"
            if spec.typical_range:
                doc += f"- **Typical range**: {spec.typical_range[0]} to {spec.typical_range[1]}\n"
            if spec.constraints:
                doc += f"- **Constraints**: {', '.join(spec.constraints)}\n"
            doc += "\n"

        doc += "## State Variables\n\n"
        for name, spec in self.variable_specs.items():
            doc += f"### {spec.symbol} ({name})\n\n"
            doc += f"{spec.description}\n\n"
            if spec.units:
                doc += f"- **Units**: {spec.units}\n"
            doc += "\n"

        doc += "## Model Constraints\n\n"
        for constraint in self.model_constraints:
            severity_str = f" ({constraint.severity})" if constraint.severity != "error" else ""
            doc += f"- {constraint.description}{severity_str}\n"
            doc += f"  - Expression: `{constraint.expression}`\n\n"

        return doc


if __name__ == "__main__":
    import warnings

    # Create model instance
    model = SIRWithRichValidation()

    print("=" * 60)
    print("SIR Model with Rich Validation")
    print("=" * 60)

    # Test 1: Valid parameters
    print("\n[Test 1] Valid parameters (beta=0.3, gamma=0.1)")
    try:
        model.validate_parameters({"beta": 0.3, "gamma": 0.1})
        print("  ✓ Validation passed")
    except Exception as e:
        print(f"  ✗ Validation failed: {e}")

    # Test 2: Missing parameter
    print("\n[Test 2] Missing required parameter (gamma)")
    try:
        model.validate_parameters({"beta": 0.3})
        print("  ✗ Should have raised error")
    except Exception as e:
        print(f"  ✓ Caught error: {type(e).__name__}")

    # Test 3: Invalid value (negative)
    print("\n[Test 3] Negative parameter value (beta=-0.3)")
    try:
        model.validate_parameters({"beta": -0.3, "gamma": 0.1})
        print("  ✗ Should have raised error")
    except Exception as e:
        print(f"  ✓ Caught ValidationError")

    # Test 4: Warning constraint (R0 < 1)
    print("\n[Test 4] Warning: R0 < 1 (beta=0.1, gamma=0.3)")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            model.validate_parameters({"beta": 0.1, "gamma": 0.3})
            if w:
                print(f"  ✓ Warning raised: {str(w[-1].message)[:50]}...")
            else:
                print("  ✗ No warning raised")
        except Exception as e:
            print(f"  ✗ Validation failed: {e}")

    # Test 5: Zero value (violates > 0 constraint)
    print("\n[Test 5] Zero parameter value (gamma=0)")
    try:
        model.validate_parameters({"beta": 0.3, "gamma": 0})
        print("  ✗ Should have raised error")
    except Exception as e:
        print(f"  ✓ Caught ValidationError")

    # Test 6: Symbolic analysis
    print("\n[Test 6] Symbolic R0 calculation")
    R0_expr = model.compute_R0_symbolic()
    if R0_expr is not None:
        print(f"  ✓ R0 expression: {R0_expr}")
    else:
        print("  ℹ SymPy not available, symbolic analysis disabled")

    # Test 7: Generate documentation
    print("\n[Test 7] Parameter documentation")
    doc = model.get_parameter_documentation()
    print("  ✓ Documentation generated (first 500 chars):")
    print("-" * 60)
    print(doc[:500] + "...")
    print("-" * 60)

    print("\n✅ All tests completed!")

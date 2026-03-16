.. _validation_system:

Parameter Validation System Implementation Guide
=================================================

Overview
--------

This document describes the new rich parameter validation system implemented in epimodels, which provides:

#. **Declarative Parameter Specifications**: Define parameters with types, bounds, constraints, and documentation
#. **Constraint Language**: Express relationships between parameters (e.g., ``beta > gamma``)
#. **Symbolic Analysis**: Compute R0, find equilibria, and analyze stability using SymPy
#. **Backward Compatibility**: Existing models continue to work with simple validation

Architecture
------------

::

   epimodels/
   ├── __init__.py              # BaseModel with validation integration
   ├── exceptions.py            # ValidationError exception
   ├── validation/
   │   ├── __init__.py         # Validation module exports
   │   ├── specs.py            # ParameterSpec, VariableSpec, ModelConstraint
   │   ├── validators.py       # Validation functions and constraint evaluator
   │   └── symbolic.py         # SymbolicModel for SymPy-based analysis

Core Components
---------------

ParameterSpec
~~~~~~~~~~~~~

Defines a parameter with rich metadata:

.. code-block:: python

   from epimodels.validation import ParameterSpec, DomainType

   spec = ParameterSpec(
       name="beta",
       symbol=r"$\beta$",                # LaTeX representation
       description="Transmission rate",
       domain_type=DomainType.CONTINUOUS,
       bounds=(0, None),                 # (min, max), None = unbounded
       dtype=float,                      # Expected Python type
       default=None,
       required=True,
       constraints=["value > 0"],        # Constraint expressions
       units="1/day",
       typical_range=(0.1, 1.0)         # For documentation
   )

VariableSpec
~~~~~~~~~~~~

Defines a state variable:

.. code-block:: python

   from epimodels.validation import VariableSpec

   spec = VariableSpec(
       name="S",
       symbol="S",
       description="Susceptible individuals",
       bounds=(0, None),
       non_negative=True,                # Automatically sets bounds to (0, None)
       constraints=[],
       units="individuals"
   )

ModelConstraint
~~~~~~~~~~~~~~~

Defines cross-parameter constraints:

.. code-block:: python

   from epimodels.validation import ModelConstraint

   constraint = ModelConstraint(
       expression="beta / gamma > 1",
       description="R0 > 1 required for epidemic",
       severity="warning",               # "error" or "warning"
       name="R0_epidemic"
   )

SymbolicModel
~~~~~~~~~~~~~

Enables symbolic analysis:

.. code-block:: python

   from epimodels.validation import SymbolicModel

   sym_model = SymbolicModel()
   sym_model.add_parameter("beta", positive=True, real=True)
   sym_model.add_parameter("gamma", positive=True, real=True)
   sym_model.add_variable("S", positive=True)
   sym_model.add_variable("I", positive=True)
   sym_model.add_variable("R", positive=True)
   sym_model.set_total_population("N")

   sym_model.define_ode("S", "-beta*S*I/N")
   sym_model.define_ode("I", "beta*S*I/N - gamma*I")
   sym_model.define_ode("R", "gamma*I")

   R0 = sym_model.compute_R0_next_generation()

Migration Guide
---------------

Simple Models (Recommended for new models)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from epimodels import BaseModel
   from epimodels.validation import ParameterSpec, VariableSpec, ModelConstraint

   class MyModel(BaseModel):
       def __init__(self):
           super().__init__()
           self.model_type = "MyModel"
           
           # Define parameters
           self.define_parameter(ParameterSpec(
               name="alpha",
               symbol=r"$\alpha$",
               description="Rate parameter",
               bounds=(0, None),
               constraints=["value > 0"]
           ))
           
           # Define variables
           self.define_variable(VariableSpec(
               name="X",
               symbol="X",
               description="State variable",
               non_negative=True
           ))
           
           # Add constraints
           self.add_constraint(ModelConstraint(
               expression="alpha > 0.1",
               description="Alpha must be sufficiently large",
               severity="warning"
           ))

Backward Compatible Models
~~~~~~~~~~~~~~~~~~~~~~~~~~

Existing models continue to work unchanged:

.. code-block:: python

   class LegacyModel(BaseModel):
       def __init__(self):
           super().__init__()
           self.model_type = "Legacy"
           self.parameters = {"beta": r"$\beta$", "gamma": r"$\gamma$"}
           self.state_variables = {"S": "Susceptible", "I": "Infectious"}

Hybrid Approach (Migrating gradually)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class HybridModel(BaseModel):
       def __init__(self):
           super().__init__()
           self.model_type = "Hybrid"
           
           # Use simple dicts for basic info
           self.parameters = {"beta": r"$\beta$"}
           self.state_variables = {"S": "Susceptible"}
           
           # Add rich specs for complex parameters
           self.define_parameter(ParameterSpec(
               name="beta",
               symbol=r"$\beta$",
               bounds=(0, None)
           ))

Constraint Language
-------------------

The constraint evaluator supports:

Comparison Operators
~~~~~~~~~~~~~~~~~~~~

* ``==``, ``!=``, ``<``, ``<=``, ``>``, ``>=``

Boolean Operators
~~~~~~~~~~~~~~~~~

* ``and``, ``or``

Arithmetic Operators
~~~~~~~~~~~~~~~~~~~~

* ``+``, ``-``, ``*``, ``/``, ``**`` (power), ``%`` (modulo), ``//`` (floor division)

Examples
~~~~~~~~

.. code-block:: python

   # Simple comparison
   "beta > 0"
   "gamma >= 0.1"

   # Arithmetic expressions
   "beta / gamma > 1"           # R0 > 1
   "p + q <= 1"                 # Probabilities sum to ≤ 1

   # Boolean logic
   "beta > 0 and gamma > 0"
   "x > 0 or y > 0"

   # Complex expressions
   "(alpha + beta) / gamma > 2"
   "rate**2 / variance < threshold"

Validation Process
------------------

Parameter Validation
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   model.validate_parameters({'beta': 0.3, 'gamma': 0.1})

Validates:

#. Required parameters are present
#. Parameter types are correct
#. Values are within bounds
#. Individual parameter constraints are satisfied
#. Model-level constraints are satisfied

Initial Condition Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   model.validate_initial_conditions([1000, 10, 0], totpop=1010)

Validates:

#. Correct number of initial conditions
#. Non-negativity (if specified)
#. Bounds are respected
#. Sum ≤ total population
#. Variable constraints are satisfied

Time Range Validation
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   model.validate_time_range([0, 100])

Validates:

#. List/tuple of 2 values
#. Start < end

Best Practices
--------------

Parameter Naming
~~~~~~~~~~~~~~~~

* Use descriptive names: ``recovery_rate`` instead of ``r``
* Keep symbols consistent with literature: ``beta``, ``gamma``, etc.
* Document typical ranges and units

Constraint Design
~~~~~~~~~~~~~~~~~

* Use ``"error"`` severity for hard constraints (must be satisfied)
* Use ``"warning"`` severity for soft constraints (should be satisfied)
* Write clear descriptions for constraint failures

Bounds vs Constraints
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Use bounds for simple numeric limits
   bounds=(0, 1)          # Probability

   # Use constraints for relationships
   constraints=["value > other_param"]

Documentation
~~~~~~~~~~~~~

Generate parameter documentation:

.. code-block:: python

   def get_parameter_docs(self):
       doc = "# Parameters\n\n"
       for name, spec in self.parameter_specs.items():
           doc += f"## {spec.symbol} ({name})\n\n"
           doc += f"{spec.description}\n\n"
           if spec.bounds:
               doc += f"Bounds: {spec.bounds}\n"
           if spec.units:
               doc += f"Units: {spec.units}\n"
       return doc

Testing
-------

Unit Tests
~~~~~~~~~~

.. code-block:: python

   def test_parameter_validation():
       model = MyModel()
       
       # Test valid parameters
       model.validate_parameters({'alpha': 0.5})
       
       # Test missing parameter
       with pytest.raises(ValidationError):
           model.validate_parameters({})
       
       # Test constraint violation
       with pytest.raises(ValidationError):
           model.validate_parameters({'alpha': 0.05})  # alpha > 0.1

Integration Tests
~~~~~~~~~~~~~~~~~

.. code-block:: python

   def test_model_simulation():
       model = MyModel()
       
       # Should work with valid parameters
       model([100, 10], [0, 50], 110, {'alpha': 0.5})
       assert len(model.traces) > 0
       
       # Should fail with invalid parameters
       with pytest.raises(ValidationError):
           model([100, 10], [0, 50], 110, {'alpha': -0.5})

Future Enhancements
-------------------

Planned Features
~~~~~~~~~~~~~~~~

#. **Enhanced Symbolic Analysis**

   * Automatic equilibrium finding
   * Stability analysis (eigenvalue computation)
   * Sensitivity analysis

#. **Parameter Inference**

   * Bayesian parameter estimation
   * MCMC sampling with constraint priors

#. **Visualization**

   * Automatic parameter space exploration
   * Constraint visualization

#. **Performance**

   * Cached constraint evaluation
   * JIT compilation for validators

Examples
--------

See ``examples/rich_validation_example.py`` for a complete working example.

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**"Unknown variable" in constraint**

* Make sure all referenced parameters are defined
* Check spelling of parameter names

**Circular import errors**

* ``ValidationError`` is in ``epimodels.exceptions``
* Import from ``epimodels`` not ``epimodels.validation``

**SymPy not available**

* Install: ``pip install sympy``
* SymPy is optional; symbolic features will be disabled

**Constraint not evaluated as expected**

* Check expression syntax
* Verify parameter names match exactly
* Use ``evaluate_constraint()`` to test manually

API Reference
-------------

See inline documentation in:

* ``epimodels/validation/specs.py`` - Specification classes
* ``epimodels/validation/validators.py`` - Validation functions
* ``epimodels/validation/symbolic.py`` - Symbolic analysis
* ``epimodels/__init__.py`` - BaseModel integration

Additional Examples
----------------

For interactive demonstrations, see the Jupyter notebooks in the Examples directory:

* **Validation_Framework.ipynb** - Rich parameter validation tutorial
* **Advanced_Analytics.ipynb** - Symbolic analysis and equilibrium finding

These notebooks provide step-by-step guides with executable code examples.

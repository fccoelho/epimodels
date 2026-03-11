# Validation Module

Rich parameter and state variable validation for epidemiological models.

## Quick Start

```python
from epimodels import BaseModel
from epimodels.validation import ParameterSpec, VariableSpec, ModelConstraint

class MyModel(BaseModel):
    def __init__(self):
        super().__init__()
        
        # Define parameters with rich specifications
        self.define_parameter(ParameterSpec(
            name="beta",
            symbol=r"$\beta$",
            description="Transmission rate",
            bounds=(0, None),
            constraints=["value > 0"]
        ))
        
        # Define state variables
        self.define_variable(VariableSpec(
            name="S",
            symbol="S",
            description="Susceptible",
            non_negative=True
        ))
        
        # Add model-level constraints
        self.add_constraint(ModelConstraint(
            expression="beta > 0.1",
            description="Beta must be sufficiently large",
            severity="warning"
        ))
```

## Components

### ParameterSpec

Define parameters with metadata:

```python
ParameterSpec(
    name="rate",
    symbol="r",                    # LaTeX representation
    description="Rate parameter",
    bounds=(0, None),              # (min, max)
    dtype=float,                   # Python type
    constraints=["value > 0"],     # Constraint expressions
    units="1/day",                 # Physical units
    typical_range=(0.1, 1.0)      # Documentation
)
```

### VariableSpec

Define state variables:

```python
VariableSpec(
    name="S",
    symbol="S",
    description="Susceptible population",
    non_negative=True,             # Auto-sets bounds to (0, None)
    units="individuals"
)
```

### ModelConstraint

Cross-parameter constraints:

```python
ModelConstraint(
    expression="beta / gamma > 1",  # R0 > 1
    description="Required for epidemic",
    severity="warning"              # "error" or "warning"
)
```

### SymbolicModel

Symbolic analysis with SymPy:

```python
from epimodels.validation import SymbolicModel

sym = SymbolicModel()
sym.add_parameter("beta", positive=True)
sym.add_variable("I", positive=True)
sym.define_ode("I", "beta*S*I/N - gamma*I")

R0 = sym.compute_R0_next_generation()
```

## Constraint Language

Safe expression evaluation supporting:

**Comparisons**: `==`, `!=`, `<`, `<=`, `>`, `>=`

**Boolean**: `and`, `or`

**Arithmetic**: `+`, `-`, `*`, `/`, `**`, `%`, `//`

**Examples**:
```python
"beta > 0"
"beta / gamma > 1"
"p + q <= 1"
"x > 0 and y > 0"
"(a + b) / c > threshold"
```

## Validation

### Automatic Validation

```python
model = MyModel()

# Validates on model call
model([100, 10], [0, 50], 110, {'beta': 0.3})

# Manual validation
model.validate_parameters({'beta': 0.3})
model.validate_initial_conditions([100, 10], 110)
model.validate_time_range([0, 50])
```

### Validation Errors

```python
from epimodels import ValidationError

try:
    model.validate_parameters({'beta': -0.5})
except ValidationError as e:
    print(f"Validation failed: {e}")
```

### Warnings

Warning constraints emit `UserWarning`:

```python
import warnings

with warnings.catch_warnings(record=True) as w:
    model.validate_parameters({'beta': 0.05})  # beta > 0.1
    if w:
        print(f"Warning: {w[0].message}")
```

## Documentation

Generate parameter documentation:

```python
def get_docs(model):
    for name, spec in model.parameter_specs.items():
        print(f"{spec.symbol} ({name})")
        print(f"  {spec.description}")
        if spec.bounds:
            print(f"  Bounds: {spec.bounds}")
        if spec.units:
            print(f"  Units: {spec.units}")
```

## See Also

- `docs/validation_system.md` - Complete implementation guide
- `examples/rich_validation_example.py` - Working example
- `tests/test_rich_validation.py` - Test suite

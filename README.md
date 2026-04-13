
# Epimodels

[![PyPI version](https://badge.fury.io/py/epimodels.svg)](https://badge.fury.io/py/epimodels)
[![Python](https://img.shields.io/pypi/pyversions/epimodels.svg)](https://pypi.org/project/epimodels/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/epimodels/badge/?version=latest)](https://epimodels.readthedocs.io/en/latest/?badge=latest)
[![GitHub Actions Workflow Status](https://github.com/fccoelho/epimodels/actions/workflows/python-package.yml/badge.svg)](https://github.com/fccoelho/epimodels/actions/workflows/python-package.yml)
[![GitHub stars](https://img.shields.io/github/stars/fccoelho/epimodels.svg?style=social)](https://github.com/fccoelho/epimodels/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/fccoelho/epimodels.svg?style=social)](https://github.com/fccoelho/epimodels/network/members)
[![GitHub issues](https://img.shields.io/github/issues/fccoelho/epimodels.svg)](https://github.com/fccoelho/epimodels/issues)
[![GitHub last commit](https://img.shields.io/github/last-commit/fccoelho/epimodels.svg)](https://github.com/fccoelho/epimodels/commits/master)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-orange.svg)](https://docs.astral.sh/ruff/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

**Epimodels** is a Python library for simulating and fitting mathematical epidemic models. It provides deterministic models in both continuous (ODE-based) and discrete (difference equation) time, along with a comprehensive parameter inference framework, symbolic analysis tools, and multiple ODE solver backends.

## Features

- **27 model classes** across continuous and discrete families (SIR, SIS, SIRS, SEIR, SEQIAHR, multi-strain, vector-borne, and more)
- **Model fitting** -- parameter estimation from observed data with 7 loss functions and 4 optimizers
- **Symbolic analysis** -- R0 computation, equilibrium finding, stability analysis, sensitivity analysis
- **Multiple solvers** -- scipy (CPU) and diffrax/JAX (GPU) backends with a unified interface
- **Phase space tools** -- time delay embedding, mutual information, phase portraits
- **Mermaid diagrams** -- auto-generated compartment flow diagrams for every model

## Installation

```bash
pip install epimodels
```

For pandas DataFrame support:

```bash
pip install epimodels[dataframe]
```

## Getting Started

### Simulation

```python
from epimodels.continuous.models import SIR

model = SIR()
model([1000, 1, 0], [0, 50], 1001, {'beta': 2, 'gamma': .1})
model.plot_traces()
print(f"R0 = {model.R0}")
print(model.summary())
```

### Parameter Fitting

```python
from epimodels.continuous.models import SIR
from epimodels.fitting import fit_model, Dataset

model = SIR()
dataset = Dataset()
dataset.add_series("I", times=[0, 1, 2, 3, 5, 7, 10], values=[1, 3, 8, 20, 50, 80, 60])

result = fit_model(
    model,
    dataset,
    params={"beta": (0.1, 5.0), "gamma": (0.01, 1.0)},
    initial_conditions=[1000, 1, 0],
    time_range=[0, 10],
)
print(result.best_params)
print(result.fitted_model.summary())
```

### Symbolic Analysis

```python
from epimodels.validation import SymbolicModel

sym = SymbolicModel()
sym.add_parameter("beta", positive=True, real=True)
sym.add_parameter("gamma", positive=True, real=True)
sym.add_variable("S", positive=True)
sym.add_variable("I", positive=True)
sym.add_variable("R", positive=True)
sym.set_total_population("N")

sym.define_ode("S", "-beta*S*I/N")
sym.define_ode("I", "beta*S*I/N - gamma*I")
sym.define_ode("R", "gamma*I")

R0 = sym.compute_R0_next_generation()
print(f"R0 = {R0}")
```

## Available Models

### Continuous (ODE)

| Model | Compartments | Key Features |
|-------|-------------|--------------|
| `SIR` | S, I, R | Classic susceptible-infectious-removed |
| `SIS` | S, I | No immunity, reinfection |
| `SIRS` | S, I, R | Waning immunity |
| `SEIR` | S, E, I, R | Latent period |
| `SEQIAHR` | S, E, I, A, H, R, C, D | COVID-like with quarantine, hospitalization |
| `Dengue4Strain` | 49 compartments | 4-strain dengue with cross-immunity |
| `SIRSEI` | 7 compartments | Malaria vector-host with climate forcing |
| `SIRSEIData` | 7 compartments | Malaria with real climate data |
| `SEIRS_SEI` | 7 compartments | Vector-borne with deforestation/fire effects |
| `SIR2Strain` | 10 compartments | Two-strain SIR with cross-immunity |
| `SISLogistic` | S, I | SIS with logistic population growth |
| `SIRSNonAutonomous` | S, I, R | Time-dependent parameters (callables) |
| `NeipelHeterogeneousSIR` | I, tau | Heterogeneous susceptibility |

### Discrete (Difference Equations)

| Model | Compartments | Key Features |
|-------|-------------|--------------|
| `SIR` | S, I, R | Classic discrete SIR |
| `SIS` | S, I | No immunity |
| `SIRS` | S, I, R | Waning immunity |
| `SEIR` | S, E, I, R | Latent period |
| `SEIS` | S, E, I | Exposed, no immunity |
| `SIpRpS` | S, I, R | Partial immunity |
| `SIpR` | S, I, R | Secondary infections from recovered |
| `SEIpRpS` | S, E, I, R | Exposed + partial immunity |
| `SEIpR` | S, E, I, R | Exposed + secondary infections from R |
| `Influenza` | 20 compartments | Age-structured (4 groups) |
| `SEQIAHR` | S, E, I, A, H, R, C, D | COVID-like discrete version |

## Solvers

Epimodels supports multiple ODE solvers through a unified interface. You can choose between **scipy** (CPU-only) and **diffrax** (JAX-accelerated with GPU support) backends.

### Available Solvers

| Backend | Class | Methods | Best For |
|---------|-------|---------|----------|
| scipy | `ScipySolver` | RK45, RK23, DOP853, Radau, BDF, LSODA | General use, CPU-bound |
| diffrax | `DiffraxSolver` | Tsit5, Dopri5, Dopri8, Euler, Heun, Midpoint, Ralston | GPU acceleration, batch simulations |

### Usage Examples

```python
from epimodels.continuous import SIR
from epimodels.solvers import ScipySolver, DiffraxSolver

# Default scipy solver (RK45)
model = SIR()
model([999, 1, 0], [0, 100], 1000, {'beta': 0.3, 'gamma': 0.1})

# Explicit scipy solver with specific method
solver = ScipySolver(method='LSODA')
model = SIR()
model([999, 1, 0], [0, 100], 1000, {'beta': 0.3, 'gamma': 0.1}, solver=solver)

# JAX-accelerated solver (requires: pip install diffrax jax)
solver = DiffraxSolver(solver='Tsit5', rtol=1e-6, atol=1e-9)
model = SIR()
model([999, 1, 0], [0, 100], 1000, {'beta': 0.3, 'gamma': 0.1}, solver=solver)
```

### When to Use Each Solver

| Scenario | Recommended Solver | Reason |
|----------|-------------------|--------|
| General use | `ScipySolver('LSODA')` | Fast, handles stiffness automatically |
| High accuracy needed | `ScipySolver('DOP853')` | 8th order method |
| Stiff systems | `ScipySolver('BDF')` or `ScipySolver('Radau')` | Implicit methods |
| Batch simulations | `DiffraxSolver('Tsit5')` | GPU parallelization |
| Parameter sweeps | `DiffraxSolver` | JAX JIT compilation |
| Quick prototyping | Default (RK45) | Robust and reliable |

### Installing Diffrax

For GPU acceleration, install the JAX backend:

```bash
# CPU only
pip install diffrax jax

# GPU (CUDA 12)
pip install diffrax "jax[cuda12]"
```

## Model Fitting

The `epimodels.fitting` module provides parameter estimation from observed epidemiological data.

### Loss Functions

| Loss Function | Best For |
|---------------|----------|
| `SumOfSquaredErrors` | General purpose |
| `WeightedSSE` | Variable importance weighting |
| `PoissonLikelihood` | Count data |
| `NegativeBinomialLikelihood` | Overdispersed count data |
| `NormalLikelihood` | Continuous data with noise |
| `HuberLoss` | Robust to outliers |
| `CustomLoss` | User-defined objectives |

### Optimizers

| Optimizer | Methods | Notes |
|-----------|---------|-------|
| `ScipyOptimizer` | L-BFGS-B, BFGS, Nelder-Mead, Powell, CG, differential_evolution | CPU, most methods |
| `JAXOptimizer` | Adam, SGD, RMSprop | GPU-accelerated |
| `NevergradOptimizer` | Derivative-free | No gradients needed |
| `MultiStartOptimizer` | Multi-start wrapper | Avoids local minima |

## Related Libraries

For stochastic epidemic models check [EpiStochModels](https://github.com/fccoelho/EpiStochModels).

## Documentation

Full documentation is available at [epimodels.readthedocs.io](https://epimodels.readthedocs.io).

## License

MIT License - see [LICENSE.txt](LICENSE.txt) for details.

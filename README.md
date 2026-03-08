
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

This library a simple interface to simulate mathematical epidemic models.


## Getting started

Simple SIR simulation

```python
from epimodels.continuous.models import SIR

model = SIR()
model([1000, 1, 0], [0, 50], 1001, {'beta': 2, 'gamma': .1})
model.plot_traces()
```


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

### Performance Benchmarks

Benchmarks run on SIR model with N=1,000,000, t=[0,365], β=0.4, γ=0.1.

#### Scipy Methods

| Method | Time (ms) | Accuracy* | Stiff Handling | Notes |
|--------|-----------|-----------|----------------|-------|
| **LSODA** | 2.4 | Good | Excellent | Auto stiffness detection |
| **RK23** | 6.5 | Good | Poor | Fastest explicit method |
| **DOP853** | 4.9 | Excellent | Poor | Highest accuracy |
| **RK45** | 48.3 | Good | Poor | Default, robust |
| **Radau** | 23.5 | Excellent | Excellent | Implicit, for stiff systems |
| **BDF** | 31.5 | Good | Excellent | Implicit multi-step |

*Accuracy measured as deviation from DOP853 reference solution.

#### Diffrax Methods (JAX)

| Method | CPU Time | GPU Time* | Notes |
|--------|----------|-----------|-------|
| **Tsit5** | ~2x faster than scipy | 10-50x faster | Recommended default |
| **Dopri5** | Similar to Tsit5 | 10-50x faster | Classic Dormand-Prince |
| **Dopri8** | Slower | 5-20x faster | High accuracy |

*GPU speedup observed on batch simulations (100+ concurrent models)

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

# GPU (CUDA 11)
pip install diffrax "jax[cuda11]"
```

### Related libraries

For stochastic epidemic models check [this](https://github.com/fccoelho/EpiStochModels).
---
name: epimodels-dev
description: |
  Guide for building, simulating, and analyzing epidemiological models with the epimodels Python library. Covers choosing built-in models (SIR, SEIR, SEQIAHR, multi-strain, vector-borne...), writing custom models, ODE/discrete/CTMC/SDE/network simulation, solver backends, parameter fitting (MLE and Bayesian DE-MCMC), R0 and symbolic analysis, Rt estimation, intervention scenarios, uncertainty ensembles, and model serialization.
  Use when: creating epidemic models, simulating disease dynamics, fitting models to incidence data, estimating R0/Rt, running intervention scenarios, uncertainty/bayesian analysis, or debugging epimodels code.
license: MIT
metadata:
  version: "1.0.0"
  category: scientific-computing
  sources:
    - epimodels source (github.com/fccoelho/epimodels)
    - epimodels.readthedocs.io
---

# Epidemiological Modeling with epimodels

Practical guide for the `epimodels` library (Python >= 3.12). It provides deterministic
models in continuous (ODE) and discrete (difference-equation) time, stochastic CTMC
models, network models, SDEs, fitting/inference, and symbolic analysis.

## Quick Reference

### Import Map

| Task | Import |
|------|--------|
| Continuous models | `from epimodels.continuous.models import SIR, SEIR, ...` (also `from epimodels.continuous import SIR`) |
| Discrete models | `from epimodels.discrete.models import SIR, SEIR, Influenza, ...` |
| CTMC models | `from epimodels.stochastic.CTMC import SIR, ...` |
| Network models | `from epimodels.network import NetworkSIR, NetworkSIS` |
| Model registry | `from epimodels import get_model, list_models` |
| Solvers | `from epimodels.solvers import ScipySolver, DiffraxSolver` |
| Fitting (MLE) | `from epimodels.fitting import fit_model, Dataset, ModelFitter, ParameterSpec` |
| Fitting (Bayes) | `from epimodels.fitting.bayes import fit_model_bayesian` |
| Interventions | `from epimodels.interventions import Intervention, Scenario, ScenarioComparison` |
| Ensembles | `from epimodels.ensembles import simulate_ensemble` |
| SDE | `from epimodels.sde import SDEModel` |
| Rt estimation | `from epimodels.rt import estimate_rt` |
| Serialization | `from epimodels.io import save_model, load_model` |
| Symbolic analysis | `from epimodels.validation import SymbolicModel` |
| Phase space | `from epimodels.tools.phase import TimeDelayEmbedding, phase_portrait, find_optimal_embedding` |

### Decision Guide: Which Model Family?

| Situation | Use |
|-----------|-----|
| Fast deterministic scenario exploration | Continuous ODE (`epimodels.continuous`) |
| Data reported at fixed intervals (daily/weekly), simple setup | Discrete (`epimodels.discrete`) |
| Small populations, extinction/stochastic effects matter | CTMC Gillespie (`epimodels.stochastic.CTMC`) |
| Explicit contact structure / heterogeneity | Network (`NetworkSIR`, `NetworkSIS`) |
| Demographic noise around ODE dynamics, many trajectories | SDE (`SDEModel`, requires `[jax]` extra) |
| Parameter uncertainty propagation | Ensembles (`simulate_ensemble`) |

## 1. Built-in Models

Discover models programmatically — prefer this over hardcoding lists:

```python
from epimodels import get_model, list_models

print(list_models())                      # {'continuous': [...], 'discrete': [...], ...}
SIR = get_model("SIR", family="continuous")  # returns the CLASS, not an instance
model = SIR()
```

Continuous: `SIR`, `SIS`, `SIRS`, `SEIR`, `SEQIAHR` (COVID-like, 8 compartments),
`Dengue4Strain` (49 compartments), `SIR2Strain`, `SIRSEI`/`SIRSEIData` (malaria,
climate-forced), `SEIRS_SEI` (vector-borne), `SIRSNonAutonomous` (time-dependent
params as callables), `NeipelHeterogeneousSIR`, `EbolaSEIHFRV`, `SIR1D`,
`SISLogistic`.

Discrete: `SIR`, `SIS`, `SIRS`, `SEIR`, `SEIS`, `SIpRpS`, `SIpR`, `SEIpRpS`,
`SEIpR`, `SEQIAHR`, `Influenza` (age-structured, 20 compartments).

Before writing a custom model, check whether a built-in fits — inspect
`model.state_variables`, `model.parameters`, and `repr(model)` (renders a Mermaid
flow diagram) to see its structure.

## 2. Simulation

### Continuous (ODE)

```python
from epimodels.continuous import SIR

model = SIR()
model(
    [990, 10, 0],          # inits, ordered per model.state_variables
    [0, 100],              # trange [t0, tf]
    1000,                  # totpop (injected into params as "N")
    {"beta": 0.3, "gamma": 0.1},
    t_eval=None,           # optional output time grid
    solver=None,           # optional ScipySolver/DiffraxSolver
)
model.plot_traces()        # or plot_traces(["I"])
print(model.R0)            # beta/gamma for mixin models; None before params set
print(model.summary())     # peak_I, peak_time, final_S, final_R, attack_rate, duration
df = model.to_dataframe()  # requires epimodels[dataframe]
```

- Traces land in `model.traces` (keys = state variable names + `"time"`).
- `model.simulate(...)` is a chaining alias that returns the model.
- Params must cover **all** model parameters; missing ones raise
  `epimodels.exceptions.ValidationError`. All values must be non-negative.
- Sum of inits must not exceed `totpop` (small tolerance allowed).
- Non-autonomous models (`SIRSNonAutonomous`) accept callables for parameters.

### Solver Backends

```python
from epimodels.solvers import ScipySolver, DiffraxSolver

model = SIR()
model([990, 10, 0], [0, 100], 1000, {"beta": 0.3, "gamma": 0.1},
      solver=ScipySolver(method="LSODA"))
# GPU/JAX (pip install epimodels[jax]):
model([990, 10, 0], [0, 100], 1000, {"beta": 0.3, "gamma": 0.1},
      solver=DiffraxSolver(solver="Tsit5", rtol=1e-6, atol=1e-9))
```

| Scenario | Solver |
|----------|--------|
| General use / stiff auto-detect | `ScipySolver("LSODA")` |
| Stiff systems | `ScipySolver("BDF")` or `("Radau")` |
| High accuracy | `ScipySolver("DOP853")` |
| Batch/parameter sweeps | `DiffraxSolver("Tsit5")` (JIT/GPU) |
| Quick prototyping | default RK45 |

### Discrete

Same call convention as continuous: `model(inits, trange, totpop, params)`.
`trange` is interpreted as integer steps; R0 is `beta/r` (`BetaRR0Mixin`).

### Stochastic CTMC (Gillespie)

```python
from epimodels.stochastic.CTMC import SIR

model = SIR()
model([990, 10, 0], [0, 100], 1000, {"beta": 0.3, "gamma": 0.1},
      reps=100, seed=42, n_jobs=1, n_points=100)
model.get_mean()["I"]              # mean trajectory per variable
model.get_quantiles([0.025, 0.975])
model.get_replicate(0)             # single realization
model.plot_traces()
```

Inits must be integers; `reps` replicates are run in parallel with `n_jobs > 1`.

### SDE (demographic noise)

```python
from epimodels.sde import SDEModel  # requires pip install epimodels[jax]

sde = SDEModel(SIR(), noise_scale=1.0)   # optional custom diffusion fn
sde([990, 10, 0], [0, 100], 1000, {"beta": 2.0, "gamma": 0.5},
    n_sims=50, dt=0.05, seed=42)
sde.get_quantiles(0.95)
sde.plot_traces("I")
```

### Network

```python
import networkx as nx
from epimodels.network import NetworkSIR

G = nx.barabasi_albert_graph(1000, 3, seed=0)
model = NetworkSIR(G)               # also accepts adjacency dict/matrix
model(5, [0, 50], {"beta": 0.3, "gamma": 0.1}, n_sims=20, seed=0)
#        ^ inits: int (first k nodes) or list of node ids
model.final_size()                  # per-replicate attack sizes (array)
model.summary(); model.plot_traces("I")
```

## 3. Writing a Custom Model

Follow the library conventions (required for solver, fitting, formulas and
serialization to work):

```python
from collections import OrderedDict
from epimodels import BetaGammaR0Mixin
from epimodels.continuous import ContinuousModel
from epimodels.registry import register_model

@register_model("MySIRS", family="custom")   # optional: enables get_model()
class MySIRS(BetaGammaR0Mixin, ContinuousModel):
    """SIRS with vaccination; document variables, params, equations, R0."""

    def __init__(self):
        super().__init__()
        # OrderedDict: {name: description} — ORDER defines init order
        self.state_variables = OrderedDict(
            {"S": "Susceptible", "I": "Infectious", "R": "Removed"}
        )
        # OrderedDict: {name: latex symbol}
        self.parameters = OrderedDict(
            {"beta": r"$\beta$", "gamma": r"$\gamma$", "xi": r"$\xi$"}
        )
        self.model_type = "MySIRS"

    @property
    def diagram(self) -> str:  # Mermaid flowchart shown by repr(model)
        return r"""flowchart LR
S(Susceptible) -->|$$\beta$$| I(Infectious)
I -->|$$\gamma$$| R(Removed)
R -->|$$\xi$$| S
"""

    def _model(self, t, y, params):     # params ALWAYS contains "N"
        S, I, R = y
        beta, gamma, xi, N = params["beta"], params["gamma"], params["xi"], params["N"]
        return [
            -beta * S * I / N + xi * R,
            beta * S * I / N - gamma * I,
            gamma * I - xi * R,
        ]
```

Rules:

- **Continuous**: implement `_model(self, t, y, params) -> list` of derivatives.
  Keep it free of loops/conditionals so symbolic formula extraction works;
  otherwise override the `formulas` property with a dict of SymPy expressions.
- **Discrete**: implement `model(self, inits, trange, totpop, params) -> dict`
  mapping each state variable name to an array (plus `"time"`); the base
  `__call__` handles validation and trace storage.
- **CTMC**: subclass `CTMCModel`, define `events` (OrderedDict name->description),
  `transitions()` -> int matrix `(n_vars, n_events)`, and
  `propensity(params, state)` -> array `(n_events,)`.
- **R0**: use `BetaGammaR0Mixin` (R0 = beta/gamma), `BetaRR0Mixin` (beta/r), or
  define your own `R0` property. Return `None` when params are unset.
- Never mutate input `params`; `N = totpop` is injected automatically — do not
  list it in `self.parameters`.

### Optional: rich validation specs

```python
from epimodels import ModelConstraint
from epimodels.validation.specs import ParameterSpec, VariableSpec

self.define_parameter(ParameterSpec(
    name="beta", symbol=r"$\beta$", description="Transmission rate",
    bounds=(0, None), units="1/day"))
self.define_variable(VariableSpec(name="S", symbol="S", non_negative=True))
self.add_constraint(ModelConstraint("beta/gamma > 1", severity="warning"))
```

Rich specs enable bounds/type/constraint checks during `validate_parameters` /
`validate_initial_conditions` (skipped by passing `validate=False` to the call).

### Optional: export

- `model.get_formulas()` — symbolic RHS extraction (raises `FormulaExtractionError`).
- `model.to_vfgen(filepath=None, initial_conditions=..., population=...)` — VFGen XML.

## 4. Parameter Fitting

### Quick path (MLE)

```python
result = model.fit(                       # or fit_model(model, data, times, ...)
    {"I": observed_I}, times=times,
    params_to_fit={"beta": (0.1, 5.0), "gamma": (0.01, 1.0)},
    total_population=10000,
    variable_mapping={"cases": "I"},      # if series names differ from state vars
)
result.best_params          # dict
result.best_loss; result.convergence
result.fitted_model.summary()             # model simulated at best params
result.predictions                        # dict var -> fitted trajectory
```

Series mapped to a state variable are compared after interpolating model output
onto observation times — observation times need not match the solver grid.

### Full control (ModelFitter)

```python
from epimodels.fitting import Dataset, ModelFitter, ParameterSpec, InitialConditionSpec
from epimodels.fitting import PoissonLikelihood, ScipyOptimizer, MultiStartOptimizer

dataset = Dataset(model).register(
    name="cases", values=observed, times=times, state_variable="I")

fitter = ModelFitter(
    model=model,
    dataset=dataset,
    parameters_to_fit=[
        ParameterSpec("beta", bounds=(0.1, 5.0), initial=1.0, log_scale=True),
        ParameterSpec("gamma", bounds=(0.01, 1.0)),
    ],
    total_population=10000,
    loss_fn=PoissonLikelihood(),           # count data
    optimizer=MultiStartOptimizer(ScipyOptimizer(method="differential_evolution")),
    fit_initial_conditions=True,           # optionally fit ICs
    initial_condition_specs=[InitialConditionSpec("I", bounds=(1, 100))],
    fixed_params={"xi": 0.01},             # params held constant
)
result = fitter.fit()
profile = fitter.profile_likelihood("beta")   # {'values','losses','confidence_interval'}
```

### Choosing loss & optimizer

| Data | Loss |
|------|------|
| Counts, ~Poisson noise | `PoissonLikelihood` |
| Overdispersed counts | `NegativeBinomialLikelihood` |
| Continuous + Gaussian noise | `NormalLikelihood` (set `sigma`) |
| Outliers present | `HuberLoss` |
| Default | `SumOfSquaredErrors` |

| Situation | Optimizer |
|-----------|-----------|
| Default | `ScipyOptimizer()` (L-BFGS-B) |
| Multimodal landscape | `MultiStartOptimizer(ScipyOptimizer(...))` |
| No gradients / global | `ScipyOptimizer(method="differential_evolution")` or `NevergradOptimizer` |
| GPU | `JAXOptimizer` (Adam/SGD/RMSprop) |

Tip: fit `beta` on log scale (`log_scale=True`); it spans orders of magnitude.

### Bayesian inference (DE-MCMC)

```python
from epimodels.fitting import Dataset, ParameterSpec
from epimodels.fitting.bayes import fit_model_bayesian

dataset = Dataset(model).register(name="cases", values=obs, times=times, state_variable="I")
bayes = fit_model_bayesian(
    model, dataset,
    parameters_to_fit=[ParameterSpec("beta", bounds=(0.1, 5.0)),
                       ParameterSpec("gamma", bounds=(0.01, 1.0))],
    total_population=10000,
    likelihood="poisson",        # or "normal" (sigma=...) / "negative_binomial"
    num_samples=1000, num_warmup=500, seed=42,
    attach_fitted_model=True,    # simulate at MAP and attach to .model
)
bayes.map_estimate(); bayes.summary(); bayes.credible_intervals(0.95)
bayes.acceptance_rate; bayes.n_draws
bayes.trace_plot()               # traces + histograms
bayes.to_inference_data()        # ArviZ InferenceData (requires arviz)
```

Sugar API: `model.fit(..., method="bayes", likelihood="poisson")` (pass a Dataset
as `data` with `params_to_fit` as a list of `ParameterSpec`).

## 5. Analysis

### Built-in statistics

`model.summary()` (after simulation): peak_I, peak_time, duration, final_S,
final_R, attack_rate, t_start/t_end. `model.parameter_table(latex=False)` for a
parameters/values/symbols table.

### Symbolic analysis (R0, equilibria, stability, sensitivity)

```python
from epimodels.validation import SymbolicModel

sym = SymbolicModel()
sym.add_parameter("beta", positive=True)
sym.add_parameter("gamma", positive=True)
for v in ("S", "I", "R"):
    sym.add_variable(v, positive=True)
sym.set_total_population("N")
sym.define_ode("S", "-beta*S*I/N")
sym.define_ode("I", "beta*S*I/N - gamma*I")
sym.define_ode("R", "gamma*I")

sym.compute_R0_next_generation()       # next-generation matrix method
sym.find_disease_free_equilibrium()
sym.find_all_equilibria(); sym.find_endemic_equilibrium()
sym.check_stability_at_dfe(sym.compute_R0_next_generation())
sym.compute_jacobian(); sym.compute_eigenvalues(point)
sym.compute_sensitivity_matrix(...)    # dR0/dparam
sym.compute_elasticity_indices(...)    # normalized sensitivity
```

For discrete models use `define_difference_equation(var, rhs)` instead of
`define_ode`.

### Rt from incidence (model-free, Cori et al. 2013)

```python
from epimodels.rt import estimate_rt

rt = estimate_rt(incidence, window=7, si_mean=4.0, si_sd=2.0,
                 prior_mean=5.0, prior_sd=5.0, level=0.95)
rt.rt_mean; rt.rt_low; rt.rt_high      # one value per window
rt.plot()                              # band + Rt=1 threshold line
```

### Phase-space tools

```python
from epimodels.tools.phase import TimeDelayEmbedding, phase_portrait, find_optimal_embedding

emb = TimeDelayEmbedding(series, tau=10, dim=3).embed()
```

## 6. Uncertainty & Interventions

### Ensembles (parameter uncertainty)

```python
import numpy as np
from epimodels.ensembles import simulate_ensemble

rng = np.random.default_rng(0)
ens = simulate_ensemble(
    model, n_sims=200,
    param_sampler=lambda: {"beta": rng.uniform(1.5, 2.5), "gamma": 0.5},
    initial_conditions=[999, 1, 0], trange=[0, 100], totpop=1000,
    n_jobs=4,                       # parallel processes
)
ens.quantiles([0.025, 0.5, 0.975]); ens.summary(); ens.plot_band("I")
```

Runs on a common `n_points` grid so realizations are comparable.

### Intervention scenarios (continuous models only)

```python
from epimodels.interventions import Intervention, Scenario, ScenarioComparison

base = Scenario("baseline", model, params={"beta": 2.0, "gamma": 0.5},
                initial_conditions=[999, 1, 0], trange=[0, 100], totpop=1000)
lock = Scenario("lockdown", model, params={"beta": 2.0, "gamma": 0.5},
                initial_conditions=[999, 1, 0], trange=[0, 100], totpop=1000,
                interventions=[Intervention("beta", start=10, end=40, factor=0.4)])
cmp = ScenarioComparison(base, [lock]).run()
cmp.peak("I"); cmp.final_size("R"); cmp.plot("I")
```

`Intervention` takes exactly one of `factor` (multiplier) or `new_value`
(absolute); `end=None` means until the end of simulation.

## 7. Serialization

```python
from epimodels.io import save_model, load_model

save_model(model, "sir_run.json", include_traces=True)   # .yaml also supported
clone = load_model("sir_run.json")                       # registry-based reconstruction
```

Custom models must be registered (`@register_model`) **before** `load_model` can
reconstruct them.

## Workflow Checklist

1. **Clarify the question**: which outcome (peak, final size, Rt, parameters)?
2. **Pick the family** (decision table above); check built-ins first via
   `list_models()` / `repr(model)`.
3. **Simulate** with plausible params; check `model.summary()`, `model.R0`,
   positivity of trajectories.
4. **Validate analytically** with `SymbolicModel` (R0, equilibria, stability).
5. **Fit to data** if available: MLE for point estimates, Bayes for
   uncertainty; always sanity-check bounds and `result.convergence`.
6. **Propagate uncertainty** (ensembles / posterior samples) before reporting.
7. **Scenarios**: use `ScenarioComparison` for interventions; compare
   `final_size`/`peak`.
8. **Persist** results with `save_model` and report with plots/tables.

## Common Pitfalls

- **`ValidationError: Missing required parameters`** — every entry in
  `model.parameters` must be in the `params` dict; `N` is NOT one of them.
- **Init order** — `inits` follows the order of `model.state_variables`, print
  it if unsure.
- **`model.R0` is `None`** — R0 properties read `param_values`, populated only
  after a simulation call.
- **Fitting diverges** — widen/narrow bounds, use `log_scale=True` for rates,
  try `MultiStartOptimizer` or `differential_evolution`; check the loss is
  finite (simulations exploding => stiff system => use LSODA/BDF).
- **SDE/diffrax ImportError** — install `epimodels[jax]`; SDE wraps
  ContinuousModels only.
- **load_model KeyError** — custom model class not registered before loading.
- **`to_dataframe`/plot ImportError** — install extras: `epimodels[dataframe]`,
  `epimodels[plot]`.
- **Stochastic runs differ** — always pass `seed=` for reproducibility.

## Verification

```bash
pytest                                   # full suite
pytest tests/test_continuous_models.py::test_SIR   # single test
ruff check epimodels/                    # lint
```

When modifying library code (new models etc.), add tests under `tests/` and run
pytest before delivering. Docs live at epimodels.readthedocs.io.

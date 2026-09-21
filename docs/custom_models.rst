.. _custom_models:

*************
Custom Models
*************

Epimodels ships many ready-made models, but the real power of the library
is that **every model is a small Python class** you can write yourself.
All model families share the same :class:`~epimodels.BaseModel` foundation,
so a custom model automatically gets plotting, summaries, DataFrame export,
fitting (MLE and Bayesian), scenario analysis, serialization and the rest
of the toolbox.

This guide shows how to implement a model of each type and how to run the
basic analyses on it. All code blocks are cumulative — later sections reuse
the ``MySIRS`` class defined first — and were tested against the library.

Choosing a model family
=======================

Pick the family that matches your assumptions about time and stochasticity:

.. list-table::
   :header-rows: 1
   :widths: 22 28 34 16

   * - Family
     - You implement
     - Use when
     - Stochastic?
   * - :class:`~epimodels.continuous.ContinuousModel`
     - ``_model(t, y, params)``
     - Rates change smoothly; large populations
     - No (ODE)
   * - :class:`~epimodels.discrete.DiscreteModel`
     - ``model(...)``
     - Data are reported at discrete intervals
     - No (difference equations)
   * - :class:`~epimodels.stochastic.CTMC.CTMCModel`
     - ``transitions()``, ``propensity()``
     - Small populations, extinction/import events
     - Yes (exact, event-driven)
   * - :class:`~epimodels.network.NetworkModel`
     - ``_simulate_one(...)``
     - Contacts are explicit (households, hubs)
     - Yes (event-driven on graphs)
   * - :class:`~epimodels.sde.SDEModel`
     - wraps any ContinuousModel
     - Demographic noise between ODE and CTMC
     - Yes (Langevin)

A custom ODE model
==================

Subclass :class:`~epimodels.continuous.ContinuousModel` and define, in
``__init__``, the state variables and parameters (used for validation and
ordering), plus the right-hand side of the ODE system in ``_model``:

.. code-block:: python

    from epimodels.continuous import ContinuousModel
    from epimodels import BetaGammaR0Mixin


    class MySIRS(BetaGammaR0Mixin, ContinuousModel):
        """SIRS model: immunity wanes at rate alpha."""

        def __init__(self):
            super().__init__()
            self.state_variables = {
                "S": "Susceptible", "I": "Infectious", "R": "Recovered"
            }
            self.parameters = {
                "beta": r"$\beta$", "gamma": r"$\gamma$", "alpha": r"$\alpha$"
            }
            self.model_type = "MySIRS"

        @property
        def diagram(self) -> str:
            return r"""flowchart LR
    S(Susceptible) -->|beta| I(Infectious)
    I -->|gamma| R(Removed)
    R -->|alpha| S
    """

        def _model(self, t, y, params):
            S, I, R = y
            beta, gamma = params["beta"], params["gamma"]
            alpha, N = params["alpha"], params["N"]
            return [
                -beta * S * I / N + alpha * R,
                beta * S * I / N - gamma * I,
                gamma * I - alpha * R,
            ]

Key points:

* **Order matters**: initial conditions are matched to
  ``state_variables`` in insertion order; ``param_values`` is filled from
  ``parameters`` in insertion order.
* **The total population is injected**: the solver calls your ``_model``
  with ``N = totpop`` added to ``params``.
* :class:`~epimodels.BetaGammaR0Mixin` provides ``R0 = beta / gamma``.
  For other expressions, define your own ``R0`` property.
* The ``diagram`` property (Mermaid syntax) is optional but rendered by
  ``print(model)`` and the docs.

Running and inspecting it uses the standard API:

.. code-block:: python

    model = MySIRS()
    model([999, 1, 0], [0, 100], 1000,
          {"beta": 0.4, "gamma": 0.1, "alpha": 0.01})

    model.R0                 # 4.0
    model.summary()          # peak_I, peak_time, attack_rate, ...
    model.to_dataframe()     # pandas DataFrame (one column per variable)
    model.plot_traces(["I"])

    # chaining form
    model2 = MySIRS().simulate(
        [999, 1, 0], [0, 100], 1000,
        {"beta": 0.4, "gamma": 0.1, "alpha": 0.01},
    )

Parameter values are validated against the ``parameters`` dict before
integration: every declared parameter must be present and non-negative,
and initial conditions must be non-negative with sum ``<= totpop``.

Symbolic analysis of your model
===============================

Automatic formula extraction
----------------------------

``ContinuousModel`` can extract the symbolic form of your equations by
symbolically executing ``_model`` (requires SymPy):

.. code-block:: python

    f = model.get_formulas()
    # {'S': -I*S*beta/N + R*alpha,
    #  'I': -I*gamma + I*S*beta/N,
    #  'R':  I*gamma - R*alpha}

Extraction works when ``_model`` is written as straight-line arithmetic.
If it uses loops, conditionals, ``np.tanh`` or interpolation callables,
a :class:`~epimodels.FormulaExtractionError` is raised; in that case,
supply the formulas manually:

.. code-block:: python

    import sympy as sp

    S, I, R = sp.symbols("S I R")
    beta, gamma, alpha, N = sp.symbols("beta gamma alpha N")

    model._formulas = {
        "S": -beta * S * I / N + alpha * R,
        "I": beta * S * I / N - gamma * I,
        "R": gamma * I - alpha * R,
    }

With formulas available, the model can also be exported with
:meth:`~epimodels.continuous.ContinuousModel.to_vfgen` for external
bifurcation-continuation tools (see :doc:`exporters`).

R0, equilibria and stability
----------------------------

The :class:`~epimodels.validation.SymbolicModel` class performs symbolic
analysis on a *declarative* description of the same equations
(see :doc:`validation_system` for the full API):

.. code-block:: python

    from epimodels.validation import SymbolicModel

    sym = SymbolicModel()
    sym.add_parameter("beta", positive=True, real=True)
    sym.add_parameter("gamma", positive=True, real=True)
    sym.add_variable("S", positive=True)
    sym.add_variable("I", positive=True)
    sym.add_variable("R", positive=True)
    sym.set_total_population("N")

    sym.define_ode("S", "-beta*S*I/N + alpha*R")
    sym.define_ode("I", "beta*S*I/N - gamma*I")
    sym.define_ode("R", "gamma*I - alpha*R")

    R0 = sym.compute_R0_next_generation()        # beta/gamma
    dfe = sym.find_disease_free_equilibrium()    # {'S': N, 'I': 0, 'R': 0}

    sym.check_stability_at_dfe(R0)               # 'symbolic (depends on ...)'
    sym.check_stability_at_dfe(
        sym.substitute_values(R0, {"beta": 0.4, "gamma": 0.1})
    )                                            # 'unstable'

Elasticity of R0 (normalized forward sensitivity indices)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import sympy as sp

    R0_expr = sym.compute_R0_next_generation()
    for pname in ("beta", "gamma"):
        p = sym.get_parameter_symbol(pname)
        print(sp.simplify(sp.diff(R0_expr, p) * p / R0_expr))
    # 1   (R0 grows 1% per 1% increase in beta)
    # -1  (R0 falls 1% per 1% increase in gamma)

A custom discrete-time model
============================

Subclass :class:`~epimodels.discrete.DiscreteModel` and implement
``model(...)``, returning a dict of 1D arrays (including ``"time"``):

.. code-block:: python

    import numpy as np
    from epimodels.discrete import DiscreteModel
    from epimodels import BetaRR0Mixin


    class MyDiscreteSIR(BetaRR0Mixin, DiscreteModel):
        """Discrete-time SIR with explicit incidence tracking."""

        def __init__(self):
            super().__init__()
            self.state_variables = {
                "S": "Susceptible", "I": "Infectious", "R": "Recovered"
            }
            self.parameters = {"beta": r"$\beta$", "r": "r"}
            self.model_type = "MyDiscreteSIR"

        def model(self, inits, trange, totpop, params):
            n_steps = trange[1] - trange[0]
            S, I, R, L = (np.zeros(n_steps) for _ in range(4))
            S[0], I[0], R[0] = inits
            beta, r = params["beta"], params["r"]
            N = totpop

            for t in range(n_steps - 1):
                L[t + 1] = beta * S[t] * I[t] / N   # new cases this step
                S[t + 1] = S[t] - L[t + 1]
                I[t + 1] = I[t] + L[t + 1] - r * I[t]
                R[t + 1] = R[t] + r * I[t]

            return {"time": np.arange(*trange), "S": S, "I": I, "R": R, "L": L}


    dm = MyDiscreteSIR()
    dm([999, 1, 0], [0, 100], 1000, {"beta": 0.4, "r": 0.1})
    dm.R0                  # 4.0  (BetaRR0Mixin: beta / r)
    dm.summary()           # peak_I, attack_rate, ...
    dm.traces["L"]         # incidence series — handy against notification data

Anything you return besides the declared state variables (here ``L``) is
kept in ``traces`` but excluded from ``plot_traces``/``summary()``, which
only use declared variables.

A custom stochastic (CTMC) model
================================

Stochastic models describe the *events* of the system: a state-change
matrix and a propensity (hazard) function per event. The Gillespie solver
takes care of the rest:

.. code-block:: python

    from collections import OrderedDict
    import numpy as np
    from epimodels.stochastic.CTMC import CTMCModel
    from epimodels import BetaGammaR0Mixin


    class MySEIS(BetaGammaR0Mixin, CTMCModel):
        """Stochastic SEIS: exposed stage, no immunity."""

        def __init__(self):
            super().__init__()
            self.state_variables = OrderedDict(
                {"S": "Susceptible", "E": "Exposed", "I": "Infectious"}
            )
            self.parameters = OrderedDict(
                {"beta": r"$\beta$", "gamma": r"$\gamma$", "sigma": r"$\sigma$"}
            )
            self.events = OrderedDict({
                "infection": "S -> E",
                "infectious": "E -> I",
                "recovery": "I -> S",
            })
            self.model_type = "MySEIS CTMC"

        def transitions(self):
            # columns = events, rows = state variables (order as declared)
            #            infection  infectious  recovery
            return np.array([[-1,  0,  1],
                             [ 1, -1,  0],
                             [ 0,  1, -1]], dtype=np.int64)

        def propensity(self, params, state):
            S, E, I = state
            beta, gamma = params["beta"], params["gamma"]
            sigma, N = params["sigma"], params["N"]
            return np.array([beta * S * I / N, sigma * E, gamma * I])


    cm = MySEIS()
    cm([990, 5, 5], [0, 100], 1000,
       {"beta": 0.4, "gamma": 0.1, "sigma": 0.5},
       reps=50, seed=42)

    cm.R0                          # 4.0
    cm.get_mean()["I"]             # mean trajectory across replicates
    cm.get_quantiles([0.025, 0.975])[0.975]["I"]   # 97.5% envelope
    cm.plot_traces()

Contract:

* ``transitions()`` returns an ``(n_vars, n_events)`` integer matrix whose
  column *j* is the state change when event *j* fires.
* ``propensity(params, state)`` returns the hazard of each event given the
  current integer state; ``N`` is available in ``params``.
* States are integers — populations are individuals, not fractions.

A custom network model
======================

Network models spread the epidemic on a contact graph (a
`networkx <https://networkx.org>`_ graph, an adjacency dict or a symmetric
matrix). Subclass :class:`~epimodels.network.NetworkModel` and implement
the single-replicate event loop ``_simulate_one``:

.. code-block:: python

    import numpy as np
    from epimodels.network import NetworkModel


    class NetworkSI(NetworkModel):
        """SI epidemic: transmission along edges, no recovery."""

        def __init__(self, graph):
            super().__init__(graph)
            self.state_variables = {"S": "Susceptible", "I": "Infectious"}

        def _simulate_one(self, adj, initial_infected, beta, gamma,
                          t0, tmax, rng):
            state = np.zeros(self.n_nodes, dtype=np.int8)  # 0 = S, 1 = I
            infected = []
            for j in initial_infected:
                state[j] = 1
                infected.append(j)

            times_list, counts = [t0], []
            counts.append([self.n_nodes - len(infected), len(infected)])
            t = t0

            while infected:
                si_edges = sum(
                    1 for j in infected for nb in adj[j] if state[nb] == 0
                )
                if si_edges == 0:
                    break
                # next transmission arrives after an exponential wait
                t += rng.exponential(1.0 / (beta * si_edges))
                if t > tmax:
                    break
                self._infect_event(adj, state, infected, si_edges, rng)
                times_list.append(t)
                counts.append([int((state == 0).sum()), len(infected)])

            return np.asarray(times_list, dtype=float), np.asarray(counts, dtype=float)


    # a 100-node ring, no networkx required
    adjacency = {i: [(i + 1) % 100, (i - 1) % 100] for i in range(100)}
    net = NetworkSI(adjacency)
    net(3, [0, 20], {"beta": 0.3, "gamma": 0.0}, n_sims=10, seed=7)

    net.get_mean()["I"][-1]     # mean prevalence at tmax
    net.plot_traces("I")        # spaghetti plot + ensemble mean

Contract:

* ``_simulate_one`` receives the adjacency lists, the initially infected
  node indices, ``beta``/``gamma``, the time window and the shared random
  generator; it returns ``(times, counts)`` where ``counts[i]`` are the
  compartment counts right after ``times[i]``. ``gamma`` must be accepted
  even if unused (here, no recovery).
* Counts columns follow the declared ``state_variables``; set the class
  attribute ``keep_removed = True`` for SIR-type models with a permanent
  R compartment (enables ``final_size()``).
* The helper ``self._infect_event(...)`` transmits along one S–I edge,
  choosing the infector proportionally to its susceptible-neighbor count.

A custom SDE
============

Any :class:`~epimodels.continuous.ContinuousModel` — including your own —
can be wrapped as a Langevin SDE by :class:`~epimodels.sde.SDEModel`. The
default diffusion is the sqrt-demographic noise; you can pass a custom
diffusion function ``(t, y, params) -> (n_vars, n_noise)``:

.. code-block:: python

    from epimodels.sde import SDEModel


    def diffusion_I_only(t, y, params):
        """Additive noise in the I compartment only."""
        import jax.numpy as jnp
        return jnp.diag(jnp.array([0.0, 0.2, 0.0]))


    sde = SDEModel(MySIRS(), diffusion=diffusion_I_only)
    sde([999, 1, 0], [0, 100], 1000,
        {"beta": 0.5, "gamma": 0.25, "alpha": 0.02},
        n_sims=30, seed=0)

    sde.get_mean()["I"]          # ensemble mean
    sde.get_quantiles(0.95)["I"]  # (lower, upper) 95% band

    # or: keep the demographic default and just scale it
    sde2 = SDEModel(MySIRS(), noise_scale=1.5)

The diffusion function must be JAX-traceable (use ``jax.numpy``, not
``numpy``). Beware that Euler–Maruyama does not preserve non-negativity —
seed with a reasonable number of infectives and keep ``dt`` moderate
(see :doc:`sde`).

Registering and sharing custom models
=====================================

The decorator :func:`~epimodels.registry.register_model` makes a model
addressable by name through :func:`~epimodels.registry.get_model`:

.. code-block:: python

    from epimodels.registry import register_model, get_model, list_models


    @register_model("MyRegisteredSIRS", family="custom")
    class MyRegisteredSIRS(MySIRS):
        pass


    cls = get_model("MyRegisteredSIRS", family="custom")
    model = cls()

Saved models round-trip through JSON/YAML; custom classes are restored
via the registry or the module path recorded at save time:

.. code-block:: python

    from epimodels.io import save_model, load_model

    model([999, 1, 0], [0, 50], 1000,
          {"beta": 0.4, "gamma": 0.1, "alpha": 0.01})
    path = save_model(model, "my_sirs.json", include_traces=True)
    clone = load_model(path)
    clone.param_values   # {'beta': 0.4, 'gamma': 0.1, 'alpha': 0.01}

Rich validation for custom models
=================================

Beyond the simple dict-based validation, you can attach rich specifications
for parameters and variables — bounds, units, cross-parameter constraints —
which are enforced on every call (see :doc:`validation_system`):

.. code-block:: python

    from epimodels.validation import (
        ParameterSpec, VariableSpec, ModelConstraint,
    )


    class ValidatedSIRS(MySIRS):
        def __init__(self):
            super().__init__()
            self.define_parameter(ParameterSpec(
                name="beta", symbol=r"$\beta$",
                description="Transmission rate",
                bounds=(0, None), units="1/day",
                typical_range=(0.1, 2.0),
            ))
            self.define_parameter(ParameterSpec(
                name="gamma", symbol=r"$\gamma$",
                description="Recovery rate", bounds=(0, None), units="1/day",
            ))
            self.define_parameter(ParameterSpec(
                name="alpha", symbol=r"$\alpha$",
                description="Waning rate", bounds=(0, 1), units="1/day",
            ))
            for name, desc in [("S", "Susceptible"), ("I", "Infectious"),
                               ("R", "Recovered")]:
                self.define_variable(VariableSpec(name=name, symbol=name,
                                                  description=desc))
            self.add_constraint(ModelConstraint(
                expression="beta / gamma > 1",
                description="R0 > 1 required for an epidemic",
                severity="warning",
            ))


    from epimodels.exceptions import ValidationError

    vm = ValidatedSIRS()
    vm([999, 1, 0], [0, 100], 1000,
       {"beta": 0.4, "gamma": 0.1, "alpha": 0.01})      # ok

    try:
        vm([999, 1, 0], [0, 100], 1000,
           {"beta": -0.4, "gamma": 0.1, "alpha": 0.01})
    except ValidationError as e:
        print(e)   # Parameter 'beta' value -0.4 is below minimum bound 0

.. note::

    Two different ``ParameterSpec`` classes exist: the validation spec
    above (``epimodels.validation.ParameterSpec``, with symbols, units and
    constraints) and the fitting spec (``epimodels.fitting.ParameterSpec``,
    with bounds and initial/log-scale options used by
    :func:`~epimodels.fitting.bayes.fit_model_bayesian`). Import the one
    you need.

Analyses on your custom model
=============================

Everything the built-in models can do works on custom models too.

Parameter fitting (MLE and Bayesian)
------------------------------------

Simulate a ground truth, then recover the parameters from the ``I``
series. Note that the fitter re-simulates the model using **only** the
parameters in ``params_to_fit``, so all parameters your model requires
must be listed there:

.. code-block:: python

    import numpy as np

    truth = MySIRS()
    truth([990, 10, 0], [0, 60], 1000,
          {"beta": 0.4, "gamma": 0.1, "alpha": 0.01},
          t_eval=np.arange(0, 61, 2.0))
    obs = truth.to_dataframe()["I"].values
    times = np.arange(0, 61, 2.0)

    # maximum likelihood
    result = MySIRS().fit(
        {"I": obs}, times=times,
        params_to_fit={"beta": (0.05, 2.0), "gamma": (0.01, 1.0),
                       "alpha": (0.0, 0.2)},
        total_population=1000,
    )
    result.best_params        # recovers ~ (0.40, 0.10, 0.01)

    # full posterior with DE-MCMC
    bres = MySIRS().fit(
        {"I": obs}, times=times,
        params_to_fit={"beta": (0.05, 2.0), "gamma": (0.01, 1.0),
                       "alpha": (0.0, 0.2)},
        total_population=1000,
        method="bayes", likelihood="normal", sigma=5.0,
        num_samples=400, num_warmup=200, seed=1,
    )
    bres.samples["beta"]          # posterior draws
    bres.credible_intervals()

See :doc:`fitting` and :doc:`bayes` for loss functions, profile
likelihoods and diagnostics.

Scenario analysis and interventions
-----------------------------------

.. code-block:: python

    from epimodels.interventions import Intervention, Scenario, ScenarioComparison

    params = {"beta": 0.4, "gamma": 0.1, "alpha": 0.01}
    common = dict(params=params, initial_conditions=[999, 1, 0],
                  trange=[0, 120], totpop=1000)

    baseline = Scenario("baseline", MySIRS(), **common)
    lockdown = Scenario(
        "lockdown", MySIRS(),
        interventions=[Intervention("beta", start=20, end=60, factor=0.4)],
        **common,
    )

    comparison = ScenarioComparison(baseline, [lockdown]).run()
    comparison.peak("I")       # {'baseline': 402.2, 'lockdown': 233.3}
    comparison.plot("I")

Uncertainty ensembles
---------------------

.. code-block:: python

    from epimodels.ensembles import simulate_ensemble

    rng = np.random.default_rng(0)
    ensemble = simulate_ensemble(
        MySIRS(), n_sims=50,
        param_sampler=lambda: {
            "beta": rng.uniform(0.3, 0.5), "gamma": 0.1, "alpha": 0.01
        },
        initial_conditions=[999, 1, 0], trange=[0, 100], totpop=1000,
    )
    q = ensemble.quantiles([0.025, 0.5, 0.975])
    q["I"][0.5]                # median trajectory

Phase-space analysis of the output
----------------------------------

.. code-block:: python

    from epimodels.tools.phase import find_optimal_embedding, phase_portrait

    emb = find_optimal_embedding(model.traces["I"], tau_max=30, dim_max=5)
    emb["tau"], emb["dim"]     # optimal delay / embedding dimension

    phase_portrait(model.traces["S"], model.traces["I"])

Pitfalls and tips
=================

* **Parameters are ordered by declaration**: ``param_values`` and
  ``to_vfgen``/symbolic extraction rely on the order of ``self.parameters``
  and ``self.state_variables``; declare them once, in ``__init__``.
* **Keep ``_model`` symbolic-friendly** (straight-line arithmetic, no
  loops/branches) so ``get_formulas()`` works; otherwise provide
  ``_formulas`` manually.
* **All declared parameters must be passed** in every call (and to the
  fitters). If a parameter is optional, give it a default in ``__init__``
  and don't declare it.
* **Time-dependent parameters** (callables) are supported by ``_model``,
  but block JAX tracing — such models cannot run under
  :class:`~epimodels.sde.SDEModel`.
* **CTMC states are integers**; keep ``transitions`` integer-typed and
  propensities non-negative.
* **Network counts are integers per node**; don't rescale them by
  population fractions — ``summary()`` handles raw counts.
* Prefer the ``validate=True`` default during development; you can disable
  it with ``validate=False`` in production runs for speed.

References
==========

The methods behind these analyses — next-generation R0 (Diekmann et al.,
1990; van den Driessche & Watmough, 2002), sensitivity indices (Chitnis et
al., 2008), likelihood-based fitting (King et al., 2015), Bayesian DE-MCMC
(ter Braak, 2006) and the numerical solvers — are cited in the
:ref:`references` page.

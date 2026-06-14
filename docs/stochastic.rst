Stochastic CTMC Models
======================

Epimodels provides stochastic epidemic models based on **Continuous-Time
Markov Chains** (CTMCs), solved using the **Gillespie Stochastic Simulation
Algorithm** (SSA). These models are the stochastic counterparts of the
deterministic ODE models and are especially useful when population sizes
are small or when extinction risk analysis is required.

.. contents::
   :local:
   :depth: 1

Available Models
----------------

Four classic models are available in stochastic form:

===========  =================  ===================================
Model        Compartments       Description
===========  =================  ===================================
``SIR``      S, I, R            Classic susceptible-infectious-removed
``SIS``      S, I               No immunity, reinfection possible
``SIRS``     S, I, R            Waning immunity
``SEIR``     S, E, I, R         Latent period (exposed)
===========  =================  ===================================

All models inherit from :class:`~epimodels.stochastic.CTMC.models.CTMCModel`,
which extends :class:`~epimodels.BaseModel` with stochastic simulation
capabilities.

Quick Start
-----------

.. code-block:: python

    from epimodels.stochastic.CTMC import SIR

    model = SIR()
    model(
        inits=[990, 10, 0],
        trange=[0, 100],
        totpop=1000,
        params={'beta': 0.3, 'gamma': 0.1},
        reps=100,
        seed=42,
    )
    model.plot_traces()

Simulation Parameters
---------------------

CTMC models accept all standard model parameters plus stochastic-specific
ones:

==================  =======  =====================================================
Parameter           Default  Description
==================  =======  =====================================================
``reps``            1        Number of independent stochastic trajectories
``seed``            None     Random seed for reproducibility
``n_jobs``          1        Number of parallel processes for replicate execution
``n_points``        1000     Number of time points in the output grid
``solver``          None     Custom solver (defaults to :class:`GillespieSolver`)
==================  =======  =====================================================

Parallel Execution
~~~~~~~~~~~~~~~~~~

For large numbers of replicates, set ``n_jobs`` > 1 to use multiprocessing:

.. code-block:: python

    model = SIR()
    model([990, 10, 0], [0, 100], 1000,
          {'beta': 0.3, 'gamma': 0.1},
          reps=1000, n_jobs=4, seed=42)

Accessing Results
-----------------

Single Replicate
~~~~~~~~~~~~~~~~

:meth:`~epimodels.stochastic.CTMC.models.CTMCModel.get_replicate` returns
a single replicate's trajectory as a dictionary:

.. code-block:: python

    rep_0 = model.get_replicate(0)
    print(rep_0['time'])  # Time points
    print(rep_0['I'])     # Infectious trajectory

Summary Statistics
~~~~~~~~~~~~~~~~~~

Compute summary statistics across all replicates:

.. code-block:: python

    # Mean trajectory across replicates
    mean_traces = model.get_mean()
    plt.plot(mean_traces['time'], mean_traces['I'], label='Mean I')

    # Variance across replicates
    var_traces = model.get_variance()
    plt.fill_between(
        mean_traces['time'],
        mean_traces['I'] - np.sqrt(var_traces['I']),
        mean_traces['I'] + np.sqrt(var_traces['I']),
        alpha=0.3, label='±1 std'
    )

Quantile Bands
~~~~~~~~~~~~~~

:meth:`~epimodels.stochastic.CTMC.models.CTMCModel.get_quantiles` returns
quantile envelopes, useful for credible/confidence intervals:

.. code-block:: python

    # 95% credible interval
    q = model.get_quantiles([0.025, 0.5, 0.975])

    plt.plot(q['time'], q['I_0.5'], 'b-', label='Median')
    plt.fill_between(q['time'], q['I_0.025'], q['I_0.975'],
                     alpha=0.3, label='95% CI')

Event Times
~~~~~~~~~~~

:meth:`~epimodels.stochastic.CTMC.models.CTMCModel.get_event_times`
returns the times at which specific events occurred across replicates:

.. code-block:: python

    peak_times = model.get_event_times('peak_detected')
    print(f"Peak occurred at t={np.mean(peak_times):.1f} ± {np.std(peak_times):.1f}")

DataFrame Export
~~~~~~~~~~~~~~~~

Export a specific replicate to a pandas DataFrame:

.. code-block:: python

    df = model.to_dataframe(replicate=0)

The Gillespie Solver
--------------------

:class:`~epimodels.stochastic.CTMC.solvers.GillespieSolver` implements
the **Gillespie Direct Method** for exact stochastic simulation. At each
step:

1. Compute propensities for all transition events
2. Draw time to next event from an exponential distribution
3. Select which event fires (weighted random selection)
4. Update the state vector

.. code-block:: python

    from epimodels.stochastic.CTMC import SIR, GillespieSolver

    solver = GillespieSolver()
    model = SIR()
    model([990, 10, 0], [0, 100], 1000,
          {'beta': 0.3, 'gamma': 0.1},
          reps=50, solver=solver)

When to Use CTMC Models
-----------------------

====================================  ============================================
Scenario                              Notes
====================================  ============================================
Small populations (< 10,000)          Stochastic effects are significant
Extinction risk analysis              Only stochastic models can capture extinction
Confidence intervals                  Run multiple replicates for uncertainty
Large populations (> 100,000)         ODE models may suffice; CTMC is slow for many replicates
====================================  ============================================

API Reference
-------------

Models
~~~~~~

.. autoclass:: epimodels.stochastic.CTMC.models.CTMCModel
   :members:
   :undoc-members:

.. autoclass:: epimodels.stochastic.CTMC.models.SIR
   :members:
   :undoc-members:

.. autoclass:: epimodels.stochastic.CTMC.models.SIS
   :members:
   :undoc-members:

.. autoclass:: epimodels.stochastic.CTMC.models.SIRS
   :members:
   :undoc-members:

.. autoclass:: epimodels.stochastic.CTMC.models.SEIR
   :members:
   :undoc-members:

Solvers
~~~~~~~

.. autoclass:: epimodels.stochastic.CTMC.solvers.GillespieSolver
   :members:
   :undoc-members:

.. autoclass:: epimodels.stochastic.CTMC.solvers.CTMCTrajectory
   :members:
   :undoc-members:

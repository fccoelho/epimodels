=========
Epimodels
=========

**Epimodels** is a Python library of mathematical models for epidemiology, designed for simulation studies and parameter inference. It contains deterministic models in both continuous (ODE-based) and discrete (difference equation) time, stochastic CTMC and network models, SDE extensions, a comprehensive fitting and Bayesian inference framework, symbolic analysis tools, Rt estimation, and multiple solver backends.

.. note::

    This library is under active development. Contributions are welcome.

Installation
============

.. code-block:: bash

    pip install epimodels

Optional extras:

.. code-block:: bash

    pip install epimodels[plot]       # matplotlib plotting
    pip install epimodels[dataframe]  # pandas DataFrame support
    pip install epimodels[jax]        # diffrax/JAX GPU solvers and SDEs
    pip install epimodels[network]    # network models (networkx)
    pip install epimodels[yaml]       # YAML model files

Quick Start
===========

.. code-block:: python

    from epimodels.continuous import SIR

    model = SIR()
    model([1000, 1, 0], [0, 50], 1001, {'beta': 0.3, 'gamma': 0.1})
    
    print(f"R0 = {model.R0}")  # Basic reproduction number
    print(model.summary())      # Epidemic statistics
    model.plot_traces()         # Plot results

Parameter Fitting
=================

The sugar API fits any model in one call, either by maximum likelihood or
Bayesian inference:

.. code-block:: python

    from epimodels.continuous import SIR

    model = SIR()
    result = model.fit(
        {"I": [1, 3, 8, 20, 50, 80, 60]},
        times=[0, 1, 2, 3, 5, 7, 10],
        params_to_fit={"beta": (0.1, 5.0), "gamma": (0.01, 1.0)},
        total_population=10000,
    )
    print(result.best_params)

Contents
========

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   geting_started
   models
   solvers
   validation_system
   fitting
   bayes
   stochastic
   sde
   network
   scenarios
   rt
   phase_space
   registry
   serialization
   exporters

.. toctree::
   :maxdepth: 2
   :caption: Examples:

   Examples/Continuous_models
   Examples/Stochastic_CTMC_models
   Examples/Discrete_models
   Examples/API_features
   Examples/Phase_space_analysis
   Examples/Model_Fitting
   Examples/SISLogistic_fitting
   Examples/SISLogistic_fitting_real_data
   Examples/SIRS_parameter_inference
   Examples/SIRS_Nonautonomous_Example
   Examples/neipel_heterogeneous_sir_example
   Examples/Validation_Framework
   Examples/Advanced_Analytics
   Examples/Model_Fitting_SIRSEI
   Examples/Model_Fitting_SIRSEI_Prevalence
   Examples/SEIRS_SEI_Model_Environment
   Examples/Model_Fitting_SEIRS_SEI

.. toctree::
   :maxdepth: 2
   :caption: Reference:

   api/modules

.. toctree::
   :maxdepth: 1
   :caption: About:

   License <license>
   Authors <authors>
   Changelog <changelog>


API Overview
============

Solvers
-------

ODE Solvers (Unified interface)
  - :class:`~epimodels.solvers.ScipySolver` - Scipy-based solver (CPU)
  - :class:`~epimodels.solvers.DiffraxSolver` - JAX-accelerated solver (GPU)

CTMC Solvers (Stochastic simulation)
  - :class:`~epimodels.stochastic.CTMC.solvers.GillespieSolver` - Gillespie Direct Method (SSA)

Model Classes
-------------

Continuous Models (ODE-based)
  - :class:`~epimodels.continuous.models.SIR` - Susceptible-Infectious-Removed
  - :class:`~epimodels.continuous.models.SIS` - Susceptible-Infectious-Susceptible
  - :class:`~epimodels.continuous.models.SIRS` - Susceptible-Infectious-Removed-Susceptible
  - :class:`~epimodels.continuous.models.SEIR` - Susceptible-Exposed-Infectious-Removed
  - :class:`~epimodels.continuous.models.SEQIAHR` - COVID-19 model with quarantine
  - :class:`~epimodels.continuous.models.Dengue4Strain` - 4-strain dengue model
  - :class:`~epimodels.continuous.models.SIRSEI` - Malaria vector-host with climate forcing
  - :class:`~epimodels.continuous.models.SIRSEIData` - Malaria with real climate data
  - :class:`~epimodels.continuous.models.SEIRS_SEI` - Vector-borne with environmental effects
  - :class:`~epimodels.continuous.models.SIR2Strain` - Two-strain SIR with cross-immunity
  - :class:`~epimodels.continuous.models.SIR1D` - 1D reduced SIR (beta/gamma tracking)
  - :class:`~epimodels.continuous.models.SISLogistic` - SIS with logistic population growth
  - :class:`~epimodels.continuous.models.SIRSNonAutonomous` - SIRS with time-dependent parameters
  - :class:`~epimodels.continuous.models.NeipelHeterogeneousSIR` - Heterogeneous susceptibility
  - :class:`~epimodels.continuous.models.EbolaSEIHFRV` - Ebola with hospital and funeral transmission

Discrete Models (Difference equations)
  - :class:`~epimodels.discrete.models.SIR` - Susceptible-Infectious-Removed
  - :class:`~epimodels.discrete.models.SIS` - Susceptible-Infectious-Susceptible
  - :class:`~epimodels.discrete.models.SEIR` - Susceptible-Exposed-Infectious-Removed
  - :class:`~epimodels.discrete.models.SEIS` - Susceptible-Exposed-Infectious-Susceptible
  - :class:`~epimodels.discrete.models.SIRS` - Susceptible-Infectious-Removed-Susceptible
  - :class:`~epimodels.discrete.models.SEQIAHR` - COVID-19 model with quarantine
  - :class:`~epimodels.discrete.models.Influenza` - Age-structured influenza model
  - :class:`~epimodels.discrete.models.SIpRpS` - Partial immunity waning
  - :class:`~epimodels.discrete.models.SEIpRpS` - Exposed + partial immunity
  - :class:`~epimodels.discrete.models.SIpR` - Secondary infections from recovered
  - :class:`~epimodels.discrete.models.SEIpR` - Exposed + secondary infections from R

Stochastic Models (CTMC / Gillespie SSA)
  - :class:`~epimodels.stochastic.CTMC.models.SIR` - Stochastic SIR
  - :class:`~epimodels.stochastic.CTMC.models.SIS` - Stochastic SIS
  - :class:`~epimodels.stochastic.CTMC.models.SIRS` - Stochastic SIRS (waning immunity)
  - :class:`~epimodels.stochastic.CTMC.models.SEIR` - Stochastic SEIR (with latent period)

Network Models (event-driven, graph-based)
  - :class:`~epimodels.network.NetworkSIR` - SIR on networkx graphs, adjacency dicts or matrices
  - :class:`~epimodels.network.NetworkSIS` - SIS on networks (infection can become endemic)

Stochastic Differential Equations
  - :class:`~epimodels.sde.SDEModel` - Demographic-noise SDE wrapper for any continuous model

Analysis and Inference Tools
----------------------------

  - :func:`~epimodels.rt.estimate_rt` - Time-varying reproduction number from incidence (Cori/EpiEstim)
  - :class:`~epimodels.sde.SDEModel` - SDE simulation
  - :func:`~epimodels.ensembles.simulate_ensemble` - Parameter-uncertainty ensembles
  - :class:`~epimodels.ensembles.TraceEnsemble` - Ensemble results with quantile bands
  - :class:`~epimodels.interventions.Intervention` - Time-bounded parameter change
  - :class:`~epimodels.interventions.Scenario` - Model + parameters + interventions
  - :class:`~epimodels.interventions.ScenarioComparison` - Side-by-side scenario runs
  - :func:`~epimodels.fitting.bayes.fit_model_bayesian` - Bayesian (DE-MCMC) inference
  - :class:`~epimodels.fitting.bayes.BayesianFitResult` - Posterior samples, credible intervals, ArviZ export

Registry and Serialization
--------------------------

  - :func:`~epimodels.registry.get_model` - Look up models by name and family
  - :func:`~epimodels.registry.list_models` - List registered models
  - :func:`~epimodels.io.save_model` / :func:`~epimodels.io.load_model` - JSON/YAML model files

Fitting Module
--------------
  - :class:`~epimodels.fitting.ModelFitter` - Full-featured parameter fitter
  - :func:`~epimodels.fitting.fit_model` - Convenience fitting function
  - :class:`~epimodels.fitting.Dataset` - Observed data container
  - :class:`~epimodels.fitting.ScipyOptimizer` - Scipy-based optimizer
  - :class:`~epimodels.fitting.JAXOptimizer` - Projected gradient optimizer
  - :class:`~epimodels.fitting.MultiStartOptimizer` - Multi-start optimizer

Common Methods
--------------

All models inherit from :class:`~epimodels.BaseModel` and share these methods:

=======================================  =====================================================
Method                                  Description
=======================================  =====================================================
``__call__(inits, trange, N, params)``  Run the simulation
``simulate(...)``                       Run and return the model (chainable)
``fit(data, params, ...)``              Fit to data (``method="mle"`` or ``"bayes"``)
``plot_traces()``                       Plot simulation results
``to_dataframe()``                      Export to pandas DataFrame
``to_dict()``                           Get a copy of traces
``summary()``                           Get epidemic statistics
``copy()``                              Create a model copy
``reset()``                             Clear simulation results
``R0``                                  Basic reproduction number (property)
``diagram``                             Mermaid compartment diagram (property)
=======================================  =====================================================

Stochastic models (CTMC) also provide:

=======================================  =====================================================
Method                                  Description
=======================================  =====================================================
``get_mean()``                          Mean trajectory across replicates
``get_variance()``                      Variance across replicates
``get_quantiles(q)``                    Quantile trajectories
``get_replicate(i)``                    Single replicate as dict
``get_event_times(event)``              Event occurrence times
``to_dataframe(replicate=i)``           DataFrame of specific replicate
=======================================  =====================================================


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

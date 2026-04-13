=========
Epimodels
=========

**Epimodels** is a Python library of mathematical models for epidemiology, designed for simulation studies and parameter inference. It contains deterministic models in both continuous (ODE-based) and discrete (difference equation) time, along with a comprehensive fitting framework, symbolic analysis tools, and multiple solver backends.

.. note::

    This library is under active development. Contributions are welcome.

Installation
============

.. code-block:: bash

    pip install epimodels

For pandas DataFrame support:

.. code-block:: bash

    pip install epimodels[dataframe]

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

.. code-block:: python

    from epimodels.continuous import SIR
    from epimodels.fitting import fit_model, Dataset

    model = SIR()
    dataset = Dataset()
    dataset.add_series("I", times=[0, 1, 2, 3, 5, 7, 10], values=[1, 3, 8, 20, 50, 80, 60])

    result = fit_model(
        model, dataset,
        params={"beta": (0.1, 5.0), "gamma": (0.01, 1.0)},
        initial_conditions=[1000, 1, 0],
        time_range=[0, 10],
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

.. toctree::
   :maxdepth: 2
   :caption: Examples:

   Examples/Continuous_models
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
  - :class:`~epimodels.continuous.models.SISLogistic` - SIS with logistic population growth
  - :class:`~epimodels.continuous.models.SIRSNonAutonomous` - SIRS with time-dependent parameters
  - :class:`~epimodels.continuous.models.NeipelHeterogeneousSIR` - Heterogeneous susceptibility

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

Fitting Module
--------------
  - :class:`~epimodels.fitting.ModelFitter` - Full-featured parameter fitter
  - :func:`~epimodels.fitting.fit_model` - Convenience fitting function
  - :class:`~epimodels.fitting.Dataset` - Observed data container
  - :class:`~epimodels.fitting.ScipyOptimizer` - Scipy-based optimizer
  - :class:`~epimodels.fitting.JAXOptimizer` - JAX/GPU optimizer
  - :class:`~epimodels.fitting.MultiStartOptimizer` - Multi-start optimizer

Common Methods
--------------

All models inherit from :class:`~epimodels.BaseModel` and share these methods:

====================================  =====================================================
Method                               Description
====================================  =====================================================
``__call__(inits, trange, N, params)``  Run the simulation
``plot_traces()``                     Plot simulation results
``to_dataframe()``                    Export to pandas DataFrame
``to_dict()``                         Get a copy of traces
``summary()``                         Get epidemic statistics
``copy()``                            Create a model copy
``reset()``                           Clear simulation results
``R0``                                Basic reproduction number (property)
``diagram``                           Mermaid compartment diagram (property)
====================================  =====================================================


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

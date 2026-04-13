.. _changes:

Changelog
=========

Version 1.1.0 (2026-04-11)
==========================

Added
-----

* **SIRSNonAutonomous** continuous model with time-dependent transmission, recovery, and waning immunity parameters (callable functions)
* **SISLogistic** fitting with real epidemiological data support
* **SIRS parameter inference** with reduced parameter space, bounded optimization, and RK23 solver
* **Model fitting framework** (``epimodels.fitting``) -- full-featured parameter estimation:
  - 7 loss functions (SSE, Weighted SSE, Poisson, Negative Binomial, Normal, Log-likelihood, Huber)
  - 4 optimizers (Scipy, JAX, Nevergrad, Multi-start)
  - Dataset management with time series validation
  - Profile likelihood confidence intervals
  - Automatic initial condition estimation
* **SIRSEIData** -- climate-data-driven malaria model with real temperature/precipitation interpolation
* **SEIRS_SEI** -- vector-borne model with deforestation and forest fire environmental effects
* **SIR2Strain** -- two-strain SIR with cross-immunity and vital dynamics
* **SISLogistic** -- SIS model with logistic population growth
* **NeipelHeterogeneousSIR** -- heterogeneous susceptibility model (Neipel et al. 2020)
* **VFGen exporter** for symbolic model export to XML format
* **Phase space tools** -- time delay embedding, mutual information, Cao's method
* **SymbolicModel** analysis framework -- R0 computation, equilibrium finding, stability analysis, sensitivity/elasticity, parameter importance ranking
* **Mermaid diagram generation** on all model classes (``model.diagram`` property)

Changed
-------

* Fixed SISLogistic R0 parametrization
* Updated notebooks and examples for new models and fitting workflows

Removed
-------

* Obsolete run scripts

Version 1.0.2
=============

* Package definition fixes

Version 1.0.1
=============

* Initial PyPI release

Version 1.0.0
=============

* First stable release

Version 0.5.2
=============

* Model fitting tutorial notebook

Version 0.5.1
=============

* SIRS non-autonomous model corrections

Version 0.5.0
=============

* Validation framework implementation
* Rich parameter specifications

Version 0.4.3
=============

* Minor bug fixes

Version 0.4.2
=============

* SEIRS-SEI model with environmental factors

Version 0.4.1
=============

* Solver interface improvements

Version 0.4.0
=============

* Diffrax/JAX solver support
* Performance benchmarks

Version 0.1
==========

* Feature A added
* FIX: nasty bug #1729 fixed

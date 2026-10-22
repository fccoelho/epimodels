.. _changes:

Changelog
=========

Version 1.5.0 (2026-10-22)
==========================

Added
-----

* **Age-structured MSEIR model** --
  ``epimodels.continuous.age_structured.AgeStructuredSEIR``: a continuous-time
  age-structured SEIR with a maternally-immune compartment, an explicit ageing
  chain, optional births/mortality and optional routine vaccination. Exposes
  ``R0`` through the next-generation matrix, ``scale_beta_for_r0`` to calibrate
  transmission to a target reproduction number, ``initial_conditions`` to
  allocate a population structure across compartments without losing anybody,
  and ``vaccination_rate_for_coverage`` to translate a coverage target into a
  rate that competes with ageing. Registered in the model registry as
  ``"AgeStructuredSEIR"`` (family ``continuous``), so ``get_model``,
  ``simulate_ensemble``, ``ScenarioComparison``, ``save_model`` and
  ``to_dataframe`` all work unchanged; the age-aggregated compartments are
  additionally exposed in ``traces`` under ``"M"``/``"S"``/``"E"``/``"I"``/``"R"``.
* **Contact-matrix utilities** -- ``epimodels.tools.contact_matrix``:
  ``polymod_like_contact_matrix`` builds a POLYMOD-shaped matrix that satisfies
  detailed balance by construction (``N_i C_ij == N_j C_ji``, verified to
  machine precision) for any population structure; ``enforce_reciprocity`` and
  ``check_reciprocity`` handle matrices from outside sources;
  ``aggregate_contact_matrix`` pools a published matrix onto a coarser age grid
  while preserving reciprocity; ``age_group_labels`` labels the groups.
* **Custom Models guide** in the documentation; ``$...$`` math is now rendered
  via MathJax 3, and figures generated from the documented examples are included
* **Worked example notebook** for ``AgeStructuredSEIR``
  (``docs/Examples/Age_Structured_SEIR.ipynb``)

Changed
-------

* Documentation: verified citations added for the methodologies behind the
  v1.4.0 features (Rt estimation, SDE layer, network models, Bayesian inference)
* Dependencies: ``jupyterlab`` and ``mistune`` bumped in the lockfile
  (dependabot #42, #39)
* Developer tooling: ``epimodels-dev`` agent skill for model creation,
  simulation and analysis

Version 1.4.0 (2026-09-17)
==========================

Added
-----

* **Model registry** -- ``epimodels.get_model(name, family=...)`` and
  ``list_models()`` for string-based lookup across continuous/discrete/stochastic
  families, with a ``@register_model`` decorator for custom models
* **Intervention scenarios** -- ``epimodels.interventions`` with ``Intervention``,
  ``Scenario`` and ``ScenarioComparison`` for time-bounded parameter changes
  (e.g. lockdowns) with comparison plots and final-size/peak metrics
* **Uncertainty ensembles** -- ``epimodels.ensembles.simulate_ensemble`` runs many
  simulations with sampled parameters/initial conditions; ``TraceEnsemble`` provides
  quantiles, summaries and uncertainty-band plots
* **Bayesian inference** -- ``epimodels.fitting.bayes.fit_model_bayesian`` implements
  DE-MCMC posterior sampling (normal/Poisson/negative-binomial observation models),
  with MAP estimates, credible intervals, trace plots and optional ArviZ export
* **Rt estimation** -- ``epimodels.rt.estimate_rt`` implements the Cori et al. (2013)
  sliding-window reproduction-number estimator (EpiEstim method) with gamma
  serial interval, gamma prior/posterior and credible bands
* **SDE layer** -- ``epimodels.sde.SDEModel`` wraps any continuous model as a
  Langevin SDE with demographic square-root noise (or a custom diffusion),
  integrated with diffrax/JAX; multi-replicate support with means/quantiles
* **Network models** -- ``epimodels.network.NetworkSIR``/``NetworkSIS``: event-driven
  epidemics on networkx graphs, adjacency dicts or matrices, with replicates,
  final-size, means/quantiles and band plots (``[network]`` extra)
* **Model serialization** -- ``epimodels.io.save_model``/``load_model`` store model
  specs (optionally with traces) as JSON or YAML; reconstruction goes through the
  registry with module-path fallback for custom models
* **Sugar API** -- ``model.simulate(...)`` returns the model for chaining and
  ``model.fit(data, params_to_fit, ..., method="mle"|"bayes")`` dispatches into
  the fitting/bayesian machinery from any model instance
* ``BetaGammaR0Mixin``/``BetaRR0Mixin`` shared R0 properties (replacing 14 duplicated
  implementations)
* ``ModelFitter(raise_on_error=True)`` option to surface model-evaluation failures

Fixed
-----

* Discrete SIS no longer expects 3 initial conditions for a 2-compartment model
* Discrete SIRS ``R0`` read a nonexistent parameter and always returned ``None``
* ``Influenza`` result key typo (``Igl`` -> ``Ig1``) and missing ``run`` alias
* SEQIAHR (continuous and discrete) unpacked parameters by dict order, silently
  breaking with differently-ordered dicts
* ``SIRSEI.R0``/``R0_t`` accessed unchecked parameters, raising ``KeyError``
  instead of returning ``None``
* ``ContinuousModel.run`` mutated the caller's parameter dict
* ``BaseModel.copy()`` was shallow: copies shared parameter dicts, specs and formulas
* CTMC parallel replicates (``n_jobs > 1``) failed because a closure cannot be pickled
* ``DiffraxSolver`` ignored ``t_eval`` (hardcoded 100-point output grid) and
  converted state to a Python list on every RHS evaluation
* ``JAXOptimizer`` produced constant gradients under autodiff (``float()`` truncation)
  and misused the optimistix API; rewritten as projected gradient descent with
  finite-difference gradients (no external dependency)
* Importing the package no longer writes ``epimodels.log`` to the working directory
  (import-time ``logging.basicConfig`` removed) or require matplotlib
* Legacy ``gillespie.py``: undeclared ``tqdm`` import removed, no-op "validation"
  loops actually convert/clip values now, bounded worker pool
* Malformed LaTeX in parameter tables and symbols (``\begin[l|c|c]``, unclosed ``$``)
* ``SymbolicModel`` silent ``except: pass/None`` failures now logged;
  unreachable dead code removed
* ``SEQIAHR`` and ``SIRSEI`` are now symbolically extractable (numpy function calls
  in ``_model`` are mapped to SymPy during extraction)

Changed
-------

* Packaging: ``matplotlib``/``pandas``/``jax``/``diffrax``/``ipykernel`` are no longer
  hard dependencies -- use extras ``[plot]``, ``[dataframe]``, ``[jax]``;
  ``scipy-stubs`` moved to dev; added ``[build-system]``, license expression,
  fixed classifiers; removed stale ``src/`` layout and ``requirements.txt``
* Performance: ``SymbolicModel`` caches symbolic R0/Jacobian; vectorized CTMC grid
  interpolation, ``phase.py`` mutual information and Cao E-statistic
* CTMC solvers share a template-method trajectory loop

Version 1.3.0 (2026-06-14)
==========================

Added
-----

* **EbolaSEIHFRV** model -- SEIHFR-V compartmental model for Ebola epidemic dynamics
  with community, hospital, and funeral transmission pathways, ring vaccination
  (rVSV-ZEBOV / Ervebo), and next-generation matrix R0 decomposition
* Example notebook ``Ebola_SEIHFRV_Example.ipynb`` with scenario analysis, R0
  decomposition, and vaccination timing analysis based on the DRC technical report
  (doi:10.5281/zenodo.20634292)

Version 1.2.0 (2026-05-13)
==========================

Added
-----

* **Documentation update** -- new user guide pages for previously undocumented features:
  - Phase space analysis tools (TimeDelayEmbedding, mutual_information, Cao's method)
  - Stochastic CTMC models (Gillespie SSA, replicate methods, quantile bands)
  - VFGen XML exporter for external tool interoperability
* **SIR1D** model added to documentation and export list
* **LogLikelihood** loss function documented in fitting guide
* **InitialConditionSpec** usage section added to fitting docs
* Updated validation system docs to reflect implemented symbolic analysis features

Fixed
-----

* Added ``SISLogistic``, ``SIRSNonAutonomous``, ``NeipelHeterogeneousSIR`` to ``continuous/__init__.py`` ``__all__``

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

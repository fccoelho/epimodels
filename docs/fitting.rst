Model Fitting
=============

Epimodels provides a comprehensive parameter inference framework through the ``epimodels.fitting`` module. It allows you to fit model parameters to observed epidemiological data using a variety of loss functions and optimization algorithms.

Quick Start
------------

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
    print(result.best_params)  # {'beta': ..., 'gamma': ...}
    print(result.loss)         # Final loss value
    result.fitted_model.plot_traces()

Dataset Management
------------------

The :class:`~epimodels.fitting.Dataset` class manages observed data:

.. code-block:: python

    from epimodels.fitting import Dataset

    dataset = Dataset()
    dataset.add_series("I", times=[0, 1, 2, 3], values=[1, 5, 20, 50])

    # From pandas DataFrame
    dataset.from_dataframe(df, time_col="date", value_col="cases", state_var="I")

    # With uncertainty
    dataset.add_series("I", times=[0, 1, 2], values=[1, 5, 20], uncertainty=[0.5, 2, 5])

Loss Functions
--------------

Choose an appropriate loss function for your data type:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Loss Function
     - Best For
   * - :class:`~epimodels.fitting.SumOfSquaredErrors`
     - General purpose, default choice
   * - :class:`~epimodels.fitting.WeightedSSE`
     - When some state variables are more important
   * - :class:`~epimodels.fitting.PoissonLikelihood`
     - Count data (case reports)
   * - :class:`~epimodels.fitting.NegativeBinomialLikelihood`
     - Overdispersed count data
   * - :class:`~epimodels.fitting.NormalLikelihood`
     - Continuous data with Gaussian noise
   * - :class:`~epimodels.fitting.HuberLoss`
     - Robust to outliers
   * - :class:`~epimodels.fitting.CustomLoss`
     - User-defined objectives

Optimizers
----------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Optimizer
     - Methods
     - Notes
   * - :class:`~epimodels.fitting.ScipyOptimizer`
     - L-BFGS-B, BFGS, Nelder-Mead, Powell, differential_evolution
     - CPU, most methods available
   * - :class:`~epimodels.fitting.JAXOptimizer`
     - Adam, SGD, RMSprop
     - GPU-accelerated via JAX
   * - :class:`~epimodels.fitting.NevergradOptimizer`
     - Derivative-free algorithms
     - No gradients needed
   * - :class:`~epimodels.fitting.MultiStartOptimizer`
     - Multi-start with Latin Hypercube / Sobol sampling
     - Avoids local minima

Advanced Usage
--------------

Multi-start optimization:

.. code-block:: python

    from epimodels.fitting import ModelFitter, MultiStartOptimizer, ScipyOptimizer

    fitter = ModelFitter(
        model=SIR(),
        dataset=dataset,
        params={"beta": (0.1, 5.0), "gamma": (0.01, 1.0)},
    )
    fitter.set_optimizer(MultiStartOptimizer(
        base_optimizer=ScipyOptimizer(method="L-BFGS-B"),
        n_starts=10,
    ))
    result = fitter.fit()

Profile likelihood for confidence intervals:

.. code-block:: python

    result = fitter.fit()
    ci = fitter.profile_likelihood("beta", result.best_params)
    print(f"beta 95% CI: {ci.lower_bound:.3f} - {ci.upper_bound:.3f}")

Examples
--------

See the Jupyter notebooks for detailed examples:

* **Model_Fitting.ipynb** -- Basic fitting workflow
* **SISLogistic_fitting.ipynb** -- Fitting SISLogistic model
* **SISLogistic_fitting_real_data.ipynb** -- Fitting with real epidemiological data
* **SIRS_parameter_inference.ipynb** -- SIRS inference with bounded optimization

API Reference
-------------

.. automodule:: epimodels.fitting
   :members:
   :undoc-members:

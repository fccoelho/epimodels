Bayesian Inference
==================

The :mod:`epimodels.fitting.bayes` module provides full posterior parameter
inference using Differential Evolution Markov Chain (DE-MCMC; ter Braak,
2006). Because the sampler only requires likelihood *values* — no gradients —
it works with any simulation-based model, including models solved with scipy,
without requiring JAX-traceable code.

Quick start with the sugar API
------------------------------

Every model exposes a one-call :meth:`~epimodels.BaseModel.fit` method.
With ``method="bayes"`` it runs Bayesian inference:

.. code-block:: python

    from epimodels.continuous import SIR

    model = SIR()

    result = model.fit(
        {"I": observed_incidence},       # dict of series
        times=observation_times,
        params_to_fit={"beta": (0.1, 5.0), "gamma": (0.01, 1.0)},
        total_population=10000,
        method="bayes",
        likelihood="poisson",
        num_samples=1000,
        num_warmup=500,
        seed=42,
    )

    result.samples["beta"]      # posterior draws
    result.summary()            # mean/std/quantiles per parameter
    result.map_estimate()       # maximum a posteriori parameters
    result.credible_intervals() # 95% credible intervals
    result.trace_plot()         # trace + density per parameter

Using the full API
------------------

For more control, use :func:`~epimodels.fitting.bayes.fit_model_bayesian`
directly with a :class:`~epimodels.fitting.Dataset` and
:class:`~epimodels.fitting.ParameterSpec` objects (the same machinery as
:doc:`maximum-likelihood fitting <fitting>`):

.. code-block:: python

    from epimodels.continuous import SIR
    from epimodels.fitting import Dataset, ParameterSpec
    from epimodels.fitting.bayes import fit_model_bayesian

    model = SIR()
    dataset = Dataset(model).register(
        name="cases",
        values=observed_I,
        times=times,
        state_variable="I",
    )

    result = fit_model_bayesian(
        model,
        dataset,
        parameters_to_fit=[
            ParameterSpec("beta", bounds=(0.1, 5.0)),
            ParameterSpec("gamma", bounds=(0.01, 1.0)),
        ],
        total_population=10000,
        likelihood="normal",
        sigma=10.0,
        num_samples=1000,
        num_warmup=500,
        seed=0,
        attach_fitted_model=True,   # simulate at the MAP and attach result
    )

    result.model.traces["I"]        # trajectory at the MAP estimate

Observation models
------------------

=========================  ==================================================
``likelihood``             Use for
=========================  ==================================================
``"normal"``               Continuous data with Gaussian noise; set ``sigma``
``"poisson"``              Count data (reports, cases)
``"negative_binomial"``    Overdispersed counts; ``sigma`` sets dispersion
=========================  ==================================================

ArviZ integration
-----------------

If `ArviZ <https://arviz-devs.github.io/arviz/>`_ is installed, the
posterior can be exported for diagnostics and visualization:

.. code-block:: python

    idata = result.to_inference_data()
    idata.posterior["beta"]   # (draws, chains)

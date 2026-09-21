Model Fitting
=============

Epimodels provides a comprehensive parameter inference framework through the ``epimodels.fitting`` module. It allows you to fit model parameters to observed epidemiological data using a variety of loss functions and optimization algorithms. For guidance on choosing likelihood functions in epidemic modelling (e.g., Poisson vs. negative binomial for count data), see King et al. (2015); for confidence intervals via profile likelihood, see Raue et al. (2009).

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

Choose an appropriate loss function for your data type. Maximum-likelihood
losses (Poisson, negative binomial, normal) correspond to the standard
observation models used in epidemic inference (King et al., 2015);
``HuberLoss`` implements the robust regression criterion of
Huber (1964), see also `Huber loss (Wikipedia)
<https://en.wikipedia.org/wiki/Huber_loss>`_.
The sum of squared errors corresponds to least-squares fitting under
independent Gaussian errors (`Least squares (Wikipedia)
<https://en.wikipedia.org/wiki/Least_squares>`_).

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
   * - :class:`~epimodels.fitting.LogLikelihood`
     - Generic log-likelihood (user-supplied function)

Optimizers
----------

The optimizers wrap well-established numerical optimization algorithms:
L-BFGS-B is the bound-constrained limited-memory quasi-Newton method of
Byrd et al. (1995); Nelder–Mead is the simplex method of Nelder and
Mead (1965); ``differential_evolution`` implements the global optimizer
of Storn and Price (1997); the JAX ``Adam`` optimizer follows
Kingma and Ba (2015); multi-start initialization uses Latin hypercube
sampling (McKay et al., 1979) and Sobol' sequences (Sobol', 1967).

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

Fitting Initial Conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also fit initial conditions alongside parameters using
:class:`~epimodels.fitting.InitialConditionSpec`:

.. code-block:: python

    from epimodels.fitting import ModelFitter, InitialConditionSpec

    fitter = ModelFitter(
        model=SIR(),
        dataset=dataset,
        params={"beta": (0.1, 5.0), "gamma": (0.01, 1.0)},
        initial_conditions=[
            InitialConditionSpec("S", bounds=(500, 1000)),
            InitialConditionSpec("I", bounds=(1, 100)),
        ],
        total_population=1000,
    )
    result = fitter.fit()
    print(result.best_initial_conditions)

Profile likelihood for confidence intervals:

.. code-block:: python

    result = fitter.fit()
    ci = fitter.profile_likelihood("beta", result.best_params)
    print(f"beta 95% CI: {ci.lower_bound:.3f} - {ci.upper_bound:.3f}")

This follows the profile-likelihood approach of Raue et al. (2009):
the parameter of interest is fixed on a grid, remaining parameters are
re-optimized, and likelihood-ratio thresholds define the confidence
interval (see also `Likelihood ratio (Wikipedia)
<https://en.wikipedia.org/wiki/Likelihood-ratio_test>`_).

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

References
----------

* King, A. A., Ionides, E. L., Bretó, C. M., Ellner, S. P., & Kendall, B. E.
  (2015). Avoidable errors in the modelling of outbreaks of emerging
  pathogens, with special reference to Ebola. *Proceedings of the Royal
  Society B*, 282(1806), 20150347.
  doi:`10.1098/rspb.2015.0347 <https://doi.org/10.1098/rspb.2015.0347>`_
* Huber, P. J. (1964). Robust estimation of a location parameter.
  *The Annals of Mathematical Statistics*, 35(1), 73–101.
  doi:`10.1214/aoms/1177703732 <https://doi.org/10.1214/aoms/1177703732>`_
* Byrd, R. H., Lu, P., Nocedal, J., & Zhu, C. (1995). A limited memory
  algorithm for bound constrained optimization. *SIAM Journal on Scientific
  Computing*, 16(5), 1190–1208.
  doi:`10.1137/0916069 <https://doi.org/10.1137/0916069>`_
* Nelder, J. A., & Mead, R. (1965). A simplex method for function
  minimization. *The Computer Journal*, 7(4), 308–313.
  doi:`10.1093/comjnl/7.4.308 <https://doi.org/10.1093/comjnl/7.4.308>`_
* Storn, R., & Price, K. (1997). Differential evolution – a simple and
  efficient heuristic for global optimization over continuous spaces.
  *Journal of Global Optimization*, 11, 341–359.
  doi:`10.1023/A:1008202821328 <https://doi.org/10.1023/A:1008202821328>`_
* Kingma, D. P., & Ba, J. (2015). Adam: a method for stochastic
  optimization. *ICLR 2015* (arXiv:`1412.6980 <https://arxiv.org/abs/1412.6980>`_)
* McKay, M. D., Beckman, R. J., & Conover, W. J. (1979). A comparison of
  three methods for selecting values of input variables in the analysis of
  output from a computer code. *Technometrics*, 21(2), 239–245.
  doi:`10.1080/00401706.1979.10489755 <https://doi.org/10.1080/00401706.1979.10489755>`_
* Sobol', I. M. (1967). On the distribution of points in a cube and the
  approximate evaluation of integrals. *USSR Computational Mathematics and
  Mathematical Physics*, 7(4), 86–112.
  doi:`10.1016/0041-5553(67)90144-9 <https://doi.org/10.1016/0041-5553(67)90144-9>`_
* Raue, A., Kreutz, C., Maiwald, T., Bachmann, J., Schilling, M.,
  Klingmüller, U., & Timmer, J. (2009). Structural and practical
  identifiability analysis of partially observed dynamical models by
  exploiting the profile likelihood. *Bioinformatics*, 25(15), 1923–1929.
  doi:`10.1093/bioinformatics/btp358 <https://doi.org/10.1093/bioinformatics/btp358>`_

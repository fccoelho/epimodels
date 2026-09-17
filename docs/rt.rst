Rt Estimation
=============

The :mod:`epimodels.rt` module estimates the time-varying reproduction
number :math:`R_t` directly from incidence data, without assuming an
underlying model. It implements the sliding-window approach of Cori et al.
(2013) — the method behind EpiEstim — with a gamma-distributed serial
interval and a gamma prior/posterior on :math:`R_t`.

Basic usage
-----------

.. code-block:: python

    from epimodels.rt import estimate_rt

    # incidence: daily case counts (regular spacing)
    result = estimate_rt(
        incidence,
        window=7,        # sliding window length (time steps)
        si_mean=4.0,     # serial interval mean (time steps)
        si_sd=2.0,       # serial interval standard deviation
    )

    result.rt_mean       # posterior mean Rt at each window end
    result.rt_low        # 95% credible interval
    result.rt_high
    result.times         # window end times

    result.plot()        # Rt curve with credible band and Rt = 1 line

Interpretation
--------------

* :math:`R_t > 1`: the epidemic is growing at that time;
* :math:`R_t < 1`: it is declining;
* When a window contains (almost) no cases, the posterior falls back to
  the prior (as in EpiEstim) — treat those windows as uninformative.

Tuning
------

============================  =============================================
Parameter                     Effect
============================  =============================================
``window``                    Larger windows smooth Rt but lag changes
``si_mean`` / ``si_sd``       Serial interval assumptions (in time steps)
``prior_mean``/``prior_sd``   Gamma prior on Rt (default mean 5, sd 5)
``level``                     Credible interval width (default 0.95)
``times``                     Real timestamps for reporting (e.g. dates)
============================  =============================================

Validation against a simulated epidemic
---------------------------------------

.. code-block:: python

    import numpy as np
    from epimodels.continuous import SIR
    from epimodels.rt import estimate_rt

    model = SIR()
    t_eval = np.arange(0, 60)
    model([989, 10, 0], [0, 60], 1000, {"beta": 0.9, "gamma": 0.3}, t_eval=t_eval)

    incidence = np.maximum(-np.diff(model.traces["S"]), 0)  # daily new cases
    result = estimate_rt(incidence, window=7, si_mean=3.0, si_sd=1.5)

    # Early Rt should be near R0 = beta/gamma = 3, declining below 1 after
    # the susceptible pool is depleted.
    result.plot()

Reference:
    Cori, A. et al. (2013). A new framework and software to estimate
    time-varying reproduction numbers during epidemics. *American Journal
    of Epidemiology*, 178(9), 1505-1512.

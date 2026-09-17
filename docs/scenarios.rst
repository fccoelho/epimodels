Scenario Analysis: Interventions and Ensembles
==============================================

Two related tools help answer "what if" questions:

* :mod:`epimodels.interventions` — time-bounded parameter changes
  (lockdowns, vaccination rollouts, behaviour change) with side-by-side
  scenario comparison;
* :mod:`epimodels.ensembles` — many simulations under parameter uncertainty,
  summarized with quantile bands.

Intervention scenarios
----------------------

An :class:`~epimodels.interventions.Intervention` modifies one parameter
over a time window, either multiplicatively (``factor``) or absolutely
(``new_value``). A :class:`~epimodels.interventions.Scenario` bundles a
model with parameters, initial conditions and interventions:

.. code-block:: python

    from epimodels.continuous import SIR
    from epimodels.interventions import Intervention, Scenario, ScenarioComparison

    model = SIR()

    baseline = Scenario(
        "baseline", model,
        params={"beta": 2.0, "gamma": 0.5},
        initial_conditions=[999, 1, 0],
        trange=[0, 100], totpop=1000,
    )

    lockdown = Scenario(
        "lockdown", model,
        params={"beta": 2.0, "gamma": 0.5},
        initial_conditions=[999, 1, 0],
        trange=[0, 100], totpop=1000,
        interventions=[
            # 60% contact reduction between days 10 and 40
            Intervention("beta", start=10, end=40, factor=0.4),
        ],
    )

    cmp = ScenarioComparison(baseline, [lockdown]).run()

    cmp.peak("I")          # {'baseline': ..., 'lockdown': ...}
    cmp.final_size("R")    # attack rate per scenario
    cmp.plot("I")          # trajectories of all scenarios on one axes

Interventions are evaluated *inside* the ODE right-hand side, so the
parameter changes smoothly follow the schedule at each integration step.
Multiple interventions can be combined per scenario, and ``new_value``
overrides a parameter absolutely instead of scaling it:

.. code-block:: python

    # Vaccination: reduce beta permanently from day 30 on, and
    # boost gamma to a fixed value during the same period
    vaccination = Scenario(
        "vaccination", model,
        params={"beta": 2.0, "gamma": 0.5},
        initial_conditions=[999, 1, 0],
        trange=[0, 100], totpop=1000,
        interventions=[
            Intervention("beta", start=30, factor=0.5),   # until the end
            Intervention("gamma", start=30, end=60, new_value=0.8),
        ],
    )

Uncertainty ensembles
---------------------

:func:`~epimodels.ensembles.simulate_ensemble` runs ``n_sims`` realizations,
each drawing parameters from a sampler you provide, and collects the
trajectories on a common time grid:

.. code-block:: python

    import numpy as np
    from epimodels.continuous import SIR
    from epimodels.ensembles import simulate_ensemble

    model = SIR()
    rng = np.random.default_rng(0)

    ensemble = simulate_ensemble(
        model,
        n_sims=200,
        param_sampler=lambda: {
            "beta": rng.uniform(1.5, 2.5),
            "gamma": 0.5,
        },
        initial_conditions=[999, 1, 0],
        trange=[0, 100],
        totpop=1000,
    )

    len(ensemble)                 # 200 realizations
    ensemble.traces["I"].shape    # (200, 201) — sims x time points

    q = ensemble.quantiles([0.025, 0.5, 0.975])
    q["I"][0.5]                   # median trajectory

    ensemble.summary()            # final size / peak statistics
    ensemble.plot_band("I")       # median with 95% band
    ensemble.plot_band("I", show_reps=True)  # overlay individual runs

Initial conditions can also be sampled by passing a callable, and
``n_jobs > 1`` parallelizes the realizations across processes.

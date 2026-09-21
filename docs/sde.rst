Stochastic Differential Equation Models
=======================================

The :mod:`epimodels.sde` module inserts demographic (environmental) noise
between the fully deterministic ODE models and the exact event-based CTMC
models. Any :class:`~epimodels.continuous.ContinuousModel` can be wrapped as
a Langevin-type stochastic differential equation,

.. math::

    dy = f(t, y)\,dt + D(t, y)\,dW,

where the drift :math:`f` is the model's deterministic right-hand side and
the diffusion defaults to the demographic square-root approximation
:math:`D_{ii} = \sqrt{\sigma\,|f_i(t, y)|}`. This Langevin approximation
of the underlying jump process is justified in the regime where many
individual transitions occur per time step (Gillespie, 2000; see also
Allen, 2017 for SDE epidemic models specifically). Integration uses
diffrax/JAX (Euler–Maruyama; Kloeden & Platen, 1992), so installation of
the ``jax`` extra is required:

.. code-block:: bash

    pip install epimodels[jax]

Basic usage
-----------

.. code-block:: python

    from epimodels.continuous import SIR
    from epimodels.sde import SDEModel

    sde = SDEModel(SIR())

    sde(
        [990, 10, 0], [0, 100], 1000,
        {"beta": 0.5, "gamma": 0.25},
        n_sims=50,          # independent trajectories
        seed=42,            # reproducibility
    )

    sde.get_mean()                    # ensemble mean trajectory
    sde.get_quantiles(0.95)           # 95% band per variable
    sde.plot_traces("I")              # replicate spaghetti + mean

.. image:: _static/sde_traces.png
    :align: center
    :alt: 50 stochastic SIR trajectories with their mean

With ``n_sims=1`` the traces are 1D arrays; with ``n_sims > 1`` they have
shape ``(n_sims, n_points)``, mirroring the CTMC interface.

Noise magnitude
---------------

``noise_scale`` multiplies the default diffusion (1.0 corresponds to
demographic noise magnitude). Setting it to 0 reproduces the deterministic
ODE exactly — useful as a sanity check:

.. code-block:: python

    deterministic = SDEModel(SIR(), noise_scale=0.0)

Custom diffusion
----------------

For full control, pass a diffusion matrix function
``(t, y, params) -> (n_vars, n_noise)``:

.. code-block:: python

    def my_diffusion(t, y, params):
        # e.g. noise only in the infectious compartment
        import jax.numpy as jnp
        return jnp.diag(jnp.array([0.0, 0.5, 0.0]))

    sde = SDEModel(SIR(), diffusion=my_diffusion)

Limitations
-----------

Models whose ``_model`` uses numpy-specific functions (e.g. ``np.tanh``),
interpolation callables or internal history state (SIRSEI-family,
``SIRSNonAutonomous`` with callables) cannot run under JAX tracing. The
classic SIR/SIS/SIRS/SEIR-family models work as-is.

Numerical caveat: the Euler–Maruyama integrator does not preserve
non-negativity. When a compartment is very small (e.g. a handful of
initial infectives) the √-demographic noise can push it negative, in the
worst case destabilizing the integration. Seed simulations with a
sufficiently large infected population (10+), and keep ``dt`` moderate.

References
----------

* Gillespie, D. T. (2000). The chemical Langevin equation.
  *The Journal of Chemical Physics*, 113(1), 297–306.
  doi:`10.1063/1.481811 <https://doi.org/10.1063/1.481811>`_
* Kloeden, P. E., & Platen, E. (1992). *Numerical Solution of Stochastic
  Differential Equations.* Springer.
  doi:`10.1007/978-3-662-12616-5 <https://doi.org/10.1007/978-3-662-12616-5>`_
* Allen, L. J. S. (2017). A primer on stochastic epidemic models:
  formulation, numerical simulation, and analysis.
  *Infectious Disease Modelling*, 2(2), 128–142.
  doi:`10.1016/j.idm.2017.03.001 <https://doi.org/10.1016/j.idm.2017.03.001>`_

See also `Stochastic differential equation (Wikipedia)
<https://en.wikipedia.org/wiki/Stochastic_differential_equation>`_ and
`Euler–Maruyama method (Wikipedia)
<https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method>`_.

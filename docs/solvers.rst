Solvers
=======

Epimodels provides solver abstractions for both deterministic ODE models and stochastic CTMC models.

ODE Solvers
-----------

ODE solvers support both **scipy** (CPU-only) and **diffrax** (JAX-accelerated with GPU support) backends.

Available Solvers
-----------------

ScipySolver
~~~~~~~~~~~

Wrapper around ``scipy.integrate.solve_ivp`` with access to all scipy integration methods.

**Methods:**

- ``RK45`` (default): Explicit Runge-Kutta of order 5(4) (Dormand & Prince, 1980)
- ``RK23``: Explicit Runge-Kutta of order 3(2)
- ``DOP853``: Explicit Runge-Kutta of order 8 (Hairer, Nørsett & Wanner, 1993)
- ``Radau``: Implicit Runge-Kutta of the Radau IIA family (Hairer & Wanner, 1996)
- ``BDF``: Implicit multi-step variable-order method (Hairer & Wanner, 1996)
- ``LSODA``: Adams/BDF method with automatic stiffness detection (Petzold, 1983)

**Example:**

.. code-block:: python

    from epimodels.continuous import SIR
    from epimodels.solvers import ScipySolver

    # Create solver with specific method
    solver = ScipySolver(method='LSODA')
    
    model = SIR()
    model([999, 1, 0], [0, 100], 1000, 
          {'beta': 0.3, 'gamma': 0.1}, 
          solver=solver)

DiffraxSolver
~~~~~~~~~~~~~

JAX-accelerated solver with GPU support. Requires ``diffrax`` and ``jax`` to be installed.

**Solvers:**

- ``Tsit5`` (default): 5th order Tsitouras method (Tsitouras, 2011)
- ``Dopri5``: 5th order Dormand-Prince method (Dormand & Prince, 1980)
- ``Dopri8``: 8th order Dormand-Prince method (Hairer, Nørsett & Wanner, 1993)
- ``Euler``: Euler method
- ``Heun``: Heun's method
- ``Midpoint``: Midpoint method
- ``Ralston``: Ralston's method

See `Runge–Kutta methods (Wikipedia)
<https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods>`_ and Hairer,
Nørsett & Wanner (1993) for background on these method families.

**Example:**

.. code-block:: python

    from epimodels.continuous import SIR
    from epimodels.solvers import DiffraxSolver

    # Create JAX-accelerated solver
    solver = DiffraxSolver(
        solver='Tsit5', 
        rtol=1e-6, 
        atol=1e-9,
        adaptive=True
    )
    
    model = SIR()
    model([999, 1, 0], [0, 100], 1000, 
          {'beta': 0.3, 'gamma': 0.1}, 
          solver=solver)

**Installation:**

.. code-block:: bash

    # CPU only
    pip install diffrax jax

    # GPU (CUDA 12)
    pip install diffrax "jax[cuda12]"


Performance Benchmarks
----------------------

Scipy Methods
~~~~~~~~~~~~~

Benchmarks run on SIR model with N=1,000,000, t=[0,365], β=0.4, γ=0.1:

+----------+------------+-----------+----------------+----------------------------------------+
| Method   | Time (ms)  | Accuracy* | Stiff Handling | Notes                                  |
+==========+============+===========+================+========================================+
| LSODA    | 2.4        | Good      | Excellent      | Auto stiffness detection, fastest      |
+----------+------------+-----------+----------------+----------------------------------------+
| RK23     | 6.5        | Good      | Poor           | Fastest explicit method                |
+----------+------------+-----------+----------------+----------------------------------------+
| DOP853   | 4.9        | Excellent | Poor           | Highest accuracy (8th order)           |
+----------+------------+-----------+----------------+----------------------------------------+
| RK45     | 48.3       | Good      | Poor           | Default, robust                        |
+----------+------------+-----------+----------------+----------------------------------------+
| Radau    | 23.5       | Excellent | Excellent      | Implicit, for stiff systems            |
+----------+------------+-----------+----------------+----------------------------------------+
| BDF      | 31.5       | Good      | Excellent      | Implicit multi-step                    |
+----------+------------+-----------+----------------+----------------------------------------+

\*Accuracy measured as deviation from DOP853 reference solution.

Diffrax Methods (JAX)
~~~~~~~~~~~~~~~~~~~~~

+-----------+-------------+--------------+----------------------------------------+
| Method    | CPU Time    | GPU Time*    | Notes                                  |
+===========+=============+==============+========================================+
| Tsit5     | ~2x scipy   | 10-50x faster| Recommended default                    |
+-----------+-------------+--------------+----------------------------------------+
| Dopri5    | ~2x scipy   | 10-50x faster| Classic Dormand-Prince                 |
+-----------+-------------+--------------+----------------------------------------+
| Dopri8    | Slower      | 5-20x faster | High accuracy                          |
+-----------+-------------+--------------+----------------------------------------+

\*GPU speedup observed on batch simulations (100+ concurrent models)


When to Use Each Solver
-----------------------

+----------------------+--------------------------------+----------------------------------------+
| Scenario             | Recommended Solver             | Reason                                 |
+======================+================================+========================================+
| General use          | ``ScipySolver('LSODA')``       | Fast, handles stiffness automatically  |
+----------------------+--------------------------------+----------------------------------------+
| High accuracy        | ``ScipySolver('DOP853')``      | 8th order method                       |
+----------------------+--------------------------------+----------------------------------------+
| Stiff systems        | ``ScipySolver('BDF')``         | Implicit methods                       |
|                      | or ``ScipySolver('Radau')``    |                                        |
+----------------------+--------------------------------+----------------------------------------+
| Batch simulations    | ``DiffraxSolver('Tsit5')``     | GPU parallelization                    |
+----------------------+--------------------------------+----------------------------------------+
| Parameter sweeps     | ``DiffraxSolver``              | JAX JIT compilation                    |
+----------------------+--------------------------------+----------------------------------------+
| Quick prototyping    | Default (RK45)                 | Robust and reliable                    |
+----------------------+--------------------------------+----------------------------------------+


CTMC Solvers (Stochastic)
-------------------------

Stochastic solvers generate exact or approximate trajectories of Continuous-Time
Markov Chain models using the Gillespie algorithm and related methods.

GillespieSolver
~~~~~~~~~~~~~~~

Exact stochastic simulation using the Gillespie Direct Method (SSA)
(Gillespie, 1976; 1977). Each step:

1. Compute propensities for all events
2. Draw time to next event from an exponential distribution
3. Select which event fires (weighted random selection)
4. Update the state

**Example:**

.. code-block:: python

    from epimodels.stochastic.CTMC import SIR, GillespieSolver

    # Create solver (optional — default when calling a CTMC model)
    solver = GillespieSolver()

    model = SIR()
    model([990, 10, 0], [0, 100], 1000,
          {'beta': 0.3, 'gamma': 0.1},
          reps=100, seed=42, solver=solver)

When to Use CTMC Solvers
~~~~~~~~~~~~~~~~~~~~~~~~~

+----------------------------------+--------------------------------------------+
| Scenario                         | Notes                                      |
+==================================+============================================+
| Small populations (<10,000)      | Stochastic effects are significant         |
+----------------------------------+--------------------------------------------+
| Extinction risk analysis         | Only stochastic models can capture this    |
+----------------------------------+--------------------------------------------+
| Confidence intervals             | Run multiple replicates                    |
+----------------------------------+--------------------------------------------+
| Large populations (>100,000)     | ODE models may be sufficient; CTMC is slow |
+----------------------------------+--------------------------------------------+


ODE Solver API Reference
------------------------

.. autoclass:: epimodels.solvers.SolverBase
   :members:
   :undoc-members:

.. autoclass:: epimodels.solvers.ScipySolver
   :members:
   :undoc-members:

.. autoclass:: epimodels.solvers.DiffraxSolver
   :members:
   :undoc-members:

.. autoclass:: epimodels.solvers.SolverResult
   :members:
   :undoc-members:

.. autofunction:: epimodels.solvers.get_default_solver

CTMC Solver API Reference
-------------------------

.. autoclass:: epimodels.stochastic.CTMC.solvers.CTMCSolverBase
   :members:

.. autoclass:: epimodels.stochastic.CTMC.solvers.GillespieSolver
   :members:

.. autoclass:: epimodels.stochastic.CTMC.solvers.CTMCTrajectory
   :members:

References
----------

* Dormand, J. R., & Prince, P. J. (1980). A family of embedded Runge–Kutta
  formulae. *Journal of Computational and Applied Mathematics*, 6(1), 19–26.
  doi:`10.1016/0771-050X(80)90013-3 <https://doi.org/10.1016/0771-050X(80)90013-3>`_
* Tsitouras, C. (2011). Optimized Runge–Kutta methods for the solution of
  the orbital problem. *Applied Numerical Mathematics*, 61(2), 158–169.
  doi:`10.1016/j.apnum.2010.09.005 <https://doi.org/10.1016/j.apnum.2010.09.005>`_
* Hairer, E., Nørsett, S. P., & Wanner, G. (1993). *Solving Ordinary
  Differential Equations I: Nonstiff Problems* (2nd ed.). Springer.
* Hairer, E., & Wanner, G. (1996). *Solving Ordinary Differential Equations
  II: Stiff and Differential-Algebraic Problems* (2nd ed.). Springer.
* Petzold, L. (1983). Automatic selection of methods for solving stiff and
  nonstiff systems of ordinary differential equations. *SIAM Journal on
  Scientific and Statistical Computing*, 4(1), 136–159.
  doi:`10.1137/0904010 <https://doi.org/10.1137/0904010>`_
* Gillespie, D. T. (1976). A general method for numerically simulating the
  stochastic time evolution of coupled chemical reactions.
  *Journal of Computational Physics*, 22(4), 403–434.
  doi:`10.1016/0021-9991(76)90041-3 <https://doi.org/10.1016/0021-9991(76)90041-3>`_
* Gillespie, D. T. (1977). Exact stochastic simulation of coupled chemical
  reactions. *The Journal of Physical Chemistry*, 81(25), 2340–2361.
  doi:`10.1021/j100540a008 <https://doi.org/10.1021/j100540a008>`_

Solvers
=======

Epimodels provides a unified interface for ODE solvers, supporting both **scipy** (CPU-only) and **diffrax** (JAX-accelerated with GPU support) backends.

Available Solvers
-----------------

ScipySolver
~~~~~~~~~~~

Wrapper around ``scipy.integrate.solve_ivp`` with access to all scipy integration methods.

**Methods:**

- ``RK45`` (default): Explicit Runge-Kutta of order 5(4)
- ``RK23``: Explicit Runge-Kutta of order 3(2)
- ``DOP853``: Explicit Runge-Kutta of order 8
- ``Radau``: Implicit Runge-Kutta of the Radau IIA family
- ``BDF``: Implicit multi-step variable-order method
- ``LSODA``: Adams/BDF method with automatic stiffness detection

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

- ``Tsit5`` (default): 5th order Tsitouras method
- ``Dopri5``: 5th order Dormand-Prince method
- ``Dopri8``: 8th order Dormand-Prince method
- ``Euler``: Euler method
- ``Heun``: Heun's method
- ``Midpoint``: Midpoint method
- ``Ralston``: Ralston's method

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

*Accuracy measured as deviation from DOP853 reference solution.

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

*GPU speedup observed on batch simulations (100+ concurrent models)


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


Solver API Reference
--------------------

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

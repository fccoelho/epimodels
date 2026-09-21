Phase Space Analysis
====================

Epimodels provides tools for analyzing epidemic model dynamics using phase
space reconstruction techniques from nonlinear time series analysis. These
methods are useful for understanding the underlying attractor structure of
disease dynamics, detecting chaos, and reconstructing system behavior from
partial observations.

Reconstruction is based on **Takens' embedding theorem** (Takens, 1981),
which states that a scalar time series carries sufficient information to
reconstruct the full phase space of a deterministic dynamical system
(see also `Takens' theorem (Wikipedia)
<https://en.wikipedia.org/wiki/Takens%27s_theorem>`_ and Kantz &
Schreiber (2004) for a comprehensive treatment).

.. contents::
   :local:
   :depth: 1

Quick Start
-----------

.. code-block:: python

    from epimodels.continuous import SIR
    from epimodels.tools.phase import find_optimal_embedding, phase_portrait

    model = SIR()
    model([1000, 1, 0], [0, 100], 1001, {'beta': 0.3, 'gamma': 0.1})

    # Find optimal embedding parameters automatically
    params = find_optimal_embedding(model.traces['I'])
    print(f"Optimal tau={params['tau']}, dim={params['dim']}")

    # Plot a phase portrait
    phase_portrait(model.traces['S'], model.traces['I'])

Time Delay Embedding
--------------------

:class:`~epimodels.tools.phase.TimeDelayEmbedding` implements Takens'
embedding theorem for reconstructing the phase space from a scalar time
series.

.. code-block:: python

    from epimodels.tools.phase import TimeDelayEmbedding
    import numpy as np

    t = np.linspace(0, 10, 1000)
    data = np.sin(t) + 0.1 * np.random.randn(1000)

    embedding = TimeDelayEmbedding(data, tau=10, dim=3)
    embedded = embedding.embed()  # shape: (n_samples, 3)

Choosing optimal values for *tau* (time delay) and *dim* (embedding
dimension) is critical for meaningful reconstruction. The module provides
two automated methods.

Mutual Information
~~~~~~~~~~~~~~~~~~

The :meth:`~epimodels.tools.phase.TimeDelayEmbedding.mutual_information`
method calculates average mutual information for different time delays.
The **first local minimum** of the mutual information function is a good
choice for the embedding delay *tau*, following Fraser and Swinney (1986).

.. code-block:: python

    embedding = TimeDelayEmbedding(data)
    tau_opt, mi_values = embedding.mutual_information(tau_max=50)

    # Visualize the results
    embedding.plot_mutual_information(tau_max=50)

Cao's Method
~~~~~~~~~~~~

:meth:`~epimodels.tools.phase.TimeDelayEmbedding.cao_embedding_dimension`
estimates the minimum embedding dimension using the E1 statistic
(Cao, 1997). When
E1 saturates (stops increasing beyond a threshold), the corresponding
dimension is sufficient.

.. code-block:: python

    embedding = TimeDelayEmbedding(data, tau=tau_opt)
    dim_opt, e1_values = embedding.cao_embedding_dimension(dim_max=10)

    # Visualize the results
    embedding.plot_embedding_dimension(dim_max=10)

Automatic Parameter Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The convenience function
:func:`~epimodels.tools.phase.find_optimal_embedding` combines both
methods to find optimal *tau* and *dim* in one call:

.. code-block:: python

    from epimodels.tools.phase import find_optimal_embedding

    params = find_optimal_embedding(data, tau_max=50, dim_max=10)
    print(params['tau'])        # Optimal time delay
    print(params['dim'])        # Optimal embedding dimension
    print(params['mi_values'])  # Mutual information values
    print(params['e1_values'])  # E1 statistic values

Phase Portraits
---------------

:func:`~epimodels.tools.phase.phase_portrait` creates a 2D phase
portrait from two time series, typically from an epidemic model's state
variables. The trajectory can be colored by time to show temporal
evolution.

.. code-block:: python

    from epimodels.tools.phase import phase_portrait
    from epimodels.continuous import SIR

    model = SIR()
    model([1000, 1, 0], [0, 100], 1001, {'beta': 0.3, 'gamma': 0.1})

    # S vs I phase portrait colored by time
    phase_portrait(model.traces['S'], model.traces['I'])

    # Custom styling
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    phase_portrait(
        model.traces['S'], model.traces['I'],
        ax=ax, color_by_time=False, color='darkblue', linewidth=1.5
    )

API Reference
-------------

.. autoclass:: epimodels.tools.phase.TimeDelayEmbedding
   :members:
   :undoc-members:

.. autofunction:: epimodels.tools.phase.phase_portrait

.. autofunction:: epimodels.tools.phase.find_optimal_embedding

References
----------

* Takens, F. (1981). Detecting strange attractors in turbulence.
  In *Dynamical Systems and Turbulence*, Lecture Notes in Mathematics 898,
  pp. 366–381. Springer.
  doi:`10.1007/BFb0091924 <https://doi.org/10.1007/BFb0091924>`_
* Fraser, A. M., & Swinney, H. L. (1986). Independent coordinates for
  strange attractors from mutual information. *Physical Review A*, 33(2),
  1134–1140. doi:`10.1103/PhysRevA.33.1134 <https://doi.org/10.1103/PhysRevA.33.1134>`_
* Cao, L. (1997). Practical method for determining the minimum embedding
  dimension of a scalar time series. *Physica D: Nonlinear Phenomena*,
  110(1–2), 43–50.
  doi:`10.1016/S0167-2789(97)00118-8 <https://doi.org/10.1016/S0167-2789(97)00118-8>`_
* Kantz, H., & Schreiber, T. (2004). *Nonlinear Time Series Analysis*
  (2nd ed.). Cambridge University Press.

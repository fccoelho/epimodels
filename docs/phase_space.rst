Phase Space Analysis
====================

Epimodels provides tools for analyzing epidemic model dynamics using phase
space reconstruction techniques from nonlinear time series analysis. These
methods are useful for understanding the underlying attractor structure of
disease dynamics, detecting chaos, and reconstructing system behavior from
partial observations.

Reconstruction is based on **Takens' embedding theorem**, which states that
a scalar time series carries sufficient information to reconstruct the
full phase space of a deterministic dynamical system.

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
choice for the embedding delay *tau*.

.. code-block:: python

    embedding = TimeDelayEmbedding(data)
    tau_opt, mi_values = embedding.mutual_information(tau_max=50)

    # Visualize the results
    embedding.plot_mutual_information(tau_max=50)

Cao's Method
~~~~~~~~~~~~

:meth:`~epimodels.tools.phase.TimeDelayEmbedding.cao_embedding_dimension`
estimates the minimum embedding dimension using the E1 statistic. When
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

=========
Epimodels
=========

**Epimodels** is a library of mathematical models for epidemiology, to be used for simulations. It contains both Deterministic and Stochastic models in continuous and discrete time. Some Stochastic models are also discrete in the state variables (birth and death processes).

.. note::

    This library is under development. Contributions are welcome.

Installation
============

.. code-block:: bash

    pip install epimodels

For pandas DataFrame support:

.. code-block:: bash

    pip install epimodels[dataframe]

Quick Start
===========

.. code-block:: python

    from epimodels.continuous import SIR

    model = SIR()
    model([1000, 1, 0], [0, 50], 1001, {'beta': 0.3, 'gamma': 0.1})
    
    print(f"R0 = {model.R0}")  # Basic reproduction number
    print(model.summary())      # Epidemic statistics
    model.plot_traces()         # Plot results

Contents
========

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   geting_started
   models

.. toctree::
   :maxdepth: 2
   :caption: Examples:

   Examples/Continuous_models
   Examples/Discrete_models
   Examples/API_features

.. toctree::
   :maxdepth: 2
   :caption: Reference:

   api/modules

.. toctree::
   :maxdepth: 1
   :caption: About:

   License <license>
   Authors <authors>
   Changelog <changelog>


API Overview
============

Model Classes
-------------

Continuous Models (ODE-based)
  - :class:`~epimodels.continuous.models.SIR` - Susceptible-Infectious-Removed
  - :class:`~epimodels.continuous.models.SIS` - Susceptible-Infectious-Susceptible
  - :class:`~epimodels.continuous.models.SIRS` - Susceptible-Infectious-Removed-Susceptible
  - :class:`~epimodels.continuous.models.SEIR` - Susceptible-Exposed-Infectious-Removed
  - :class:`~epimodels.continuous.models.SEQIAHR` - COVID-19 model with quarantine
  - :class:`~epimodels.continuous.models.Dengue4Strain` - 4-strain dengue model

Discrete Models (Difference equations)
  - :class:`~epimodels.discrete.models.SIR`
  - :class:`~epimodels.discrete.models.SIS`
  - :class:`~epimodels.discrete.models.SEIR`
  - :class:`~epimodels.discrete.models.SEIS`
  - :class:`~epimodels.discrete.models.SIRS`
  - :class:`~epimodels.discrete.models.SEQIAHR`
  - :class:`~epimodels.discrete.models.Influenza`

Common Methods
--------------

All models inherit from :class:`~epimodels.BaseModel` and share these methods:

====================================  =====================================================
Method                                Description
====================================  =====================================================
``__call__(inits, trange, N, params)``  Run the simulation
``plot_traces()``                     Plot simulation results
``to_dataframe()``                    Export to pandas DataFrame
``to_dict()``                         Get a copy of traces
``summary()``                         Get epidemic statistics
``copy()``                            Create a model copy
``reset()``                           Clear simulation results
``R0``                                Basic reproduction number (property)
====================================  =====================================================


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _toctree: http://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html
.. _reStructuredText: http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
.. _references: http://www.sphinx-doc.org/en/stable/markup/inline.html
.. _Python domain syntax: http://sphinx-doc.org/domains.html#the-python-domain
.. _Sphinx: http://www.sphinx-doc.org/
.. _Python: http://docs.python.org/
.. _Numpy: http://docs.scipy.org/doc/numpy
.. _SciPy: http://docs.scipy.org/doc/scipy/reference/
.. _matplotlib: https://matplotlib.org/contents.html#
.. _Pandas: http://pandas.pydata.org/pandas-docs/stable
.. _Scikit-Learn: http://scikit-learn.org/stable
.. _autodoc: http://www.sphinx-doc.org/en/stable/ext/autodoc.html
.. _Google style: https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings
.. _NumPy style: https://numpydoc.readthedocs.io/en/latest/format.html
.. _classical style: http://www.sphinx-doc.org/en/stable/domains.html#info-field-lists

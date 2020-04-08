Getting Started
===============

Epimodels offers a variety of deterministic models pre-implemented for Infectious disease modeling.
These models are split into `discrete` (based of difference equations) and `continuous` (based on Ordinary differential equations).

The APIs exposed for both families of models is identical to facilitate use.


Continuous models
-----------------

.. code-block:: python

   import epimodels.continuous.models as CM
   model = CM.SIR()
   model([1000, 1, 0], [0, 50], 1001, {'beta': 2, 'gamma': .1})
   model.plot_traces()

.. image:: _static/cSIR.png

Discrete models
---------------

.. code-block:: python

   import epimodels.discrete.models as DM
   model = DM.SIR()
   model(([1000, 1, 0], [0,50], 1001, {'beta': 2, 'gamma': 1}))
   model.plot_traces()

.. image:: _static/dSIR.png



Model Registry
==============

The :mod:`epimodels.registry` module provides string-based, collision-aware
lookup of models across all families. This solves the practical problem that
class names such as ``SIR`` exist simultaneously in the ``continuous``,
``discrete`` and ``stochastic`` subpackages.

Looking up models
-----------------

.. code-block:: python

    from epimodels import get_model, list_models

    # Disambiguate by family
    SIR = get_model("SIR", family="continuous")
    model = SIR()
    model([1000, 1, 0], [0, 50], 1001, {"beta": 2, "gamma": 0.1})

    # Without a family, precedence order is used:
    # continuous -> discrete -> stochastic -> network -> custom
    get_model("SIR")                       # continuous SIR
    get_model("NetworkSIR", family="network")

    # Discover what is available
    list_models()
    # {'continuous': ['Dengue4Strain', 'EbolaSEIHFRV', 'SIR', ...],
    #  'discrete': [...], 'stochastic': [...], 'network': [...]}

The registry also accepts the ``"network"`` family for
:class:`~epimodels.network.NetworkSIR`/:class:`~epimodels.network.NetworkSIS`.

Registering custom models
-------------------------

Use the :func:`~epimodels.registry.register_model` decorator to make your own
models discoverable through the same API:

.. code-block:: python

    from epimodels.registry import register_model
    from epimodels.continuous import ContinuousModel

    @register_model("MySIR", family="custom")
    class MySIR(ContinuousModel):
        ...

    from epimodels.registry import get_model
    get_model("MySIR", family="custom")  # -> MySIR

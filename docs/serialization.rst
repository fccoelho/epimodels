Saving and Loading Models
=========================

The :mod:`epimodels.io` module serializes model *specifications* — family,
class name, state variables, parameter symbols, the last-run parameter
values and (optionally) simulation traces — to JSON or YAML files.
Reconstruction goes through the :doc:`model registry <registry>`, so saved
models round-trip across sessions and machines.

Saving and loading
------------------

.. code-block:: python

    from epimodels.continuous import SIR
    from epimodels.io import save_model, load_model

    model = SIR()
    model([1000, 1, 0], [0, 50], 1001, {"beta": 2, "gamma": 0.1})

    save_model(model, "sir_run.json", include_traces=True)
    clone = load_model("sir_run.json")

    clone.param_values["beta"]   # 2
    clone.traces["I"]            # restored as numpy arrays

YAML requires the optional ``yaml`` extra (``pip install epimodels[yaml]``):

.. code-block:: python

    save_model(model, "sir_run.yaml")

Serialization also works for discrete, CTMC and network models:

.. code-block:: python

    from epimodels.network import NetworkSIR
    import networkx as nx

    net = NetworkSIR(nx.barabasi_albert_graph(500, 3, seed=0))
    save_model(net, "network.json")   # structure spec (graph not serialized)

Dict-based API
--------------

For programmatic use (e.g. storing specs in a database), the same
functionality is available as dictionaries:

.. code-block:: python

    from epimodels.io import model_to_dict, model_from_dict

    spec = model_to_dict(model, include_traces=True)
    clone = model_from_dict(spec)

Note:
    The graph object of network models is not serialized — only the model
    class and parameters are. Rebuild the graph (or pass it again) when
    re-running network simulations.

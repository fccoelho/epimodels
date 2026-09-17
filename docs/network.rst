Network Models
==============

The :mod:`epimodels.network` module simulates epidemics directly on contact
networks with an event-driven (Gillespie) algorithm: infection fires along
each susceptible–infectious edge at rate ``beta`` and each infected node
recovers at rate ``gamma``.

Graphs can be `networkx <https://networkx.org>`_ graphs (install the
``network`` extra: ``pip install epimodels[network]``), plain adjacency
dictionaries ``{node: [neighbors]}`` or symmetric adjacency matrices.

SIR on a network
----------------

.. code-block:: python

    import networkx as nx
    from epimodels.network import NetworkSIR

    G = nx.barabasi_albert_graph(1000, 3, seed=0)  # scale-free contact network

    model = NetworkSIR(G)
    model(
        5,                       # 5 initially infected nodes (or a list of nodes)
        [0, 50],                 # simulation time range
        {"beta": 0.3, "gamma": 0.1},
        n_sims=20,               # stochastic replicates
        seed=0,
    )

    model.traces["I"].shape      # (20, 101) — replicates x time grid
    model.final_size()           # attack rate per replicate
    model.get_mean()             # mean trajectories
    model.get_quantiles(0.95)    # 95% band
    model.plot_traces("I")       # spaghetti plot + mean

SIS on a network
----------------

:class:`~epimodels.network.NetworkSIS` sends recovered nodes back to the
susceptible pool, so infection can become endemic:

.. code-block:: python

    from epimodels.network import NetworkSIS

    G_small = nx.barabasi_albert_graph(300, 3, seed=0)
    model = NetworkSIS(G_small)
    model(5, [0, 50], {"beta": 0.6, "gamma": 0.1}, n_sims=10, seed=0)

    model.traces["I"][:, -1].mean()   # mean prevalence at tmax

Note:
    The event-driven sampler picks the next infector proportionally to its
    susceptible-neighbor count, which is O(infected) per event. Long-running
    SIS simulations on large graphs (persistent epidemics generate many
    events) are therefore slow; prefer smaller graphs or shorter horizons.

Targeting specific nodes
------------------------

Instead of a count, pass the identifiers of the initially infected nodes —
useful for studying super-spreading from hubs:

.. code-block:: python

    hub = max(G.nodes(), key=G.degree)
    model([hub], [0, 50], {"beta": 0.3, "gamma": 0.1}, n_sims=5, seed=1)

Working without networkx
------------------------

Adjacency dictionaries and matrices avoid the networkx dependency:

.. code-block:: python

    from epimodels.network import NetworkSIR

    adjacency = {0: [1, 2], 1: [0], 2: [0]}
    model = NetworkSIR(adjacency)
    model(1, [0, 20], {"beta": 0.5, "gamma": 0.2}, n_sims=3, seed=0)

Note:
    Network models follow the library conventions (``parameters``,
    ``traces``, ``summary()``) but take the graph at construction time and
    the number of initially infected nodes in place of ``inits``; there is
    no ``totpop`` argument since the population is the set of nodes.

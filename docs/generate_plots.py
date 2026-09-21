"""
Generate the figures used by the Sphinx documentation.

Run from the repository root:

    python docs/generate_plots.py

Every figure is produced by executing (essentially verbatim) the example
code shown in the corresponding documentation page, so the images always
match what readers will reproduce.
"""

import pathlib
import sys

# Make the repository root importable regardless of invocation directory.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STATIC_DIR = pathlib.Path(__file__).parent / "_static"
STATIC_DIR.mkdir(exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def save(fig_or_ax, name: str) -> None:
    if isinstance(fig_or_ax, np.ndarray):  # array of axes from plt.subplots
        fig_or_ax = fig_or_ax.ravel()[0].figure
    fig = getattr(fig_or_ax, "figure", fig_or_ax)
    path = STATIC_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.name}")


def scenarios_interventions() -> None:
    """docs/scenarios.rst — intervention comparison."""
    from epimodels.continuous import SIR
    from epimodels.interventions import Intervention, Scenario, ScenarioComparison

    model = SIR()
    params = {"beta": 0.4, "gamma": 0.2}  # R0 = 2
    common = dict(params=params, initial_conditions=[995, 5, 0],
                  trange=[0, 120], totpop=1000)

    baseline = Scenario("baseline", model, **common)
    lockdown = Scenario(
        "lockdown", model,
        interventions=[
            # 60% contact reduction between days 20 and 60
            Intervention("beta", start=20, end=60, factor=0.4),
        ],
        **common,
    )
    vaccination = Scenario(
        "vaccination", model,
        interventions=[
            # permanent 50% transmission reduction from day 20,
            # plus a temporary recovery boost
            Intervention("beta", start=20, factor=0.5),
            Intervention("gamma", start=20, end=90, new_value=0.4),
        ],
        **common,
    )

    cmp = ScenarioComparison(baseline, [lockdown, vaccination]).run()
    ax = cmp.plot("I", linewidth=2)
    ax.set_ylabel("Infectious individuals")
    ax.set_title("Scenario comparison")
    save(ax, "scenarios_interventions.png")


def scenarios_ensemble() -> None:
    """docs/scenarios.rst — uncertainty ensemble."""
    from epimodels.continuous import SIR
    from epimodels.ensembles import simulate_ensemble

    model = SIR()
    rng = np.random.default_rng(0)

    ensemble = simulate_ensemble(
        model,
        n_sims=200,
        param_sampler=lambda: {
            "beta": rng.uniform(0.3, 0.6),
            "gamma": 0.2,
        },
        initial_conditions=[995, 5, 0],
        trange=[0, 120],
        totpop=1000,
    )
    ax = ensemble.plot_band("I", color="tab:red")
    ax.set_ylabel("Infectious individuals")
    save(ax, "scenarios_ensemble.png")


def rt_validation() -> None:
    """docs/rt.rst — Rt estimated from a simulated SIR epidemic."""
    from epimodels.continuous import SIR
    from epimodels.rt import estimate_rt

    model = SIR()
    t_eval = np.arange(0, 80)
    model([995, 5, 0], [0, 80], 1000, {"beta": 0.4, "gamma": 0.2}, t_eval=t_eval)

    incidence = np.maximum(-np.diff(model.traces["S"]), 0)  # daily new cases
    result = estimate_rt(incidence, window=14, si_mean=5.0, si_sd=2.0)

    ax = result.plot()
    ax.set_ylim(bottom=0)
    save(ax, "rt_sir_validation.png")


def sde_traces() -> None:
    """docs/sde.rst — stochastic trajectories."""
    from epimodels.continuous import SIR
    from epimodels.sde import SDEModel

    sde = SDEModel(SIR())
    sde(
        [990, 10, 0], [0, 100], 1000,
        {"beta": 0.5, "gamma": 0.25},
        n_sims=50,
        seed=42,
    )
    ax = sde.plot_traces("I")
    ax.set_ylabel("Infectious individuals")
    save(ax, "sde_traces.png")


def network_sir() -> None:
    """docs/network.rst — SIR on a scale-free network."""
    import networkx as nx

    from epimodels.network import NetworkSIR

    G = nx.barabasi_albert_graph(1000, 3, seed=0)
    model = NetworkSIR(G)
    model(5, [0, 50], {"beta": 0.3, "gamma": 0.1}, n_sims=20, seed=0)

    ax = model.plot_traces("I")
    ax.set_ylabel("Infectious nodes")
    save(ax, "network_sir.png")


def bayes_posterior() -> None:
    """docs/bayes.rst — posterior trace and density."""
    from epimodels.continuous import SIR
    from epimodels.fitting import Dataset, ParameterSpec
    from epimodels.fitting.bayes import fit_model_bayesian

    # Synthetic "observed" data with known parameters (beta=2, gamma=0.5)
    truth = SIR()
    t_eval = np.linspace(0, 20, 21)
    truth([999, 1, 0], [0, 20], 1000, {"beta": 2.0, "gamma": 0.5}, t_eval=t_eval)

    model = SIR()
    dataset = Dataset(model).register(
        name="cases",
        values=truth.traces["I"],
        times=t_eval,
        state_variable="I",
    )

    result = fit_model_bayesian(
        model,
        dataset,
        parameters_to_fit=[
            ParameterSpec("beta", bounds=(0.1, 5.0)),
            ParameterSpec("gamma", bounds=(0.01, 1.0)),
        ],
        total_population=1000,
        likelihood="normal",
        sigma=10.0,
        num_samples=500,
        num_warmup=300,
        seed=0,
    )
    save(result.trace_plot(), "bayes_posterior.png")


if __name__ == "__main__":
    print("Generating documentation figures...")
    scenarios_interventions()
    scenarios_ensemble()
    rt_validation()
    sde_traces()
    network_sir()
    bayes_posterior()
    print("Done.")

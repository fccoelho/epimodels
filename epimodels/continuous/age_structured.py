"""
Age-structured MSEIR model with a POLYMOD-type contact matrix.

Standard compartmental models assume the population mixes homogeneously, which
is a poor description of directly transmitted childhood infections: the force
of infection depends strongly on age because children have many more contacts
than adults and mix assortatively. This module implements the age-structured
counterpart of :class:`~epimodels.continuous.models.SEIR`, with a
maternally-immune compartment, an explicit ageing chain, optional demography
and an optional routine vaccination programme.

See :mod:`epimodels.tools.contact_matrix` for the contact-matrix helpers,
including the exact detailed-balance condition that a usable matrix has to
satisfy.
"""

from __future__ import annotations

import warnings
from collections import OrderedDict

import numpy as np

from epimodels import FormulaExtractionError
from epimodels.continuous.models import ContinuousModel
from epimodels.tools.contact_matrix import (
    DEFAULT_AGE_BOUNDS,
    age_group_labels,
    check_reciprocity,
    polymod_like_contact_matrix,
)
from epimodels.tools.contact_matrix import enforce_reciprocity as _enforce_reciprocity

__all__ = ["AgeStructuredSEIR"]

COMPARTMENTS = ("M", "S", "E", "I", "R")

_COMPARTMENT_DESCRIPTIONS = {
    "M": "Maternally immune",
    "S": "Susceptible",
    "E": "Exposed",
    "I": "Infectious",
    "R": "Removed",
}


class AgeStructuredSEIR(ContinuousModel):
    r"""
    Age-structured MSEIR model with a POLYMOD-type contact matrix.

    The population is split into :math:`A` age groups. Each group carries five
    compartments — maternally immune (:math:`M`), susceptible (:math:`S`),
    exposed (:math:`E`), infectious (:math:`I`) and removed (:math:`R`) — so the
    model has :math:`5A` state variables.

    For age group :math:`i` the force of infection is

    .. math::

        \lambda_i = \beta \sum_j C_{ij} \frac{I_j}{N_j^{ref}}

    and the dynamics are

    .. math::

        \dot M_i &= B \delta_{i0} - (\omega + v_i) M_i - \mu M_i - a_i M_i
                    + a_{i-1} M_{i-1} \\
        \dot S_i &= \omega M_i - \lambda_i S_i - v_i S_i - \mu S_i - a_i S_i
                    + a_{i-1} S_{i-1} \\
        \dot E_i &= \lambda_i S_i - (\varepsilon + \mu) E_i - a_i E_i
                    + a_{i-1} E_{i-1} \\
        \dot I_i &= \varepsilon E_i - (\gamma + \mu) I_i - a_i I_i
                    + a_{i-1} I_{i-1} \\
        \dot R_i &= \gamma I_i + v_i S_i + v_i M_i - \mu R_i - a_i R_i
                    + a_{i-1} R_{i-1}

    where :math:`a_i` is the ageing rate out of group :math:`i` (the reciprocal
    of the group width, and zero for the open-ended last group), :math:`B` the
    birth rate (newborns enter the maternally-immune compartment), :math:`\mu`
    the per-capita death rate and :math:`v_i` the vaccination rate (non-zero
    only in the group selected at construction).

    Note that the ageing terms telescope: with ``births=0`` and ``mu=0`` the
    total population is exactly conserved, and in general it obeys
    ``d(total)/dt = births - mu * total``.

    The basic reproduction number is the spectral radius of the
    next-generation matrix

    .. math::

        R_0 = \rho\left( \frac{\beta}{\gamma + \mu} \,
                        \frac{N_i}{N_j} C_{ij} \right)

    which is generally **not symmetric**. It is therefore computed with
    :func:`numpy.linalg.eigvals`; using :func:`numpy.linalg.eigvalsh` (which
    silently reads one triangle) returns wrong values whenever the contact
    matrix is not reciprocal. When detailed balance holds,
    :math:`N_i C_{ij} = N_j C_{ji}`, the operator is similar to a symmetric
    matrix through :math:`N^{1/2}` and the two agree; the model checks this in
    :meth:`__call__` and warns when it does not.

    :param age_bounds: Lower age edges in years (default
        :data:`~epimodels.tools.contact_matrix.DEFAULT_AGE_BOUNDS`, 21 groups
        with fine resolution in infancy). The last group is open-ended.
    :param contact_matrix: ``(A, A)`` matrix of per-capita contact rates.
        Defaults to
        :func:`~epimodels.tools.contact_matrix.polymod_like_contact_matrix`.
    :param population: Optional reference group sizes. When omitted they are
        taken from the initial conditions at simulation time.
    :param enforce_reciprocity: If ``True`` (default) the contact matrix is
        reconciled with the reference population so that
        ``N[i]*C[i, j] == N[j]*C[j, i]`` holds exactly. Set to ``False`` to use
        a validated matrix verbatim.
    :param time_per_year: Number of model time units in one year, used to turn
        the age-group widths into ageing rates (default ``365.25``, i.e. time
        is in days).
    :param vaccination_age_index: Index of the group in which routine
        vaccination is delivered (default: the 12-23 month group on the default
        grid, i.e. index 1). Use :meth:`vaccination_rate_for_coverage` to
        translate a coverage target into a rate.

    .. note::
       :func:`~epimodels.io.save_model` stores state variables, parameter
       values and traces, but not the constructor configuration
       (``age_bounds``, ``contact_matrix``, ``population``). A model reloaded
       with :func:`~epimodels.io.load_model` therefore keeps its 21-group
       default matrix; reconstruct it with the same arguments when the
       contact structure matters.

    Example:
        >>> from epimodels.continuous.age_structured import AgeStructuredSEIR
        >>> model = AgeStructuredSEIR()
        >>> pop = [2.5e6] + [2.6e6] * 4 + [13e6] * 15 + [3e6]
        >>> beta = model.scale_beta_for_r0(12.0, gamma=0.1, population=pop)
        >>> inits = model.initial_conditions(pop, infectious=[10.0] + [0.0] * 20)
        >>> model(inits, [0, 3650], sum(pop),
        ...       {"beta": beta, "epsilon": 0.125, "gamma": 0.125})
        >>> round(model.R0, 2)
        12.0
    """

    def __init__(
        self,
        age_bounds=None,
        contact_matrix=None,
        population=None,
        *,
        enforce_reciprocity: bool = True,
        time_per_year: float = 365.25,
        vaccination_age_index: int | None = None,
    ) -> None:
        super().__init__()

        edges = np.asarray(
            DEFAULT_AGE_BOUNDS if age_bounds is None else age_bounds, dtype=float
        )
        if edges.ndim != 1 or edges.size == 0:
            raise ValueError("age_bounds must be a non-empty 1-D sequence")
        if np.any(np.diff(edges) <= 0):
            raise ValueError("age_bounds must be strictly increasing")
        if time_per_year <= 0:
            raise ValueError(f"time_per_year must be positive, got {time_per_year}")

        self.age_bounds = edges
        self.n_ages = int(edges.size)
        self.age_labels = age_group_labels(edges)
        self.time_per_year = float(time_per_year)

        if vaccination_age_index is None:
            # Default target: the 12-23 month group on the default grid, or the
            # second group when the grid is coarser.
            vaccination_age_index = min(1, self.n_ages - 1)
        if not 0 <= vaccination_age_index < self.n_ages:
            raise ValueError(
                f"vaccination_age_index must be in [0, {self.n_ages - 1}], "
                f"got {vaccination_age_index}"
            )
        self.vaccination_age_index = int(vaccination_age_index)

        if population is None:
            self.population = None
        else:
            sizes = np.asarray(population, dtype=float)
            if sizes.shape != (self.n_ages,):
                raise ValueError(
                    f"population must have length {self.n_ages}, got shape {sizes.shape}"
                )
            if np.any(sizes <= 0):
                raise ValueError("population entries must all be positive")
            self.population = sizes

        if contact_matrix is None:
            matrix = polymod_like_contact_matrix(edges, self.population)
        else:
            matrix = np.asarray(contact_matrix, dtype=float)
            if matrix.shape != (self.n_ages, self.n_ages):
                raise ValueError(
                    f"contact_matrix must have shape ({self.n_ages}, {self.n_ages}), "
                    f"got {matrix.shape}"
                )
            if np.any(matrix < 0):
                raise ValueError("contact_matrix entries must be non-negative")
        self.contact_matrix = matrix

        self.enforce_reciprocity = bool(enforce_reciprocity)

        #: Ageing rates out of each group (per model time unit); the last group
        #: is open-ended and therefore has no outflow.
        ageing = np.zeros(self.n_ages)
        widths = np.diff(edges) * self.time_per_year
        ageing[:-1] = 1.0 / widths
        self.ageing_rates = ageing

        #: Vaccination rate applied to each group (only one group is targeted).
        self.vaccination_target = np.zeros(self.n_ages)
        self.vaccination_target[self.vaccination_age_index] = 1.0

        #: Defaults injected for the optional parameters.
        self.default_parameters = {
            "omega": 2.0 / self.time_per_year,  # maternal immunity wanes in ~6 months
            "births": 0.0,
            "mu": 0.0,
            "vaccination": 0.0,
        }

        self.state_variables = OrderedDict()
        for compartment in COMPARTMENTS:
            for index, label in enumerate(self.age_labels):
                self.state_variables[f"{compartment}_{index}"] = (
                    f"{_COMPARTMENT_DESCRIPTIONS[compartment]}, {label}"
                )

        self.parameters = OrderedDict(
            {
                "beta": r"$\beta$",
                "epsilon": r"$\epsilon$",
                "gamma": r"$\gamma$",
                "omega": r"$\omega$",
                "births": r"$B$",
                "mu": r"$\mu$",
                "vaccination": r"$v$",
            }
        )
        self.model_type = "AgeStructuredSEIR"

        self.contact_operator: np.ndarray | None = None
        self._resolved_matrix: np.ndarray | None = None
        self._resolved_population: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Structure helpers
    # ------------------------------------------------------------------
    @property
    def diagram(self) -> str:
        """Mermaid diagram of the compartmental structure."""
        return r"""flowchart LR

B(Births) -->|$$B$$| M(Maternally immune)
M -->|$$\omega$$| S(Susceptible)
S -->|$$\beta C_{ij} I_j / N_j$$| E(Exposed)
E -->|$$\varepsilon$$| I(Infectious)
I -->|$$\gamma$$| R(Removed)
S -->|$$v$$| R
M -.->|ageing| S2(next age group)

classDef default fill:#eef,stroke:#446;
"""

    @property
    def dimension(self) -> int:
        """Number of state variables (``5 * n_ages``)."""
        return len(self.state_variables)

    def compartment_labels(self, compartment: str) -> list[str]:
        """
        State-variable names of one compartment across all age groups.

        :param compartment: One of ``'M'``, ``'S'``, ``'E'``, ``'I'``, ``'R'``.
        :return: List of state-variable names, ordered by age group.
        """
        if compartment not in COMPARTMENTS:
            raise ValueError(
                f"Unknown compartment {compartment!r}; expected one of {COMPARTMENTS}"
            )
        return [f"{compartment}_{i}" for i in range(self.n_ages)]

    def initial_conditions(
        self,
        population,
        infectious,
        *,
        immune_fraction=0.0,
        maternal_fraction=0.0,
    ) -> list[float]:
        """
        Build a valid initial-condition vector from a population structure.

        Ages the population into the five compartments without leaving anybody
        unaccounted for: susceptibles absorb the remainder, so that

        ``sum(inits) == sum(population)`` exactly.

        :param population: Length-``A`` group sizes.
        :param infectious: Length-``A`` counts of infectious individuals.
        :param immune_fraction: Fraction already removed (immune from previous
            exposure or vaccination). Scalar or length-``A``.
        :param maternal_fraction: Fraction maternally immune. Scalar or
            length-``A``.
        :return: Initial conditions in ``state_variables`` order.
        :raises ValueError: If any group would end up with a negative count.
        """
        sizes = np.asarray(population, dtype=float)
        if sizes.shape != (self.n_ages,):
            raise ValueError(
                f"population must have length {self.n_ages}, got shape {sizes.shape}"
            )
        infected = np.broadcast_to(np.asarray(infectious, dtype=float), sizes.shape)
        immune = np.broadcast_to(np.asarray(immune_fraction, dtype=float), sizes.shape)
        maternal = np.broadcast_to(np.asarray(maternal_fraction, dtype=float), sizes.shape)

        removed = immune * sizes
        maternally_immune = maternal * sizes
        susceptible = sizes - infected - removed - maternally_immune
        if np.any(susceptible < -1e-9):
            bad = np.flatnonzero(susceptible < -1e-9).tolist()
            raise ValueError(
                f"Age group(s) {bad} have more infectious + immune + maternal "
                "individuals than population"
            )
        susceptible = np.clip(susceptible, 0.0, None)

        blocks = {
            "M": maternally_immune,
            "S": susceptible,
            "E": np.zeros(self.n_ages),
            "I": infected,
            "R": removed,
        }
        return np.concatenate([blocks[c] for c in COMPARTMENTS]).tolist()

    def vaccination_rate_for_coverage(self, coverage: float, age_index: int | None = None) -> float:
        r"""
        Vaccination rate needed to reach a target coverage in one age group.

        Individuals pass through the targeted group at rate :math:`a_i` and are
        vaccinated at rate :math:`v`. The fraction vaccinated before ageing out
        is :math:`v / (v + a_i)`, so reaching coverage :math:`p` requires

        .. math:: v = a_i \frac{p}{1 - p}.

        :param coverage: Target fraction of the cohort vaccinated in the group,
            in ``[0, 1)``.
        :param age_index: Group index (defaults to ``vaccination_age_index``).
        :return: Vaccination rate per model time unit.
        """
        index = self.vaccination_age_index if age_index is None else int(age_index)
        if not 0 <= index < self.n_ages:
            raise ValueError(f"age_index must be in [0, {self.n_ages - 1}], got {index}")
        if not 0.0 <= coverage < 1.0:
            raise ValueError(f"coverage must be in [0, 1), got {coverage}")
        ageing = self.ageing_rates[index]
        if ageing <= 0:
            raise ValueError(
                "The open-ended last age group has no outflow, so a coverage "
                "target cannot be converted into a rate; target another group."
            )
        return float(ageing * coverage / (1.0 - coverage))

    # ------------------------------------------------------------------
    # Transmission structure
    # ------------------------------------------------------------------
    def reference_population(self, inits=None) -> np.ndarray:
        """
        Reference population structure used to normalise the force of infection.

        :param inits: Initial conditions, used when no ``population`` was given
            at construction.
        :return: Length-``A`` array of group sizes.
        """
        if self.population is not None:
            return self.population
        if inits is None:
            raise ValueError(
                "No reference population available: pass `population=` to the "
                "constructor or provide initial conditions."
            )
        states = np.asarray(inits, dtype=float)
        if states.size < self.dimension:
            raise ValueError(
                f"Expected at least {self.dimension} initial conditions, got {states.size}"
            )
        sizes = states[: self.dimension].reshape(len(COMPARTMENTS), self.n_ages).sum(axis=0)
        if np.any(sizes <= 0):
            bad = np.flatnonzero(sizes <= 0).tolist()
            raise ValueError(
                f"Age group(s) {bad} are empty in the initial conditions, so the "
                "force of infection cannot be normalised. Pass `population=` "
                "explicitly."
            )
        return sizes

    def _matrix_for(self, sizes: np.ndarray) -> np.ndarray:
        """
        Contact matrix reconciled with a specific population structure.

        :param sizes: Length-``A`` array of group sizes.
        :return: The matrix actually used in the force of infection.
        """
        if self.enforce_reciprocity:
            return _enforce_reciprocity(self.contact_matrix, sizes)
        return np.asarray(self.contact_matrix, dtype=float)

    def _contact_operator(self, population=None) -> np.ndarray:
        """
        The matrix ``N_i * C_ij / N_j`` whose spectral radius drives R0.

        With a reciprocal matrix this equals the transpose of the contact
        matrix, which is why it is only symmetric when all groups have the same
        size.
        """
        if population is not None:
            sizes = np.asarray(population, dtype=float)
        elif self._resolved_population is not None:
            sizes = self._resolved_population
        elif self.population is not None:
            sizes = self.population
        else:
            raise ValueError(
                "No reference population available: pass `population=`, run the "
                "model once, or set population= at construction."
            )
        if sizes.shape != (self.n_ages,):
            raise ValueError(f"population must have length {self.n_ages}, got {sizes.shape}")
        return self._matrix_for(sizes) * (sizes[:, None] / sizes[None, :])

    @property
    def R0(self) -> float | None:
        """
        Basic reproduction number, from the next-generation matrix.

        :return: Spectral radius of ``beta/(gamma+mu) * N_i C_ij / N_j``, or
            ``None`` if the model has not been run yet.
        """
        pv = self.param_values
        if not pv or "beta" not in pv or "gamma" not in pv:
            return None
        removal = float(pv["gamma"]) + float(pv.get("mu", 0.0))
        if removal <= 0:
            return None
        # The next-generation matrix is not symmetric in general, so its
        # eigenvalues must be obtained with `eigvals` (NOT `eigvalsh`, which
        # assumes symmetry and silently reads a single triangle).
        eigenvalues = np.linalg.eigvals(self._contact_operator())
        return float(np.max(np.abs(eigenvalues)) * float(pv["beta"]) / removal)

    def scale_beta_for_r0(
        self,
        target_r0: float,
        gamma: float,
        mu: float = 0.0,
        population=None,
    ) -> float:
        """
        Transmission rate that produces a requested :math:`R_0`.

        :param target_r0: Desired basic reproduction number.
        :param gamma: Recovery rate.
        :param mu: Per-capita death rate.
        :param population: Reference group sizes (defaults to the constructor
            value or the last simulation).
        :return: The value of ``beta`` giving ``R0 == target_r0``.
        """
        removal = gamma + mu
        if removal <= 0:
            raise ValueError("gamma + mu must be positive")
        eigenvalues = np.linalg.eigvals(self._contact_operator(population))
        spectral_radius = float(np.max(np.abs(eigenvalues)))
        if spectral_radius <= 0:
            raise ValueError("degenerate contact structure: spectral radius is zero")
        return float(target_r0 * removal / spectral_radius)

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    def __call__(self, inits, trange, totpop, params, **kwargs):
        """
        Run the model.

        Optional parameters (``omega``, ``births``, ``mu``, ``vaccination``)
        are filled from the defaults before validation. The contact matrix is
        reconciled with the reference population and the next-generation
        operator is cached so that :attr:`R0` is available right after the run.

        :param inits: Initial conditions in ``state_variables`` order.
        :param trange: ``[t0, t1]`` integration range.
        :param totpop: Total population (used for validation and ``N``).
        :param params: Model parameters; at least ``beta``, ``epsilon`` and
            ``gamma``.
        :param kwargs: Passed on to the solver.
        """
        merged = {**self.default_parameters, **dict(params)}
        sizes = self.reference_population(inits)

        self._resolved_matrix = self._matrix_for(sizes)
        if not self.enforce_reciprocity and not check_reciprocity(
            self._resolved_matrix, sizes
        ):
            warnings.warn(
                "contact_matrix does not satisfy detailed balance "
                "(N_i C_ij != N_j C_ji) for this population structure; R0 and "
                "any eigen-based analysis will not reduce to the symmetric "
                "case. Consider enforce_reciprocity=True.",
                UserWarning,
            )
        self._resolved_population = sizes
        self.contact_operator = self._contact_operator(sizes)

        super().__call__(inits, trange, totpop, merged, **kwargs)

        # Expose the age-aggregated compartments under their plain names as
        # well, so that utilities written for single-group models (scenario
        # comparison, ensembles, peak/final-size metrics, `to_dataframe`)
        # work unchanged. `state_variables` is left untouched: only the
        # per-age variables map onto the solver output.
        for compartment in COMPARTMENTS:
            keys = self.compartment_labels(compartment)
            if all(key in self.traces for key in keys):
                self.traces[compartment] = np.sum(
                    [self.traces[key] for key in keys], axis=0
                )

    def _transport(self, values: np.ndarray) -> np.ndarray:
        """Net rate of change contributed by ageing out of each group."""
        flow = self.ageing_rates[:-1] * values[:-1]
        change = np.zeros_like(values)
        change[:-1] -= flow
        change[1:] += flow
        return change

    def _model(self, t: float, y, params: dict[str, float]) -> list[float]:
        states = np.asarray(y, dtype=float).reshape(len(COMPARTMENTS), self.n_ages)
        maternal, susceptible, exposed, infectious, removed = states

        beta = params["beta"]
        epsilon = params["epsilon"]
        gamma = params["gamma"]
        omega = params["omega"]
        births = params["births"]
        mu = params["mu"]
        vaccination = params["vaccination"] * self.vaccination_target

        matrix = self._resolved_matrix if self._resolved_matrix is not None else self.contact_matrix
        sizes = self._resolved_population
        if sizes is None:
            sizes = self.population
        force = beta * (matrix @ (infectious / sizes))

        d_maternal = -omega * maternal - vaccination * maternal - mu * maternal
        d_maternal = d_maternal + self._transport(maternal)
        d_maternal[0] += births

        d_susceptible = omega * maternal - force * susceptible - vaccination * susceptible
        d_susceptible = d_susceptible - mu * susceptible + self._transport(susceptible)

        d_exposed = force * susceptible - (epsilon + mu) * exposed
        d_exposed = d_exposed + self._transport(exposed)

        d_infectious = epsilon * exposed - (gamma + mu) * infectious
        d_infectious = d_infectious + self._transport(infectious)

        d_removed = gamma * infectious + vaccination * (susceptible + maternal)
        d_removed = d_removed - mu * removed + self._transport(removed)

        return np.concatenate(
            [d_maternal, d_susceptible, d_exposed, d_infectious, d_removed]
        ).tolist()

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------
    def aggregate(self, compartment: str) -> np.ndarray:
        """
        Sum one compartment over all age groups.

        :param compartment: One of ``'M'``, ``'S'``, ``'E'``, ``'I'``, ``'R'``.
        :return: Array with the same length as the simulation time grid.
        """
        if compartment not in COMPARTMENTS:
            raise ValueError(
                f"Unknown compartment {compartment!r}; expected one of {COMPARTMENTS}"
            )
        if not self.traces:
            raise ValueError("No simulation results. Run the model first.")
        return np.sum([self.traces[k] for k in self.compartment_labels(compartment)], axis=0)

    def summary(self) -> dict[str, float | int]:
        """
        Epidemic summary statistics, aggregated over age groups.

        Mirrors :meth:`epimodels.BaseModel.summary` but reports totals rather
        than per-age-group compartments.

        :return: Dictionary with ``peak_I``, ``peak_time``, ``duration``,
            ``final_S``, ``final_R``, ``attack_rate`` and ``t_start``/``t_end``.
        """
        if not self.traces:
            raise ValueError("No simulation results. Run the model first.")

        time = np.asarray(self.traces["time"])
        infectious = self.aggregate("I")
        susceptible = self.aggregate("S")
        removed = self.aggregate("R")

        stats: dict[str, float | int] = {
            "model": self.model_type or "unknown",
            "t_start": float(time[0]),
            "t_end": float(time[-1]),
            "peak_I": float(infectious.max()),
            "peak_time": float(time[infectious.argmax()]),
        }
        if infectious[-1] < 1:
            below_one = np.flatnonzero(infectious < 1)
            if below_one.size and below_one[0] > 0:
                stats["duration"] = float(time[below_one[0]])

        stats["final_S"] = float(susceptible[-1])
        stats["final_R"] = float(removed[-1])
        if susceptible[0] > 0:
            stats["attack_rate"] = float(
                (susceptible[0] - susceptible[-1]) / susceptible[0]
            )
        return stats

    def plot_traces(
        self, vars=None, population: float | None = None, ax=None
    ) -> dict[str, np.ndarray]:
        """
        Plot compartment trajectories.

        By default the five compartments are summed over age groups, which is
        the readable view for a model with ``5 * A`` state variables. Pass
        individual state-variable names to plot specific age groups.

        :param vars: ``None`` for aggregated compartments, a compartment letter,
            a list of compartment letters, or a list of state-variable names
            such as ``['I_0', 'I_1']``.
        :param population: Population size used to express totals per 100 000
            instead of absolute counts.
        :param ax: Optional matplotlib axes.
        :return: Dictionary of the plotted series.
        """
        from matplotlib import pyplot as plt

        if vars is None:
            requested = list(COMPARTMENTS)
        elif isinstance(vars, str):
            requested = [vars]
        else:
            requested = list(vars)

        if not self.traces:
            raise ValueError("No simulation results. Run the model first.")

        if ax is None:
            _, ax = plt.subplots()

        scale = 1.0 if population is None else 1e5 / population
        plotted: dict[str, np.ndarray] = {}
        for name in requested:
            if name in COMPARTMENTS:
                series = self.aggregate(name)
            elif name in self.traces:
                series = np.asarray(self.traces[name])
            else:
                raise ValueError(
                    f"{name!r} is neither a compartment {COMPARTMENTS} nor a "
                    "state variable of this model"
                )
            plotted[name] = series * scale
            ax.plot(self.traces["time"], plotted[name], label=name)

        ax.legend(loc=0)
        ax.grid(True)
        label = "per 100 000" if population is not None else "count"
        ax.set_ylabel(label)
        ax.set_xlabel("time")
        ax.set_title(f"{self.model_type} model")
        return plotted

    def get_formulas(self) -> dict:
        """
        Symbolic formulas are not available for this model.

        :raises FormulaExtractionError: Always — the right-hand side is built
            from ``5 * A`` coupled equations assembled with array operations, so
            neither introspective extraction nor a hand-written SymPy dict is
            practical. Use :attr:`R0` (next-generation matrix) or a reduced
            model for symbolic work.
        """
        raise FormulaExtractionError(
            "AgeStructuredSEIR",
            f"the right-hand side is assembled from {self.dimension} coupled "
            "equations built with array operations",
            "Use the R0 property (next-generation matrix) or analyse a reduced "
            "model symbolically instead.",
        )

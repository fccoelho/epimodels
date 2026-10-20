"""
Contact-matrix utilities for age-structured epidemic models.

Age-structured transmission models need a *who-acquires-infection-from-whom*
(WAIFW) matrix ``C`` whose entry ``C[i, j]`` is the average number of contacts
per unit time that an individual of age group ``i`` makes with individuals of
age group ``j``.

Two properties matter in practice and are easy to get wrong:

1. **Detailed balance (reciprocity).** If ``N[i]`` is the size of group ``i``,
   the *number* of contacts between two groups must be symmetric:
   ``N[i] * C[i, j] == N[j] * C[j, i]``. Matrices built from mixing profiles
   generally violate this. The consequence is not cosmetic: the
   next-generation operator ``K[i, j] = beta * (N[i]/N[j]) * C[i, j]`` stops
   being similar to a symmetric matrix, and any eigen-analysis that assumes
   symmetry (``numpy.linalg.eigvalsh``, for instance) then returns values that
   are simply wrong. See :func:`check_reciprocity`.
2. **Age resolution.** Empirical matrices (e.g. POLYMOD) are published on a
   fixed set of age bands. Targeting a different set requires aggregating the
   matrix, which must be done with population weights to preserve (1).

This module provides dependency-light helpers for both, plus a parametric
POLYMOD-*like* generator that is reciprocal by construction.

.. warning::
   :func:`polymod_like_contact_matrix` is a *parametric approximation* of the
   qualitative features of the empirical POLYMOD matrices (peak contact rates
   in school-age children, decline with age, age-assortative mixing). It is
   meant for prototyping and for models whose age structure matches no
   published matrix. It is **not** a fit to the POLYMOD data — pass your own
   measured matrix to the model when the answer depends on the actual contact
   rates.

References
----------
Mossong J, et al. Social contacts and mixing patterns relevant to the spread
of infectious diseases. PLoS Med. 2008;5(3):e74.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "DEFAULT_AGE_BOUNDS",
    "age_group_labels",
    "aggregate_contact_matrix",
    "check_reciprocity",
    "enforce_reciprocity",
    "polymod_like_contact_matrix",
]

#: Lower age edges (in years) of a 21-group structure with fine resolution in
#: the first year of life and in infancy, then five-year bands up to 80+.
#: The last band is open-ended.
DEFAULT_AGE_BOUNDS: tuple[float, ...] = (0, 1, 2, 3, 4, 5, *range(10, 85, 5))


def _validate_age_bounds(age_bounds) -> np.ndarray:
    """Coerce and validate a sequence of lower age edges."""
    edges = np.asarray(age_bounds, dtype=float)
    if edges.ndim != 1 or edges.size == 0:
        raise ValueError("age_bounds must be a non-empty 1-D sequence")
    if np.any(np.diff(edges) <= 0):
        raise ValueError("age_bounds must be strictly increasing")
    return edges


def age_group_labels(age_bounds) -> tuple[str, ...]:
    """
    Human-readable labels for the age groups defined by ``age_bounds``.

    :param age_bounds: Strictly increasing sequence of lower age edges in
        years; the last group is treated as open-ended.
    :return: Tuple of labels such as ``('0-1', '1-2', ..., '80+')``.
    """
    edges = _validate_age_bounds(age_bounds)
    labels = [f"{edges[i]:g}-{edges[i + 1]:g}" for i in range(edges.size - 1)]
    labels.append(f"{edges[-1]:g}+")
    return tuple(labels)


def enforce_reciprocity(contact_matrix, population) -> np.ndarray:
    """
    Symmetrise a contact matrix so that it satisfies detailed balance exactly.

    Given a per-capita matrix ``C`` and group sizes ``N``, the number of
    contacts between groups ``i`` and ``j`` is ``M[i, j] = N[i] * C[i, j]``.
    The symmetric part ``(M + M.T) / 2`` is taken and converted back to
    per-capita rates, so the result satisfies ``N[i]*C[i, j] == N[j]*C[j, i]``
    to machine precision.

    .. note::
       Reconciliation in *count* space averages each direction of the pair, so
       an age group that is small compared to its neighbours can see its
       per-capita rate rise noticeably. If your matrix comes from validated
       data, prefer passing it through unchanged (``enforce_reciprocity=False``
       in :class:`~epimodels.continuous.age_structured.AgeStructuredSEIR`) and
       verify it with :func:`check_reciprocity`.

    :param contact_matrix: ``(A, A)`` array of per-capita contact rates.
    :param population: Length-``A`` array of group sizes (must be positive).
    :return: Reciprocity-enforcing ``(A, A)`` array.
    :raises ValueError: If shapes are incompatible or a group is empty.
    """
    matrix = np.asarray(contact_matrix, dtype=float)
    sizes = np.asarray(population, dtype=float)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"contact_matrix must be square, got shape {matrix.shape}")
    if sizes.shape != (matrix.shape[0],):
        raise ValueError(
            f"population must have length {matrix.shape[0]}, got shape {sizes.shape}"
        )
    if np.any(sizes <= 0):
        bad = np.flatnonzero(sizes <= 0).tolist()
        raise ValueError(
            "Cannot enforce reciprocity: age group(s) "
            f"{bad} have non-positive population. Pass an explicit "
            "`population=` with every group populated."
        )

    counts = sizes[:, None] * matrix
    counts = 0.5 * (counts + counts.T)
    return counts / sizes[:, None]


def check_reciprocity(contact_matrix, population, tol: float = 1e-8) -> bool:
    """
    Test whether ``N[i]*C[i, j] == N[j]*C[j, i]`` holds within ``tol``.

    The comparison is made on contact *counts* and scaled by their maximum, so
    ``tol`` is a relative tolerance.

    :param contact_matrix: ``(A, A)`` per-capita contact matrix.
    :param population: Length-``A`` array of group sizes.
    :param tol: Relative tolerance.
    :return: ``True`` when the matrix satisfies detailed balance.
    """
    matrix = np.asarray(contact_matrix, dtype=float)
    sizes = np.asarray(population, dtype=float)
    if matrix.shape != (sizes.size, sizes.size):
        raise ValueError(
            f"contact_matrix shape {matrix.shape} incompatible with "
            f"population of length {sizes.size}"
        )

    counts = sizes[:, None] * matrix
    scale = np.abs(counts).max()
    if scale == 0:
        return True
    return bool(np.abs(counts - counts.T).max() <= tol * scale)


def _activity_profile(
    midpoints: np.ndarray,
    age_peak: float,
    young_spread: float,
    old_spread: float,
    old_age_ratio: float,
) -> np.ndarray:
    """Asymmetric bell-shaped target contact rate by age."""
    spread = np.where(midpoints < age_peak, young_spread, old_spread)
    return old_age_ratio + (1.0 - old_age_ratio) * np.exp(
        -(((midpoints - age_peak) / spread) ** 2)
    )


def polymod_like_contact_matrix(
    age_bounds=None,
    population=None,
    *,
    peak_contacts: float = 18.0,
    age_peak: float = 10.0,
    young_spread: float = 8.0,
    old_spread: float = 50.0,
    old_age_ratio: float = 0.39,
    contact_decay: float = 25.0,
    n_iter: int = 100,
) -> np.ndarray:
    """
    Parametric contact matrix with the qualitative shape of POLYMOD.

    The matrix is built in two steps. First a symmetric *affinity* is formed
    from an age-specific activity profile and an age-assortative kernel,

    .. math::

        A_{ij} = u_i u_j e^{-|a_i - a_j| / \tau}

    and the per-capita rates follow as

    .. math::

        C_{ij} = \\frac{A_{ij} N_j}{\\sum_k N_k}

    which satisfies ``N[i]*C[i, j] == N[j]*C[j, i]`` **exactly**, for any
    activity vector ``u`` and any population vector ``N``. The activity vector
    is then found by a damped fixed-point iteration so that group ``i`` makes
    a number of contacts proportional to the requested profile, and the whole
    matrix is rescaled so that the busiest group makes ``peak_contacts``
    contacts per unit time.

    The result reproduces the shape reported by Mossong et al. (2008) — of the
    order of 10 contacts/day in infants, a peak near 18 in 10-14 year-olds and
    7-8 in the elderly — but it is not a fit to those data.

    :param age_bounds: Lower age edges in years (default
        :data:`DEFAULT_AGE_BOUNDS`). The last group is open-ended.
    :param population: Group sizes used to impose detailed balance. Defaults to
        equal-sized groups; pass your population structure to make the matrix
        reciprocal for the population you will actually model.
    :param peak_contacts: Contacts per unit time of the busiest age group.
    :param age_peak: Age (years) at which contact activity peaks.
    :param young_spread: Spread of the activity peak below ``age_peak``.
    :param old_spread: Spread of the activity peak above ``age_peak``.
    :param old_age_ratio: Contact rate of the oldest group, relative to the peak.
    :param contact_decay: Age-assortativity length scale, in years.
    :param n_iter: Fixed-point iterations used to match the activity profile.
    :return: ``(A, A)`` array of per-capita contact rates, reciprocal with
        respect to ``population``.
    """
    edges = _validate_age_bounds(
        DEFAULT_AGE_BOUNDS if age_bounds is None else age_bounds
    )
    n_groups = edges.size

    sizes = (
        np.ones(n_groups, dtype=float)
        if population is None
        else np.asarray(population, dtype=float)
    )
    if sizes.shape != (n_groups,):
        raise ValueError(
            f"population must have length {n_groups}, got shape {sizes.shape}"
        )
    if np.any(sizes <= 0):
        raise ValueError("population entries must all be positive")

    midpoints = np.empty(n_groups)
    midpoints[:-1] = 0.5 * (edges[:-1] + edges[1:])
    # The open-ended group is represented by a nominal midpoint ten years past
    # its lower edge.
    midpoints[-1] = edges[-1] + 10.0

    kernel = np.exp(-np.abs(midpoints[:, None] - midpoints[None, :]) / contact_decay)
    target = _activity_profile(
        midpoints, age_peak, young_spread, old_spread, old_age_ratio
    )

    # Damped fixed point: u <- sqrt(u * target / (K @ (u * N) / sum(N)))
    total = sizes.sum()
    activity = target.copy()
    for _ in range(n_iter):
        average = (kernel * sizes[None, :]).dot(activity) / total
        activity = np.sqrt(activity * (target / np.maximum(average, 1e-12)))

    affinity = activity[:, None] * activity[None, :] * kernel
    matrix = affinity * (sizes[None, :] / total)

    busiest = matrix.sum(axis=1).max()
    if busiest <= 0:
        raise ValueError("degenerate contact matrix: no mixing between any groups")
    return matrix * (peak_contacts / busiest)


def aggregate_contact_matrix(
    contact_matrix,
    source_population,
    source_to_target,
    n_targets: int | None = None,
) -> np.ndarray:
    """
    Coarsen a contact matrix onto a coarser age grouping.

    Source groups are pooled with population weights, so the per-capita rate
    between target groups ``i`` and ``j`` is the total number of contacts made
    by all members of ``i`` with all members of ``j``, divided by the size of
    ``i``:

    .. math::

        C^{T}_{ij} = \\frac{\\sum_{a \\in i} \\sum_{b \\in j} n_a C_{ab}}
                          {\\sum_{a \\in i} n_a}

    If the input satisfies detailed balance with ``source_population``, the
    output satisfies it with the pooled population — which is what makes the
    aggregation safe to use inside a model.

    :param contact_matrix: ``(S, S)`` per-capita matrix on the source grid.
    :param source_population: Length-``S`` array of source group sizes.
    :param source_to_target: Length-``S`` integer array giving the target group
        index of each source group (labels need not be sorted or contiguous).
    :param n_targets: Number of target groups (inferred when omitted).
    :return: ``(T, T)`` aggregated per-capita matrix.
    """
    matrix = np.asarray(contact_matrix, dtype=float)
    sizes = np.asarray(source_population, dtype=float)
    mapping = np.asarray(source_to_target)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"contact_matrix must be square, got shape {matrix.shape}")
    if sizes.shape != (matrix.shape[0],):
        raise ValueError(
            f"source_population must have length {matrix.shape[0]}, "
            f"got shape {sizes.shape}"
        )
    if mapping.shape != (matrix.shape[0],):
        raise ValueError(
            f"source_to_target must have length {matrix.shape[0]}, "
            f"got shape {mapping.shape}"
        )
    if not np.issubdtype(mapping.dtype, np.integer):
        if not np.all(np.equal(np.mod(mapping, 1), 0)):
            raise ValueError("source_to_target must contain integer group indices")
        mapping = mapping.astype(int)
    if np.any(mapping < 0):
        raise ValueError("source_to_target must contain non-negative indices")

    n_groups = int(mapping.max()) + 1 if n_targets is None else int(n_targets)
    if mapping.max() >= n_groups:
        raise ValueError(
            f"source_to_target references group {int(mapping.max())} but "
            f"n_targets={n_groups}"
        )

    counts = sizes[:, None] * matrix
    aggregated = np.zeros((n_groups, n_groups))
    np.add.at(aggregated, (mapping[:, None], mapping[None, :]), counts)

    pooled = np.bincount(mapping, weights=sizes, minlength=n_groups)
    if np.any(pooled <= 0):
        empty = np.flatnonzero(pooled <= 0).tolist()
        raise ValueError(f"Target age group(s) {empty} receive no source groups")
    return aggregated / pooled[:, None]

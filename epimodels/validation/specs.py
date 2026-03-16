"""
Specification classes for parameters, state variables, and constraints.
"""

from dataclasses import dataclass, field
from typing import Any
from enum import Enum


class DomainType(Enum):
    """Type of numeric domain for parameters and variables."""

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"
    INTEGER = "integer"


@dataclass
class ParameterSpec:
    """
    Rich specification for a model parameter.

    Attributes:
        name: Parameter identifier (used in code)
        symbol: LaTeX representation for display
        description: Human-readable description
        domain_type: Type of numeric domain
        bounds: Optional (min, max) tuple. Use None for unbounded.
                Example: (0, None) means positive, (0, 1) means probability
        dtype: Expected Python type (float, int, list, etc.)
        default: Default value if parameter is optional
        required: Whether parameter must be provided
        constraints: List of constraint expressions (e.g., "value > 0", "value < other_param")
        units: Physical units (e.g., "1/day", "individuals")
        typical_range: Typical range for documentation (min, max)

    Example:
        >>> spec = ParameterSpec(
        ...     name="beta",
        ...     symbol=r"$\\beta$",
        ...     description="Transmission rate",
        ...     bounds=(0, None),
        ...     constraints=["value > 0"]
        ... )
    """

    name: str
    symbol: str
    description: str = ""
    domain_type: DomainType = DomainType.CONTINUOUS
    bounds: tuple[float | None, float | None] | None = None
    dtype: type = float
    default: Any | None = None
    required: bool = True
    constraints: list[str] = field(default_factory=list)
    units: str | None = None
    typical_range: tuple[float, float] | None = None

    def __post_init__(self):
        """Validate spec consistency."""
        if self.bounds is not None:
            min_val, max_val = self.bounds
            if min_val is not None and max_val is not None and min_val >= max_val:
                raise ValueError(
                    f"Invalid bounds for parameter '{self.name}': "
                    f"min ({min_val}) must be less than max ({max_val})"
                )


@dataclass
class VariableSpec:
    """
    Specification for a state variable.

    Attributes:
        name: Variable identifier (used in code)
        symbol: LaTeX representation
        description: Human-readable description
        bounds: Optional (min, max) tuple
        non_negative: Whether variable must be >= 0
        constraints: List of constraint expressions
        units: Physical units

    Example:
        >>> spec = VariableSpec(
        ...     name="S",
        ...     symbol="S",
        ...     description="Susceptible individuals",
        ...     non_negative=True
        ... )
    """

    name: str
    symbol: str
    description: str = ""
    bounds: tuple[float | None, float | None] | None = None
    non_negative: bool = True
    constraints: list[str] = field(default_factory=list)
    units: str | None = None

    def __post_init__(self):
        """Validate spec consistency."""
        if self.bounds is not None:
            min_val, max_val = self.bounds
            if min_val is not None and max_val is not None and min_val >= max_val:
                raise ValueError(
                    f"Invalid bounds for variable '{self.name}': "
                    f"min ({min_val}) must be less than max ({max_val})"
                )
        if self.non_negative and self.bounds is None:
            self.bounds = (0.0, None)
        elif self.non_negative and self.bounds is not None:
            min_val, max_val = self.bounds
            if min_val is not None and min_val < 0:
                raise ValueError(
                    f"Variable '{self.name}' is marked non_negative but has min bound {min_val}"
                )


@dataclass
class ModelConstraint:
    """
    Cross-parameter or model-level constraint.

    Attributes:
        expression: Constraint expression (Python expression or SymPy-parseable)
                   Can reference parameter names directly.
                   Examples: "beta > gamma", "p + q <= 1", "R0 > 1"
        description: Human-readable description of the constraint
        severity: "error" (raises exception) or "warning" (logs warning)
        name: Optional name for the constraint

    Example:
        >>> constraint = ModelConstraint(
        ...     expression="beta / gamma > 1",
        ...     description="R0 > 1 required for epidemic",
        ...     severity="warning"
        ... )
    """

    expression: str
    description: str = ""
    severity: str = "error"
    name: str | None = None

    def __post_init__(self):
        """Validate constraint."""
        if self.severity not in ("error", "warning"):
            raise ValueError(f"Invalid severity '{self.severity}'. Must be 'error' or 'warning'")
        if not self.description:
            self.description = f"Constraint: {self.expression}"

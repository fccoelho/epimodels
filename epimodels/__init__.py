from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any
import copy as copy_module
import warnings

if TYPE_CHECKING:
    import pandas as pd

from epimodels.exceptions import ValidationError
from epimodels.validation.specs import ParameterSpec, VariableSpec, ModelConstraint
from epimodels.validation.validators import (
    validate_parameter_value,
    validate_initial_condition,
    evaluate_constraint,
)


try:
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

from matplotlib import pyplot as plt


class BaseModel:
    """
    Base class for all models both discrete and continuous

    Supports two validation modes:
    1. Simple mode (backward compatible): Uses `parameters` and `state_variables` dicts
    2. Rich mode (new): Uses `parameter_specs` and `variable_specs` for enhanced validation
    """

    name: str | None
    model_type: str | None
    state_variables: dict[str, str]
    parameters: dict[str, str]
    param_values: dict[str, Any]
    traces: dict[str, Any]

    parameter_specs: dict[str, ParameterSpec]
    variable_specs: dict[str, VariableSpec]
    model_constraints: list[ModelConstraint]
    symbolic_model: Any | None

    def __init__(self) -> None:
        self.name = None
        self.model_type = None
        self.state_variables = {}
        self.parameters = {}
        self.param_values = {}
        self.traces = {}

        self.parameter_specs = {}
        self.variable_specs = {}
        self.model_constraints = []
        self.symbolic_model = None

    def validate_parameters(self, params: dict[str, Any]) -> None:
        """
        Validate parameters using either rich specifications or simple validation.

        If parameter_specs are defined, uses rich validation with:
        - Type checking
        - Domain/bounds validation
        - Constraint evaluation

        Otherwise falls back to simple validation (backward compatible).

        Non-numeric parameters (like lists for imported cases) are allowed
        for advanced model features.

        :param params: Dictionary of parameter values
        :raises ValidationError: If validation fails
        """
        if self.parameter_specs:
            self._validate_parameters_rich(params)
        else:
            self._validate_parameters_simple(params)

    def _validate_parameters_simple(self, params: dict[str, Any]) -> None:
        """Simple validation for backward compatibility."""
        missing = set(self.parameters.keys()) - set(params.keys())
        if missing:
            raise ValidationError(f"Missing required parameters: {missing}")

        for param, value in params.items():
            if param in self.parameters:
                if isinstance(value, (int, float)) and value < 0:
                    raise ValidationError(f"Parameter '{param}' must be non-negative, got {value}")

    def _validate_parameters_rich(self, params: dict[str, Any]) -> None:
        """Rich validation using ParameterSpec."""
        errors = []
        warnings_list = []

        for name, spec in self.parameter_specs.items():
            if spec.required and name not in params:
                errors.append(f"Missing required parameter: {name}")
            elif name in params:
                value = params[name]
                param_errors = validate_parameter_value(name, value, spec, params)
                errors.extend(param_errors)

        for constraint in self.model_constraints:
            satisfied, error_msg = evaluate_constraint(constraint.expression, params)

            if not satisfied:
                msg = f"Constraint violated: {constraint.description or constraint.expression}" + (
                    f" ({error_msg})" if error_msg else ""
                )

                if constraint.severity == "error":
                    errors.append(msg)
                else:
                    warnings_list.append(msg)

        for warning_msg in warnings_list:
            warnings.warn(warning_msg, UserWarning)

        if errors:
            raise ValidationError("\n".join(errors))

    def define_parameter(self, spec: ParameterSpec) -> None:
        """
        Register a parameter specification.

        :param spec: ParameterSpec object defining the parameter
        """
        self.parameter_specs[spec.name] = spec
        self.parameters[spec.name] = spec.symbol

    def define_variable(self, spec: VariableSpec) -> None:
        """
        Register a state variable specification.

        :param spec: VariableSpec object defining the variable
        """
        self.variable_specs[spec.name] = spec
        self.state_variables[spec.name] = spec.symbol

    def add_constraint(self, constraint: ModelConstraint) -> None:
        """
        Add a cross-parameter constraint.

        :param constraint: ModelConstraint object
        """
        self.model_constraints.append(constraint)

    def validate_initial_conditions(self, inits: list[float], totpop: float) -> None:
        """
        Validate initial conditions using either rich specifications or simple validation.

        If variable_specs are defined, uses rich validation with:
        - Bounds checking
        - Non-negativity
        - Constraint evaluation

        Otherwise falls back to simple validation (backward compatible).

        :param inits: List of initial condition values
        :param totpop: Total population
        :raises ValidationError: If validation fails
        """
        if self.variable_specs:
            self._validate_initial_conditions_rich(inits, totpop)
        else:
            self._validate_initial_conditions_simple(inits, totpop)

    def _validate_initial_conditions_simple(self, inits: list[float], totpop: float) -> None:
        """Simple validation for backward compatibility."""
        if len(inits) < len(self.state_variables):
            raise ValidationError(
                f"Expected at least {len(self.state_variables)} initial conditions, got {len(inits)}"
            )

        for i, val in enumerate(inits):
            if val < 0:
                raise ValidationError(
                    f"Initial condition at index {i} must be non-negative, got {val}"
                )

        if sum(inits) > totpop * 1.001:  # Allow small numerical tolerance
            raise ValidationError(
                f"Sum of initial conditions ({sum(inits)}) exceeds total population ({totpop})"
            )

    def _validate_initial_conditions_rich(self, inits: list[float], totpop: float) -> None:
        """Rich validation using VariableSpec."""
        errors = []

        var_names = list(self.variable_specs.keys())

        if len(inits) < len(var_names):
            raise ValidationError(
                f"Expected at least {len(var_names)} initial conditions, got {len(inits)}"
            )

        init_dict = {}
        for i, (name, val) in enumerate(zip(var_names, inits)):
            init_dict[name] = val

            if name in self.variable_specs:
                spec = self.variable_specs[name]
                var_errors = validate_initial_condition(name, val, spec, init_dict)
                errors.extend(var_errors)

        if sum(inits) > totpop * 1.001:
            errors.append(
                f"Sum of initial conditions ({sum(inits)}) exceeds total population ({totpop})"
            )

        if errors:
            raise ValidationError("\n".join(errors))

    def validate_time_range(self, trange: list[float]) -> None:
        """
        Validate time range input.

        :param trange: Time range [t0, tf]
        :raises ValidationError: If validation fails
        """
        if not isinstance(trange, (list, tuple)) or len(trange) != 2:
            raise ValidationError(f"trange must be a list or tuple of 2 values, got {trange}")

        t0, tf = trange
        if t0 >= tf:
            raise ValidationError(f"Invalid time range: start ({t0}) must be less than end ({tf})")

    def plot_traces(self, vars: list | None = None) -> None:
        """
        Plots the simulations
        :param vars: variables to plot
        """
        if vars is None:
            vars = []
        for series, data in self.traces.items():
            if series in self.state_variables:
                plt.plot(self.traces["time"], data, label=series)
        plt.legend(loc=0)
        plt.grid()
        plt.title("{} model".format(self.model_type))

    def parameter_table(self, latex: bool = False) -> dict | str:
        if self.parameters:
            tbl = {
                "Parameter": list(self.parameters.keys()),
                "Value": list(self.param_values.values()),
                "Symbol": list(self.parameters.values()),
            }

            if latex:
                out = (
                    r"""\begin[l|c|c]{tabular}
                \hline
                Parameter & Value & Symbol \\
                \hline
                """
                    + r"\\".join(
                        [
                            f"{p}&{v}&{s}"
                            for p, v, s in zip(tbl["Parameter"], tbl["Value"], tbl["Symbol"])
                        ]
                    )
                    + r"""
                    \hline
                    \end{tabular}"""
                )
            else:
                out = tbl
        else:
            out = {}
        return out

    def to_dataframe(self) -> "pd.DataFrame":
        """
        Return simulation results as a pandas DataFrame.

        :return: DataFrame with time and state variable columns
        :raises ImportError: If pandas is not installed
        :raises ValueError: If no simulation has been run
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for to_dataframe(). "
                "Install with: pip install epimodels[dataframe]"
            )

        if not self.traces:
            raise ValueError("No simulation results. Run the model first.")

        return pd.DataFrame(self.traces)

    def to_dict(self) -> dict[str, Any]:
        """
        Return a deep copy of simulation traces.

        :return: Dictionary with time and state variable arrays
        """
        return copy_module.deepcopy(self.traces)

    def summary(self) -> dict[str, float | int]:
        """
        Return epidemic summary statistics.

        Requires simulation results to be available.

        :return: Dictionary with statistics:
            - peak_I: Maximum number of infectious individuals
            - peak_time: Time at which I is maximum
            - final_S: Final number of susceptible individuals
            - final_R: Final number of removed/recovered individuals
            - attack_rate: Proportion of population that was infected
            - duration: Time until I drops below 1 (if applicable)
        :raises ValueError: If no simulation has been run
        """
        if not self.traces:
            raise ValueError("No simulation results. Run the model first.")

        stats: dict[str, float | int] = {"model": self.model_type or "unknown"}

        if "time" in self.traces:
            time = self.traces["time"]
            stats["t_start"] = float(time[0])
            stats["t_end"] = float(time[-1])

        if "I" in self.traces:
            I = self.traces["I"]
            stats["peak_I"] = float(I.max())
            if "time" in self.traces:
                stats["peak_time"] = float(self.traces["time"][I.argmax()])

            if I[-1] < 1:
                below_one = I < 1
                if below_one.any():
                    idx = below_one.argmax()
                    if idx > 0:
                        stats["duration"] = float(self.traces["time"][idx])

        if "S" in self.traces:
            stats["final_S"] = float(self.traces["S"][-1])

        if "R" in self.traces:
            stats["final_R"] = float(self.traces["R"][-1])

        if "S" in self.traces and "I" in self.traces:
            S0 = self.traces["S"][0]
            S_final = self.traces["S"][-1]
            if S0 > 0:
                stats["attack_rate"] = float((S0 - S_final) / S0)

        return stats

    def copy(self, include_traces: bool = False) -> "BaseModel":
        """
        Create a copy of the model configuration.

        :param include_traces: If True, copy simulation results too
        :return: New model instance with same configuration
        """
        new_model = copy_module.copy(self)
        if not include_traces:
            new_model.traces = {}
            new_model.param_values = {}
        else:
            new_model.traces = copy_module.deepcopy(self.traces)
            new_model.param_values = copy_module.deepcopy(self.param_values)
        return new_model

    def reset(self) -> None:
        """Clear simulation results and parameter values."""
        self.traces = {}
        self.param_values = {}

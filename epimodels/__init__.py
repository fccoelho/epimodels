from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any
import copy as copy_module

if TYPE_CHECKING:
    import pandas as pd


class ValidationError(Exception):
    """Raised when parameter or initial condition validation fails."""

    pass


class FormulaExtractionError(Exception):
    """
    Raised when automatic formula extraction fails for a ContinuousModel.

    This typically occurs when the model's _model method uses constructs
    that cannot be symbolically executed (e.g., loops, conditionals).

    Attributes:
        model_name: Name of the model that failed extraction
        reason: Description of why extraction failed
        suggestion: Suggested fix for the user
    """

    def __init__(self, model_name: str, reason: str, suggestion: str = ""):
        self.model_name = model_name
        self.reason = reason
        self.suggestion = suggestion

        message = f"Cannot extract formulas from {model_name}: {reason}"
        if suggestion:
            message += f"\n{suggestion}"
        super().__init__(message)


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
    """

    name: str | None
    model_type: str | None
    state_variables: dict[str, str]
    parameters: dict[str, str]
    param_values: dict[str, Any]
    traces: dict[str, Any]

    def __init__(self) -> None:
        self.name = None
        self.model_type = None
        self.state_variables = {}
        self.parameters = {}
        self.param_values = {}
        self.traces = {}

    def validate_parameters(self, params: dict[str, Any]) -> None:
        """
        Validate that all required parameters are present.

        Non-numeric parameters (like lists for imported cases) are allowed
        for advanced model features.

        :param params: Dictionary of parameter values
        :raises ValidationError: If validation fails
        """
        missing = set(self.parameters.keys()) - set(params.keys())
        if missing:
            raise ValidationError(f"Missing required parameters: {missing}")

        for param, value in params.items():
            if param in self.parameters:
                if isinstance(value, (int, float)) and value < 0:
                    raise ValidationError(f"Parameter '{param}' must be non-negative, got {value}")

    def validate_initial_conditions(self, inits: list[float], totpop: float) -> None:
        """
        Validate initial conditions.

        :param inits: List of initial condition values
        :param totpop: Total population
        :raises ValidationError: If validation fails
        """
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

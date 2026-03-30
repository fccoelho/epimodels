"""
Created on 09/11/18
by fccoelho
license: GPL V3 or Later
"""

import numpy as np
import sympy as sp
from epimodels import BaseModel
import logging
from collections import OrderedDict
from typing import Any, Union
import copy
import matplotlib.pyplot as plt

logging.basicConfig(filename="epimodels.log", filemode="w", level=logging.DEBUG)


class ContinuousModel(BaseModel):
    """
    Exposes a library of continuous time population models
    """

    _formulas: dict[str, Any] | None

    def __init__(self) -> None:
        """
        Base class for Continuous models
        """
        super().__init__()
        self._formulas = None

    def __call__(
        self,
        inits: list[float],
        trange: list[float],
        totpop: float,
        params: dict,
        method: str = "RK45",
        solver: Any = None,
        validate: bool = True,
        **kwargs,
    ):
        """
        Run the model

        :param inits: initial conditions
        :param trange: time range: [t0, tf]
        :param totpop: total population size
        :param params: dictionary of parameters
        :param method: integration method (deprecated, use solver instead). default is 'RK45'
        :param solver: SolverBase instance (ScipySolver or DiffraxSolver). If None, uses ScipySolver with method.
        :param validate: whether to validate parameters and initial conditions
        :param kwargs: Additional parameters passed on to the solver
        """
        if validate:
            self.validate_parameters(params)
            self.validate_initial_conditions(inits, totpop)

        self.param_values = OrderedDict((k, params[k]) for k in self.parameters.keys())

        if solver is not None:
            self.solver = solver
            self.method = getattr(solver, "method", getattr(solver, "solver_name", "custom"))
        else:
            from epimodels.solvers import ScipySolver

            self.solver = ScipySolver(method=method)
            self.method = method

        self.kwargs = kwargs
        sol = self.run(inits, trange, totpop, params, **kwargs)
        res = {v: sol.y[s, :] for v, s in zip(self.state_variables.keys(), range(sol.y.shape[0]))}
        res["time"] = sol.t
        self.traces.update(res)

    def _model(self, t: float, y: list[float], params: dict[str, float]) -> list[float]:
        raise NotImplementedError

    def __repr__(self):
        f = copy.deepcopy(self._model)
        desc = f"""
# Model: {self.model_type}

```mermaid
{self.diagram}
```
"""
        return desc

    @property
    def diagram(self) -> str:
        return "A[Define a diagram for this model]"

    @property
    def dimension(self) -> int:
        return len(self.state_variables)

    def run(self, inits, trange, totpop, params, **kwargs):
        params["N"] = totpop

        def fn(t, y):
            return self._model(t, y, params)

        sol = self.solver.solve(fn, tuple(trange), inits, **kwargs)
        return sol

    @property
    def formulas(self) -> dict[str, Any] | None:
        """
        Symbolic formulas for state variable derivatives.

        Subclasses can override by setting _formulas attribute for cases
        where automatic extraction fails.

        Returns:
            Dict mapping state variable names to SymPy expressions,
            or None if not available.
        """
        if hasattr(self, "_formulas") and self._formulas is not None:
            return self._formulas
        return None

    def get_formulas(self) -> dict[str, Any]:
        """
        Get formulas for state variable derivatives.

        Attempts automatic extraction first, falls back to manual _formulas.

        Returns:
            Dict mapping state variable names to SymPy expressions

        Raises:
            FormulaExtractionError: If extraction fails and no manual override
        """
        # Check for manual override first
        if self.formulas is not None:
            self._validate_formulas(self.formulas)
            return self.formulas

        # Attempt automatic extraction
        from epimodels.formulas import extract_formulas

        return extract_formulas(self)

    def _validate_formulas(self, formulas: dict) -> None:
        """Validate manually defined formulas."""
        from sympy import Expr

        missing_states = set(self.state_variables) - set(formulas)
        if missing_states:
            raise ValueError(f"Formulas missing for state variables: {missing_states}")

        for name, expr in formulas.items():
            if not isinstance(expr, (Expr, int, float)):
                raise TypeError(
                    f"Formula for '{name}' must be a SymPy expression, got {type(expr).__name__}"
                )

    def to_vfgen(
        self,
        filepath: str | None = None,
        default_values: dict[str, float] | None = None,
        initial_conditions: dict[str, float] | None = None,
        population: float | None = None,
        **kwargs,
    ) -> str | None:
        """
        Export the model to vfgen XML format.

        Args:
            filepath: Optional path to write XML file. If None, returns XML string.
            default_values: Parameter default values. Uses model.param_values if None.
            initial_conditions: Initial conditions for state variables.
            population: Total population N value.
            **kwargs: Additional arguments passed to VFGenExporter.export()

        Returns:
            XML string if filepath is None, otherwise None (writes to file)

        Raises:
            FormulaExtractionError: If formula extraction fails

        Example:
            >>> model = SIR()
            >>> model.param_values = {'beta': 0.3, 'gamma': 0.1}
            >>> xml = model.to_vfgen(
            ...     initial_conditions={'S': 990, 'I': 10, 'R': 0},
            ...     population=1000
            ... )
        """
        from epimodels.exporters import VFGenExporter

        return VFGenExporter(self).export(
            filepath=filepath,
            default_values=default_values,
            initial_conditions=initial_conditions,
            population=population,
            **kwargs,
        )


class SIR(ContinuousModel):
    """
    SIR (Susceptible-Infectious-Removed) Model.

    A classic compartmental model for infectious disease dynamics.

    State Variables:
        - S: Susceptible individuals
        - I: Infectious individuals
        - R: Removed (recovered/immune) individuals

    Parameters:
        - beta (β): Transmission rate (contact rate × probability of transmission)
        - gamma (γ): Recovery rate (1 / average infectious period)

    Equations:

        dS/dt = -βSI/N
        dI/dt = βSI/N - γI
        dR/dt = γI

    Basic Reproduction Number:
        R₀ = β/γ

    Example:
        >>> model = SIR()
        >>> model([990, 10, 0], [0, 100], 1000, {'beta': 0.3, 'gamma': 0.1})
        >>> print(model.R0)  # 3.0
    """

    def __init__(self):
        super().__init__()
        self.state_variables = OrderedDict({"S": "Susceptible", "I": "Infectious", "R": "Removed"})
        self.parameters = OrderedDict({"beta": r"$\beta$", "gamma": r"$\gamma$"})
        self.model_type = "SIR"

    @property
    def diagram(self) -> str:
        """Mermaid diagram of the compartmental model"""
        return r"""flowchart LR
         
S(Susceptible) -->|$$\beta$$| I(Infectious)
I -->|$$\gamma$$| R(Removed)
"""

    @property
    def R0(self) -> float | None:
        """
        Basic reproduction number for SIR model.

        R0 = β / γ

        :return: Basic reproduction number, or None if parameters not set
        """
        if self.param_values and "beta" in self.param_values and "gamma" in self.param_values:
            return float(self.param_values["beta"] / self.param_values["gamma"])
        return None

    def _model(self, t: float, y: list[float], params: dict[str, float]) -> list[float]:
        S, I, R = y
        beta, gamma, N = params["beta"], params["gamma"], params["N"]
        return [-beta * S * I / N, beta * S * I / N - gamma * I, gamma * I]


class SIR1D(ContinuousModel):
    """
    One dimensional SIR model
    """

    def __init__(self):
        super().__init__()
        self.state_variables = OrderedDict({"R": "Recovered"})
        self.parameters = {"R0": r"{\cal R}_0", "gamma": r"\gamma", "S0": r"S_0"}
        self.model_type = "SIR1D"

    @property
    def diagram(self) -> str:
        """Mermaid diagram of the compartmental model"""
        return r"""flowchart LR
         
S(Susceptible) -->|$$\beta$$| I(Infectious)
I -->|$$\gamma$$| R(Recovered)
"""

    @property
    def R0(self) -> float | None:
        """
        Basic reproduction number for SIR1D model.

        R0 is a direct parameter in this model formulation.

        :return: Basic reproduction number, or None if parameters not set
        """
        if self.param_values and "R0" in self.param_values:
            return float(self.param_values["R0"])
        return None

    def _model(self, t: float, y: list[float], params: dict[str, float]) -> list[float]:
        N = params["N"]
        R = y[0]
        R0, gamma, S0 = params["R0"], params["gamma"], params["S0"]
        return [gamma * (N - R - (S0 * np.exp(-R0 * R)))]


class SIS(ContinuousModel):
    """
    SIS (Susceptible-Infectious-Susceptible) Model.

    A model for diseases that do not confer immunity after recovery.

    State Variables:
        - S: Susceptible individuals
        - I: Infectious individuals

    Parameters:
        - beta (β): Transmission rate
        - gamma (γ): Recovery rate

    Equations:

        dS/dt = -βSI/N + γI
        dI/dt = βSI/N - γI

    Basic Reproduction Number:
        R₀ = β/γ

    Note:
        Total population N = S + I is conserved.
        When R₀ > 1, endemic equilibrium exists at I* = N(1 - 1/R₀).
    """

    def __init__(self):
        super().__init__()
        self.state_variables = OrderedDict({"S": "Susceptible", "I": "Infectious"})
        self.parameters = {"beta": r"\beta", "gamma": r"\gamma"}
        self.model_type = "SIS"

    @property
    def diagram(self) -> str:
        """Mermaid diagram of the compartmental model"""
        return r"""flowchart LR
         
S(Susceptible) -->|$$\beta$$| I(Infectious)
I -->|$$\gamma$$| S
"""

    @property
    def R0(self) -> float | None:
        """
        Basic reproduction number for SIS model.

        R0 = β / γ

        :return: Basic reproduction number, or None if parameters not set
        """
        if self.param_values and "beta" in self.param_values and "gamma" in self.param_values:
            return float(self.param_values["beta"] / self.param_values["gamma"])
        return None

    def _model(self, t: float, y: list[float], params: dict[str, float]) -> list[float]:
        S, I = y
        beta, gamma, N = params["beta"], params["gamma"], params["N"]
        return [
            -beta * S * I / N + gamma * I,
            beta * S * I / N - gamma * I,
        ]


class SIRS(ContinuousModel):
    """
    SIRS (Susceptible-Infectious-Removed-Susceptible) Model.

    A model for diseases where immunity wanes over time.

    State Variables:
        - S: Susceptible individuals
        - I: Infectious individuals
        - R: Removed (temporarily immune) individuals

    Parameters:
        - beta (β): Transmission rate
        - gamma (γ): Recovery rate
        - xi (ξ): Waning immunity rate (1 / average immune period)

    Equations:

        dS/dt = -βSI/N + ξR
        dI/dt = βSI/N - γI
        dR/dt = γI - ξR

    Basic Reproduction Number:
        R₀ = β/γ

    Note:
        Unlike SIR, individuals in R return to S at rate ξ.
    """

    def __init__(self):
        super().__init__()
        self.state_variables = OrderedDict({"S": "Susceptible", "I": "Infectious", "R": "Removed"})
        self.parameters = OrderedDict({"beta": r"$\beta$", "gamma": r"$\gamma$", "xi": r"$\xi$"})
        self.model_type = "SIRS"

    @property
    def diagram(self) -> str:
        """Mermaid diagram of the compartmental model"""
        return r"""flowchart LR
         
S(Susceptible) -->|$$\beta$$| I(Infectious)
I -->|$$\gamma$$| R(Removed)
R -->|$$\xi$$| S
"""

    @property
    def R0(self) -> float | None:
        """
        Basic reproduction number for SIRS model.

        R0 = β / γ

        :return: Basic reproduction number, or None if parameters not set
        """
        if self.param_values and "beta" in self.param_values and "gamma" in self.param_values:
            return float(self.param_values["beta"] / self.param_values["gamma"])
        return None

    def _model(self, t: float, y: list[float], params: dict[str, float]) -> list[float]:
        S, I, R = y
        beta, gamma, xi, N = params["beta"], params["gamma"], params["xi"], params["N"]
        return [
            -beta * S * I / N + xi * R,
            beta * S * I / N - gamma * I,
            gamma * I - xi * R,
        ]


class SEIR(ContinuousModel):
    """
    SEIR (Susceptible-Exposed-Infectious-Removed) Model.

    A model with an exposed (latent) compartment for diseases with incubation period.

    State Variables:
        - S: Susceptible individuals
        - E: Exposed (infected but not yet infectious) individuals
        - I: Infectious individuals
        - R: Removed individuals

    Parameters:
        - beta (β): Transmission rate
        - gamma (γ): Recovery rate
        - epsilon (ε): Incubation rate (1 / average latent period)

    Equations:

        dS/dt = -βSI/N
        dE/dt = βSI/N - εE
        dI/dt = εE - γI
        dR/dt = γI

    Basic Reproduction Number:
        R₀ = β/γ

    Note:
        The exposed compartment E represents individuals who have been infected
        but are not yet infectious (latent period).
    """

    def __init__(self):
        super().__init__()
        self.state_variables = OrderedDict(
            {"S": "Susceptible", "E": "Exposed", "I": "Infectious", "R": "Removed"}
        )
        self.parameters = OrderedDict(
            {"beta": r"$\beta$", "gamma": r"$\gamma$", "epsilon": r"$\epsilon$"}
        )
        self.model_type = "SEIR"

    @property
    def diagram(self) -> str:
        """Mermaid diagram of the compartmental model"""
        return r"""flowchart LR
         
S(Susceptible) -->|$$\beta$$| E(Exposed)
E -->|$$\epsilon$$| I(Infectious)
I -->|$$\gamma$$| R(Removed)
"""

    @property
    def R0(self) -> float | None:
        """
        Basic reproduction number for SEIR model.

        R0 = β / γ

        :return: Basic reproduction number, or None if parameters not set
        """
        if self.param_values and "beta" in self.param_values and "gamma" in self.param_values:
            return float(self.param_values["beta"] / self.param_values["gamma"])
        return None

    def _model(self, t: float, y: list[float], params: dict[str, float]) -> list[float]:
        S, E, I, R = y
        beta, gamma, epsilon, N = (
            params["beta"],
            params["gamma"],
            params["epsilon"],
            params["N"],
        )
        return [
            -beta * S * I / N,
            beta * S * I / N - epsilon * E,
            epsilon * E - gamma * I,
            gamma * I,
        ]


class SEQIAHR(ContinuousModel):
    def __init__(self):
        super().__init__()
        self.state_variables = OrderedDict(
            {
                "S": "Susceptible",
                "E": "Exposed",
                "I": "Infectious",
                "A": "Asymptomatic",
                "H": "Hospitalized",
                "R": "Removed",
                "C": "Cumulative hospitalizations",
                "D": "Cumulative deaths",
            }
        )
        self.parameters = OrderedDict(
            {
                "chi": r"$\chi$",
                "phi": r"$\phi$",
                "beta": r"$\beta$",
                "rho": r"$\rho$",
                "delta": r"$\delta$",
                "gamma": r"$\gamma$",
                "alpha": r"$\alpha$",
                "mu": r"$\mu$",
                "p": "$p$",
                "q": "$q$",
                "r": "$r$",
            }
        )
        self.model_type = "SEQIAHR"

    @property
    def diagram(self) -> str:
        """Mermaid diagram of the compartmental model"""
        return r"""flowchart LR
         
S(Susceptible) -->|$$\beta$$| E(Exposed)
E -->|"$$\alpha(1-p)$$"| I(Infectious)
E -->|$$\alpha p$$| A(Asymptomatic)
I -->|$$\phi$$| H(Hospitalized)
I -->|$$\delta$$| R(Removed)
A -->|$$\gamma$$| R
H -->|$$\rho$$| R
H -->|$$\mu$$| D(Deaths)
"""

    def _model(self, t: float, y: list[float], params: dict[str, float]) -> list[float]:
        S, E, I, A, H, R, C, D = y
        chi, phi, beta, rho, delta, gamma, alpha, mu, p, q, r, N = params.values()
        lamb = beta * (I + A)
        # Turns on Quarantine on day q and off on day q+r
        chi *= ((1 + np.tanh(t - q)) / 2) * ((1 - np.tanh(t - (q + r))) / 2)
        return [
            -lamb * ((1 - chi) * S),  # dS/dt
            lamb * ((1 - chi) * S) - alpha * E,  # dE/dt
            (1 - p) * alpha * E - delta * I - phi * I,  # dI/dt
            p * alpha * E - gamma * A,
            phi * I - (rho + mu) * H,  # dH/dt
            delta * I + rho * H + gamma * A,  # dR/dt
            phi * I,  # (1-p)*alpha*E+ p*alpha*E # Hospit. acumuladas
            mu * H,  # Morte acumuladas
        ]


class Dengue4Strain(ContinuousModel):
    """
    Dengue 4 strain model
    """

    def __init__(self):
        super().__init__()
        self.state_variables = OrderedDict(
            {
                "S": "Susceptible",
                "I_1": "Infectious 1",
                "I_2": "Infectious 2",
                "I_3": "Infectious 3",
                "I_4": "Infectious 4",
                "R_1": "Removed 1",
                "R_2": "Removed 2",
                "R_3": "Removed 3",
                "R_4": "Removed 4",
                "I_12": "Infectious 1 and 2",
                "I_13": "Infectious 1 and 3",
                "I_14": "Infectious 1 and 4",
                "I_21": "Infectious 2 and 1",
                "I_23": "Infectious 2 and 3",
                "I_24": "Infectious 2 and 4",
                "I_31": "Infectious 3 and 1",
                "I_32": "Infectious 3 and 2",
                "I_34": "Infectious 3 and 4",
                "I_41": "Infectious 4 and 1",
                "I_42": "Infectious 4 and 2",
                "I_43": "Infectious 4 and 3",
                "R_12": "Removed 1 and 2",
                "R_13": "Removed 1 and 3",
                "R_14": "Removed 1 and 4",
                "R_23": "Removed 2 and 3",
                "R_24": "Removed 2 and 4",
                "R_34": "Removed 3 and 4",
                "I_231": "Infectious 2 and 3 and 1",
                "I_241": "Infectious 2 and 4 and 1",
                "I_341": "Infectious 3 and 4 and 1",
                "I_132": "Infectious 1 and 3 and 2",
                "I_142": "Infectious 1 and 4 and 2",
                "I_342": "Infectious 3 and 4 and 2",
                "I_123": "Infectious 1 and 2 and 3",
                "I_143": "Infectious 1 and 4 and 3",
                "I_243": "Infectious 2 and 4 and 3",
                "I_124": "Infectious 1 and 2 and 4",
                "I_134": "Infectious 1 and 3 and 4",
                "I_234": "Infectious 2 and 3 and 4",
                "R_123": "Removed 1 and 2 and 3",
                "R_124": "Removed 1 and 2 and 4",
                "R_134": "Removed 1 and 3 and 4",
                "R_234": "Removed 2 and 3 and 4",
                "I_1234": "Infectious 1 and 2 and 3 and 4",
                "I_1243": "Infectious 1 and 2 and 4 and 3",
                "I_1342": "Infectious 1 and 3 and 4 and 2",
                "I_2341": "Infectious 2 and 3 and 4 and 1",
                "R_1234": "Removed 1 and 2 and 3 and 4",
            }
        )
        self.parameters = OrderedDict(
            {
                "beta": r"$\beta$",  #  transmission rate
                "N": r"$N$",  #  total population
                "delta": r"$\delta$",  #  cross-immunity protection
                "mu": r"$\mu$",  #  mortality rate
                "sigma": r"$\sigma$",  #  recovery rate
                "im": r"$i_m$",  #  imported cases
            }
        )
        self.model_type = "Dengue4Strain"

    @property
    def diagram(self) -> str:
        """Mermaid diagram of the compartmental model"""
        return r"""flowchart LR
    S(Susceptible) -->|$$\beta$$| I1(I 1)
    S -->|$$\beta$$| I2(I 2) 
    S -->|$$\beta$$| I3(I 3)
    S -->|$$\beta$$| I4(I 4)
    
    I1 -->|$$\sigma$$| R1(R 1)
    I2 -->|$$\sigma$$| R2(R 2)
    I3 -->|$$\sigma$$| R3(R 3)
    I4 -->|$$\sigma$$| R4(R 4)
    
    R1 -->|$$\delta$$| I12(I 1+2)
    R1 -->|$$\delta$$| I13(I 1+3)
    R1 -->|$$\delta$$| I14(I 1+4)
    
    R2 -->|$$\delta$$| I21(I 2+1)
    R2 -->|$$\delta$$| I23(I 2+3)
    R2 -->|$$\delta$$| I24(I 2+4)
    
    R3 -->|$$\delta$$| I31(I 3+1)
    R3 -->|$$\delta$$| I32(I 3+2)
    R3 -->|$$\delta$$| I34(I 3+4)
    
    R4 -->|$$\delta$$| I41(I 4+1)
    R4 -->|$$\delta$$| I42(I 4+2)
    R4 -->|$$\delta$$| I43(I 4+3)
    
    I12 -->|$$\sigma$$| R12(R 1+2)
    I13 -->|$$\sigma$$| R13(R 1+3)
    I14 -->|$$\sigma$$| R14(R 1+4)
    
    I21 -->|$$\sigma$$| R12(R 1+2)
    I23 -->|$$\sigma$$| R23(R 2+3)
    I24 -->|$$\sigma$$| R24(R 2+4)
    
    I31 -->|$$\sigma$$| R13(R 1+3)
    I32 -->|$$\sigma$$| R23(R 2+3)
    I34 -->|$$\sigma$$| R34(R 3+4)
    
    I41 -->|$$\sigma$$| R14(R 1+4)
    I42 -->|$$\sigma$$| R24(R 2+4)
    I43 -->|$$\sigma$$| R34(R 3+4)
    
    R12 -->|$$\delta$$| I123(I 1+2+3)
    R13 -->|$$\delta$$| I132(I 1+3+2)
    R14 -->|$$\delta$$| I142(I 1+4+2)
    
    R12 -->|$$\delta$$| I213(I 2+1+3)
    R23 -->|$$\delta$$| I231(I 2+3+1)
    R24 -->|$$\delta$$| I241(I 2+4+1)
     
    R24 -->|$$\delta$$| I243(I 2+4+3)
    R34 -->|$$\delta$$| I341(I 3+4+1)
    R34 -->|$$\delta$$| I342(I 3+4+2)
     
    R23 -->|$$\delta$$| I234(I 2+3+4)
    R14 -->|$$\delta$$| I143(I 1+4+3)
    R13 -->|$$\delta$$| I134(I 1+3+4)
    
    R12 -->|$$\delta$$| I124(I 1+2+4)
    
    I123 -->|$$\sigma$$| R123(R 1+2+3)
    I132 -->|$$\sigma$$| R123(R 1+2+3)
    I124 -->|$$\sigma$$| R124(R 1+2+4)
    
    I142 -->|$$\sigma$$| R124(R 1+2+4)
    I143 -->|$$\sigma$$| R134(R 1+3+4)
    I134 -->|$$\sigma$$| R134(R 1+3+4)
    
    I234 -->|$$\sigma$$| R234(R 2+3+4)
    I243 -->|$$\sigma$$| R234(R 2+3+4)
    I341 -->|$$\sigma$$| R134(R 1+3+4)
    
    I342 -->|$$\sigma$$| R234(R 2+3+4)
    I231 -->|$$\sigma$$| R123(R 1+2+3)
    I241 -->|$$\sigma$$| R124(R 1+2+4)
    
    R123 -->|$$\delta$$| I1234(I 1+2+3+4)
    R124 -->|$$\delta$$| I1243(I 1+2+4+3)
    R134 -->|$$\delta$$| I1342(I 1+3+4+2)
    R234 -->|$$\delta$$| I2341(I 2+3+4+1)
    
    I1234 -->|$$\sigma$$| R1234(R 1+2+3+4)
    I1243 -->|$$\sigma$$| R1234(R 1+2+3+4)
    I1342 -->|$$\sigma$$| R1234(R 1+2+3+4)
    I2341 -->|$$\sigma$$| R1234(R 1+2+3+4)
    
    classDef strain1 fill:#ffcccc,stroke:#ff0000
    classDef strain2 fill:#ccffcc,stroke:#00ff00
    classDef strain3 fill:#ccccff,stroke:#0000ff
    classDef strain4 fill:#ffccff,stroke:#ff00ff
    
    class I1,I21,I31,I41,I231,I241,I341,I2341 strain1
    class I2,I12,I32,I42,I132,I213,I342,I142,I1342 strain2
    class I3,I13,I23,I43,I123,I143,I243,I1243 strain3
    class I4,I14,I24,I34,I142,I124,I134,I342,I124,I134,I234,I1234 strain4;
"""

    def _model(self, t: float, y: list[float], params: dict[str, Any]) -> list[float]:
        (
            S,
            I_1,
            I_2,
            I_3,
            I_4,
            R_1,
            R_2,
            R_3,
            R_4,
            I_12,
            I_13,
            I_14,
            I_21,
            I_23,
            I_24,
            I_31,
            I_32,
            I_34,
            I_41,
            I_42,
            I_43,
            R_12,
            R_13,
            R_14,
            R_23,
            R_24,
            R_34,
            I_231,
            I_241,
            I_341,
            I_132,
            I_142,
            I_342,
            I_123,
            I_143,
            I_243,
            I_124,
            I_134,
            I_234,
            R_123,
            R_124,
            R_134,
            R_234,
            I_1234,
            I_1243,
            I_1342,
            I_2341,
            R_1234,
        ) = y

        beta, N, delta, mu, sigma, im = (
            params["beta"],
            params["N"],
            params["delta"],
            params["mu"],
            params["sigma"],
            params["im"],
        )
        m1 = lambda t: 1 if (im[0] < t < (im[0] + 5)) else 0
        m2 = lambda t: 1 if (im[1] < t < (im[1] + 5)) else 0
        m3 = lambda t: 1 if (im[2] < t < (im[2] + 5)) else 0
        m4 = lambda t: 1 if (im[3] < t < (im[3] + 5)) else 0
        return [
            -beta
            * S
            * (
                I_1
                + I_21
                + I_31
                + I_41
                + I_231
                + I_241
                + I_341
                + I_2341
                + I_2
                + I_12
                + I_32
                + I_42
                + I_132
                + I_142
                + I_342
                + I_1342
                + I_3
                + I_13
                + I_23
                + I_43
                + I_123
                + I_143
                + I_243
                + I_1243
                + I_4
                + I_14
                + I_24
                + I_34
                + I_124
                + I_134
                + I_234
                + I_1234
            )
            + mu * N
            - mu * S,  # S
            m1(t)
            + beta * S * (I_1 + I_21 + I_31 + I_41 + I_231 + I_241 + I_341 + I_2341)
            - sigma * I_1
            - mu * I_1,  # I_1
            m2(t)
            + beta * S * (I_2 + I_12 + I_32 + I_42 + I_132 + I_142 + I_342 + I_1342)
            - sigma * I_2
            - mu * I_2,  # I_2
            m3(t)
            + beta * S * (I_3 + I_13 + I_23 + I_43 + I_123 + I_143 + I_243 + I_1243)
            - sigma * I_3
            - mu * I_3,  # I_3
            m4(t)
            + beta * S * (I_4 + I_14 + I_24 + I_34 + I_124 + I_134 + I_234 + I_1234)
            - sigma * I_4
            - mu * I_4,  # I_4
            sigma * I_1
            - beta
            * delta
            * R_1
            * (
                I_2
                + I_12
                + I_32
                + I_42
                + I_132
                + I_142
                + I_342
                + I_1342
                + I_3
                + I_13
                + I_23
                + I_43
                + I_123
                + I_143
                + I_243
                + I_1243
                + I_4
                + I_14
                + I_24
                + I_34
                + I_124
                + I_134
                + I_234
                + I_1234
            )
            - mu * R_1,  # R_1
            sigma * I_2
            - beta
            * delta
            * R_2
            * (
                I_1
                + I_21
                + I_31
                + I_41
                + I_231
                + I_241
                + I_341
                + I_2341
                + I_3
                + I_13
                + I_23
                + I_43
                + I_123
                + I_143
                + I_243
                + I_1243
                + I_4
                + I_14
                + I_24
                + I_34
                + I_124
                + I_134
                + I_234
                + I_1234
            )
            - mu * R_2,  # R_2
            sigma * I_3
            - beta
            * delta
            * R_3
            * (
                I_1
                + I_21
                + I_31
                + I_41
                + I_231
                + I_241
                + I_341
                + I_2341
                + I_2
                + I_12
                + I_32
                + I_42
                + I_132
                + I_142
                + I_342
                + I_1342
                + I_4
                + I_14
                + I_24
                + I_34
                + I_124
                + I_134
                + I_234
                + I_1234
            )
            - mu * R_3,  # R_3
            sigma * I_4
            - beta
            * delta
            * R_4
            * (
                I_1
                + I_21
                + I_31
                + I_41
                + I_231
                + I_241
                + I_341
                + I_2341
                + I_2
                + I_12
                + I_32
                + I_42
                + I_132
                + I_142
                + I_342
                + I_1342
                + I_3
                + I_13
                + I_23
                + I_43
                + I_123
                + I_143
                + I_243
                + I_1243
            )
            - mu * R_4,  # R_4
            beta * delta * R_1 * (I_2 + I_12 + I_32 + I_42 + I_132 + I_142 + I_342 + I_1342)
            - sigma * I_12
            - mu * I_12,  # I_12
            beta * delta * R_1 * (I_3 + I_13 + I_23 + I_43 + I_123 + I_143 + I_243 + I_1243)
            - sigma * I_13
            - mu * I_13,  # I_13
            beta * delta * R_1 * (I_4 + I_14 + I_24 + I_34 + I_124 + I_134 + I_234 + I_1234)
            - sigma * I_14
            - mu * I_14,  # I_14
            beta * delta * R_2 * (I_1 + I_21 + I_31 + I_41 + I_231 + I_241 + I_341 + I_2341)
            - sigma * I_21
            - mu * I_21,  # I_21
            beta * delta * R_2 * (I_3 + I_13 + I_23 + I_43 + I_123 + I_143 + I_243 + I_1243)
            - sigma * I_23
            - mu * I_23,  # I_23
            beta * delta * R_2 * (I_4 + I_14 + I_24 + I_34 + I_124 + I_134 + I_234 + I_1234)
            - sigma * I_24
            - mu * I_24,  # I_24
            beta * delta * R_3 * (I_1 + I_21 + I_31 + I_41 + I_231 + I_241 + I_341 + I_2341)
            - sigma * I_31
            - mu * I_31,  # I_31
            beta * delta * R_3 * (I_2 + I_12 + I_32 + I_42 + I_132 + I_142 + I_342 + I_1342)
            - sigma * I_32
            - mu * I_32,  # I_32
            beta * delta * R_3 * (I_4 + I_14 + I_24 + I_34 + I_124 + I_134 + I_234 + I_1234)
            - sigma * I_34
            - mu * I_34,  # I_34
            beta * delta * R_4 * (I_1 + I_21 + I_31 + I_41 + I_231 + I_241 + I_341 + I_2341)
            - sigma * I_41
            - mu * I_41,  # I_41
            beta * delta * R_4 * (I_2 + I_12 + I_32 + I_42 + I_132 + I_142 + I_342 + I_1342)
            - sigma * I_42
            - mu * I_42,  # I_42
            beta * delta * R_4 * (I_3 + I_13 + I_23 + I_43 + I_123 + I_143 + I_243 + I_1243)
            - sigma * I_43
            - mu * I_43,  # I_43
            sigma * (I_12 + I_21)
            - beta
            * delta
            * R_12
            * (
                I_3
                + I_13
                + I_23
                + I_43
                + I_123
                + I_143
                + I_243
                + I_1243
                + I_4
                + I_14
                + I_24
                + I_34
                + I_124
                + I_134
                + I_234
                + I_1234
            )
            - mu * R_12,  # R_12
            sigma * (I_13 + I_31)
            - beta
            * delta
            * R_13
            * (
                I_2
                + I_12
                + I_32
                + I_42
                + I_132
                + I_142
                + I_342
                + I_1342
                + I_4
                + I_14
                + I_24
                + I_34
                + I_124
                + I_134
                + I_234
                + I_1234
            )
            - mu * R_13,  # R_13
            sigma * (I_14 + I_41)
            - beta
            * delta
            * R_14
            * (
                I_2
                + I_12
                + I_32
                + I_42
                + I_132
                + I_142
                + I_342
                + I_1342
                + I_3
                + I_13
                + I_23
                + I_43
                + I_123
                + I_143
                + I_243
                + I_1243
            )
            - mu * R_14,  # R_14
            sigma * (I_23 + I_32)
            - beta
            * delta
            * R_23
            * (
                I_1
                + I_21
                + I_31
                + I_41
                + I_231
                + I_241
                + I_341
                + I_2341
                + I_4
                + I_14
                + I_24
                + I_34
                + I_124
                + I_134
                + I_234
                + I_1234
            )
            - mu * R_23,  # R_23
            sigma * (I_24 + I_42)
            - beta
            * delta
            * R_24
            * (
                I_1
                + I_21
                + I_31
                + I_41
                + I_231
                + I_241
                + I_341
                + I_2341
                + I_3
                + I_13
                + I_23
                + I_43
                + I_123
                + I_143
                + I_243
                + I_1243
            )
            - mu * R_24,  # R_24
            sigma * (I_34 + I_43)
            - beta
            * delta
            * R_34
            * (
                I_1
                + I_21
                + I_31
                + I_41
                + I_231
                + I_241
                + I_341
                + I_2341
                + I_2
                + I_12
                + I_32
                + I_42
                + I_132
                + I_142
                + I_342
                + I_1342
            )
            - mu * R_34,  # R_34
            beta * delta * R_23 * (I_1 + I_21 + I_31 + I_41 + I_231 + I_241 + I_341 + I_2341)
            - sigma * I_231
            - mu * I_231,  # I_231
            beta * delta * R_24 * (I_1 + I_21 + I_31 + I_41 + I_231 + I_241 + I_341 + I_2341)
            - sigma * I_241
            - mu * I_241,  # I_241
            beta * delta * R_34 * (I_1 + I_21 + I_31 + I_41 + I_231 + I_241 + I_341 + I_2341)
            - sigma * I_341
            - mu * I_341,  # I_341
            beta * delta * R_13 * (I_2 + I_12 + I_32 + I_42 + I_132 + I_142 + I_342 + I_1342)
            - sigma * I_132
            - mu * I_132,  # I_132
            beta * delta * R_14 * (I_2 + I_12 + I_32 + I_42 + I_132 + I_142 + I_342 + I_1342)
            - sigma * I_142
            - mu * I_142,  # I_142
            beta * delta * R_34 * (I_2 + I_12 + I_32 + I_42 + I_132 + I_142 + I_342 + I_1342)
            - sigma * I_342
            - mu * I_342,  # I_342
            beta * delta * R_12 * (I_3 + I_13 + I_23 + I_43 + I_123 + I_143 + I_243 + I_1243)
            - sigma * I_123
            - mu * I_123,  # I_123
            beta * delta * R_14 * (I_3 + I_13 + I_23 + I_43 + I_123 + I_143 + I_243 + I_1243)
            - sigma * I_143
            - mu * I_143,  # I_143
            beta * delta * R_24 * (I_3 + I_13 + I_23 + I_43 + I_123 + I_143 + I_243 + I_1243)
            - sigma * I_243
            - mu * I_243,  # I_243
            beta * delta * R_12 * (I_4 + I_14 + I_24 + I_34 + I_124 + I_134 + I_234 + I_1234)
            - sigma * I_124
            - mu * I_124,  # I_124
            beta * delta * R_13 * (I_4 + I_14 + I_24 + I_34 + I_124 + I_134 + I_234 + I_1234)
            - sigma * I_134
            - mu * I_134,  # I_134
            beta * delta * R_23 * (I_4 + I_14 + I_24 + I_34 + I_124 + I_134 + I_234 + I_1234)
            - sigma * I_234
            - mu * I_234,  # I_234
            sigma * (I_123 + I_132 + I_231)
            - beta * delta * R_123 * (I_4 + I_14 + I_24 + I_34 + I_124 + I_134 + I_234 + I_1234)
            - mu * R_123,  # R_123
            sigma * (I_124 + I_241 + I_142)
            - beta * delta * R_124 * (I_3 + I_13 + I_23 + I_43 + I_123 + I_143 + I_243 + I_1243)
            - mu * R_124,  # R_124
            sigma * (I_134 + I_341 + I_143)
            - beta * delta * R_134 * (I_2 + I_12 + I_32 + I_42 + I_132 + I_142 + I_342 + I_1342)
            - mu * R_134,  # R_134
            sigma * (I_234 + I_342 + I_243)
            - beta * delta * R_234 * (I_2 + I_12 + I_32 + I_42 + I_132 + I_142 + I_342 + I_1342)
            - mu * R_234,  # R_234
            beta * delta * R_123 * (I_4 + I_14 + I_24 + I_34 + I_124 + I_134 + I_234 + I_1234)
            - sigma * I_1234
            - mu * I_1234,  # I_1234
            beta * delta * R_124 * (I_3 + I_13 + I_23 + I_43 + I_123 + I_143 + I_243 + I_1243)
            - sigma * I_1243
            - mu * I_1243,  # I_1243
            beta * delta * R_134 * (I_2 + I_12 + I_32 + I_42 + I_132 + I_142 + I_342 + I_1342)
            - sigma * I_1342
            - mu * I_1342,  # I_1342
            beta * delta * R_234 * (I_1 + I_21 + I_31 + I_41 + I_231 + I_241 + I_341 + I_2341)
            - sigma * I_2341
            - mu * I_2341,  # I_2341
            sigma * (I_1234 + I_1243 + I_1342 + I_2341) - mu * R_1234,  # R_1234
        ]


###################################################

### Malaria SIR/SEI model


class SIRSEI(ContinuousModel):
    """
    SIR–SEI Vector-Borne Disease Model.

    This base model is based on the paper
    'Modelling Climate Change and on Malaria Transmission (Parham and Michael, 2010)'
    and inspired by the Trajetórias Project developed by SinBiose/CNPq.

    Further development will be done to include the effects of deforestation
    and forest fires on mosquito habitat and transmission dynamics.

    Later versions of the model replaced the use of temperature and precipitation
    functions for real data from the Mosqlimate datastore.

    Humans:
        Sh : Susceptible Humans
        Ih : Infectious Humans
        Rh : Recovered Humans

    Mosquitoes:
        Sv : Susceptible Mosquitoes
        Ev : Exposed Mosquitoes
        Iv : Infectious Mosquitoes

    Transmission:
        Mosquito → Human: a * b2
        Human → Mosquito: a * b1

    Climate Forcing:
        Temperature and rainfall drive mosquito demography and development rates.

    Temperature forcing:
        T(t) = T1 + T2 * cos(omega1 * t + phi1)

    Rainfall forcing:
        R(t) = R1 + R2 * cos(omega2 * t + phi2)

    Basic Reproduction Number:

        R0 = sqrt( (a² b1 b2 b3) / ((b3 + l + μ) γ μ) )

    where:
        a   : biting rate
        b1  : human → mosquito transmission probability
        b2  : mosquito → human transmission probability
        b3  : mosquito incubation rate
        μ  : mosquito mortality
        l   : mosquito latent-stage mortality
        γ   : human recovery rate
    """

    def __init__(self):
        super().__init__()
        self.state_variables = OrderedDict(
            {
                "Sh": "Susceptible Humans",
                "Ih": "Infectious Humans",
                "Rh": "Recovered Humans",
                "Sv": "Susceptible Mosquitoes",
                "Ev": "Exposed Mosquitoes",
                "Iv": "Infectious Mosquitoes",
            }
        )
        self.parameters = OrderedDict(
            {
                "b1": r"$b_1$",
                "b2": r"$b_2$",
                "gamma": r"$\gamma$",
                "mu_H": r"$\mu_H$",
                "T1": r"$T_1$",
                "T2": r"$T_2$",
                "omega1": r"$\omega_1$",
                "phi1": r"$\phi_1$",
                "R1": r"$R_1$",
                "R2": r"$R_2$",
                "omega2": r"$\omega_2$",
                "phi2": r"$\phi_2$",
                "BE": r"$B_E$",
                "pME": r"$p_{ME}$",
                "pML": r"$p_{ML}$",
                "pMP": r"$p_{MP}$",
                "tauE": r"$\tau_E$",
                "tauP": r"$\tau_P$",
                "RL": r"$R_L$",
                "DD": r"$DD$",
                "Tmin": r"$T_{min}$",
                "A": r"$A$",
                "B": r"$B$",
                "C": r"$C$",
                "D1": r"$D_1$",
                "c1": r"$c_1$",
                "c2": r"$c_2$",
                "T_prime": r"$T'$",
            }
        )
        self.model_type = "SIR-SEI"

    @property
    def diagram(self) -> str:
        """Mermaid diagram of the compartmental model"""
        return r"""flowchart LR
        subgraph Humans
        Sh(S_h)
        Ih(I_h)
        Rh(R_h)
        end

        subgraph Mosquitoes
        Sv(S_v)
        Ev(E_v)
        Iv(I_v)
        end

        Iv -->|$$a b_2$$| Sh
        Sh -->|$$a b_2 I_v/N$$| Ih
        Ih -->|$$\gamma$$| Rh

        Ih -->|$$a b_1$$| Sv
        Sv -->|$$a b_1 I_h/N$$| Ev
        Ev -->|$$b_3$$| Iv
        """

    @property
    def R0(self) -> float | None:
        """
        Basic reproduction number for the SIR-SEI model.

        R0 = sqrt((a^2 * b1 * b2 * b3) / ((b3 + l + mu) * gamma * mu))

        :return: Basic reproduction number, or None if parameters not set
        """

        if not self.param_values:
            return None

        p = self.param_values

        required = ["b1", "b2", "gamma"]

        if not all(k in p for k in required):
            return None

        # Use reference temperature for evaluation
        T = p["T1"]

        a = (T - p["T_prime"]) / p["D1"]
        p_survive = np.exp(-1 / (p["A"] * T**2 + p["B"] * T + p["C"]))
        mu = -np.log(p_survive)

        tau_M = p["DD"] / (T - p["Tmin"])
        b3 = 1 / tau_M

        l = p_survive**tau_M

        return float(np.sqrt((a**2 * p["b1"] * p["b2"] * b3) / ((b3 + l + mu) * p["gamma"] * mu)))

    def R0_t(self, t: float) -> float | None:
        """
        Time-dependent reproduction number based on climate forcing.
        """

        if not self.param_values:
            return None

        p = self.param_values

        T = p["T1"] + p["T2"] * np.cos(p["omega1"] * t + p["phi1"])

        a = (T - p["T_prime"]) / p["D1"]

        p_survive = np.exp(-1 / (p["A"] * T**2 + p["B"] * T + p["C"]))

        mu = -np.log(p_survive)

        tau_M = p["DD"] / (T - p["Tmin"])
        b3 = 1 / tau_M

        l = p_survive**tau_M

        return float(np.sqrt((a**2 * p["b1"] * p["b2"] * b3) / ((b3 + l + mu) * p["gamma"] * mu)))

    def _model(self, t: float, y: list[float], p: dict[str, float]) -> list[float]:

        Sh, Ih, Rh, Sv, Ev, Iv = y

        N = Sh + Ih + Rh

        # Climate forcing
        T = p["T1"] + p["T2"] * np.cos(p["omega1"] * t + p["phi1"])
        R = p["R1"] + p["R2"] * np.cos(p["omega2"] * t + p["phi2"])

        # Temperature dependent quantities
        a = (T - p["T_prime"]) / p["D1"]

        p_survive = np.exp(-1 / (p["A"] * T**2 + p["B"] * T + p["C"]))

        mu = -np.log(p_survive)

        tau_M = p["DD"] / (T - p["Tmin"])

        b3 = 1 / tau_M

        l = p_survive**tau_M

        tau_L = 1 / (p["c1"] * T + p["c2"])

        # Rainfall dependent larval survival
        pL_R = (4 * p["pML"] / p["RL"] ** 2) * R * (p["RL"] - R)
        pL_R = max(pL_R, 0)

        pL_T = np.exp(-(p["c1"] * T + p["c2"]))

        p_L = pL_R * pL_T

        # Mosquito birth rate
        b = p["BE"] * p["pME"] * p_L * p["pMP"] / (p["tauE"] + tau_L + p["tauP"])

        b1 = p["b1"]
        b2 = p["b2"]
        gamma = p["gamma"]

        # Differential equations

        dSh = p["mu_H"] * N - a * b2 * (Iv / N) * Sh
        dIh = a * b2 * (Iv / N) * Sh - gamma * Ih
        dRh = gamma * Ih

        dSv = b - a * b1 * (Ih / N) * Sv - mu * Sv

        dEv = a * b1 * (Ih / N) * Sv - mu * Ev - b3 * Ev - l * Ev

        dIv = b3 * Ev - mu * Iv

        return [dSh, dIh, dRh, dSv, dEv, dIv]

    def plot(self, compartments=None, figsize=(12, 6)):
        """
        Plot selected compartments with proper time axis.

        Parameters:
        -----------
        compartments : list, optional
            List of compartment names to plot. If None, plots all compartments.
        figsize : tuple, optional
            Figure size (width, height)
        """
        if not self.traces or "time" not in self.traces:
            print(
                "No data available. Run the model first with: model(inits, trange, totpop, params)"
            )
            return

        t = self.traces["time"]

        if compartments is None:
            # Plot all compartments except 'time'
            compartments = [k for k in self.traces.keys() if k != "time"]

        plt.figure(figsize=figsize)
        for comp in compartments:
            if comp in self.traces:
                plt.plot(t, self.traces[comp], label=comp, linewidth=2)

        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.xlabel("Time (days)", fontsize=12)
        plt.ylabel("Population", fontsize=12)

        # Smart title based on which compartments are being plotted
        human_comps = ["Sh", "Ih", "Rh"]
        mosquito_comps = ["Sv", "Ev", "Iv"]

        # Check what we're plotting
        has_humans = any(comp in human_comps for comp in compartments)
        has_mosquitoes = any(comp in mosquito_comps for comp in compartments)

        if has_humans and has_mosquitoes:
            title = f"{self.model_type} Model - Full System"
        elif has_humans:
            title = f"{self.model_type} Model - Human Dynamics (SIR)"
        elif has_mosquitoes:
            title = f"{self.model_type} Model - Mosquito Dynamics (SEI)"
        else:
            title = f"{self.model_type} Model Results"

        plt.title(title, fontsize=14)
        plt.tight_layout()
        plt.show()

class SIRSEIData(ContinuousModel):
    """
    SIR–SEI Vector-Borne Disease Model with Real Climate Data.

    This model is based on the SIRSEI model but accepts real temperature and
    precipitation data through interpolation functions, instead of using
    sinusoidal approximations.

    Based on 'Modelling Climate Change and on Malaria Transmission
    (Parham and Michael, 2010)'.

    Humans:
        Sh : Susceptible Humans
        Ih : Infectious Humans
        Rh : Recovered Humans

    Mosquitoes:
        Sv : Susceptible Mosquitoes
        Ev : Exposed Mosquitoes
        Iv : Infectious Mosquitoes

    Transmission:
        Mosquito → Human: a * b2
        Human → Mosquito: a * b1

    Climate Data:
        Temperature and precipitation are provided as interpolation functions
        that map time to climate values, allowing the use of real observed data.

    Basic Reproduction Number:

        R0 = sqrt( (a² b1 b2 b3) / ((b3 + l + μ) γ μ) )

    where:
        a   : biting rate
        b1  : human → mosquito transmission probability
        b2  : mosquito → human transmission probability
        b3  : mosquito incubation rate
        μ  : mosquito mortality
        l   : mosquito latent-stage mortality
        γ   : human recovery rate
    """

    def __init__(self, temp_func=None, precip_func=None):
        super().__init__()
        self.state_variables = OrderedDict(
            {
                "Sh": "Susceptible Humans",
                "Ih": "Infectious Humans",
                "Rh": "Recovered Humans",
                "Sv": "Susceptible Mosquitoes",
                "Ev": "Exposed Mosquitoes",
                "Iv": "Infectious Mosquitoes",
            }
        )
        self.parameters = OrderedDict(
            {
                "b1": r"$b_1$",
                "b2": r"$b_2$",
                "gamma": r"$\gamma$",
                "mu_H": r"$\mu_H$",
                "BE": r"$B_E$",
                "pME": r"$p_{ME}$",
                "pML": r"$p_{ML}$",
                "pMP": r"$p_{MP}$",
                "tauE": r"$\tau_E$",
                "tauP": r"$\tau_P$",
                "c1": r"$c_1$",
                "c2": r"$c_2$",
                "D1": r"$D_1$",
                "RL": r"$R_L$",
                "DD": r"$DD$",
                "Tmin": r"$T_{min}$",
                "A": r"$A$",
                "B": r"$B$",
                "C": r"$C$",
                "T_prime": r"$T'$",
            }
        )
        self.model_type = "SIR-SEI-Data"
        self.temp_func = temp_func
        self.precip_func = precip_func

    def set_climate_functions(self, temp_func, precip_func):
        """
        Set the interpolation functions for temperature and precipitation.

        Parameters:
        -----------
        temp_func : callable
            Function that takes time (float) and returns temperature (float)
        precip_func : callable
            Function that takes time (float) and returns precipitation (float)
        """
        self.temp_func = temp_func
        self.precip_func = precip_func

    @property
    def diagram(self) -> str:
        """Mermaid diagram of the compartmental model"""
        return r"""flowchart LR
        subgraph Humans
        Sh(S_h)
        Ih(I_h)
        Rh(R_h)
        end

        subgraph Mosquitoes
        Sv(S_v)
        Ev(E_v)
        Iv(I_v)
        end

        Iv -->|$$a b_2$$| Sh
        Sh -->|$$a b_2 I_v/N$$| Ih
        Ih -->|$$\gamma$$| Rh

        Ih -->|$$a b_1$$| Sv
        Sv -->|$$a b_1 I_h/N$$| Ev
        Ev -->|$$b_3$$| Iv
        """

    def R0_t(self, t: float) -> float | None:
        """
        Time-dependent reproduction number based on climate data.

        Uses real temperature data from temp_func to compute R0.

        Returns:
            float: Time-dependent R0 value
        """
        if not hasattr(self, "param_values") or not self.param_values:
            return None

        p = self.param_values

        if self.temp_func is not None:
            T = float(self.temp_func(t))
        else:
            return None

        a = (T - p["T_prime"]) / p["D1"]

        p_survive = np.exp(-1 / (p["A"] * T**2 + p["B"] * T + p["C"]))
        p_survive = np.clip(p_survive, 0, 1)

        mu = -np.log(p_survive)
        mu = max(mu, 1e-10)

        tau_M = p["DD"] / (T - p["Tmin"])
        if tau_M <= 0:
            tau_M = 1.0
        b3 = 1 / tau_M

        l = p_survive**tau_M
        l = np.clip(l, 0, 1)

        return float(np.sqrt((a**2 * p["b1"] * p["b2"] * b3) / ((b3 + l + mu) * p["gamma"] * mu)))

    def _model(self, t: float, y: list[float], params: dict[str, float]) -> list[float]:
        """
        Compute derivatives for the SIRSEI model with real climate data.

        Parameters:
        -----------
        t : float
            Current time point
        y : list[float]
            State variables [Sh, Ih, Rh, Sv, Ev, Iv]
        params : dict[str, float]
            Model parameters

        Returns:
        --------
        list[float]
            Derivatives [dSh, dIh, dRh, dSv, dEv, dIv]
        """
        Sh, Ih, Rh, Sv, Ev, Iv = y
        N = Sh + Ih + Rh

        if self.temp_func is not None and self.precip_func is not None:
            T = float(self.temp_func(t))
            R = float(self.precip_func(t))
        else:
            raise ValueError(
                "Temperature and precipitation functions must be provided. "
                "Use set_climate_functions() or provide them in the constructor."
            )

        a = (T - params["T_prime"]) / params["D1"]
        a = max(a, 0)

        p_survive = np.exp(-1 / (params["A"] * T**2 + params["B"] * T + params["C"]))
        p_survive = np.clip(p_survive, 0, 1)

        mu = -np.log(p_survive)
        mu = max(mu, 1e-10)

        tau_M = params["DD"] / (T - params["Tmin"])
        if tau_M <= 0:
            tau_M = 1.0
        b3 = 1 / tau_M

        l = p_survive**tau_M
        l = np.clip(l, 0, 1)

        tau_L = 1 / (params["c1"] * T + params["c2"])
        if tau_L <= 0:
            tau_L = 1.0

        pL_R = (4 * params["pML"] / params["RL"] ** 2) * R * (params["RL"] - R)
        pL_R = max(pL_R, 0)

        pL_T = np.exp(-(params["c1"] * T + params["c2"]))
        p_L = pL_R * pL_T
        p_L = np.clip(p_L, 0, 1)

        b = (
            params["BE"]
            * params["pME"]
            * p_L
            * params["pMP"]
            / (params["tauE"] + tau_L + params["tauP"])
        )

        b1 = params["b1"]
        b2 = params["b2"]
        gamma = params["gamma"]

        dSh = params["mu_H"] * N - a * b2 * (Iv / N) * Sh
        dIh = a * b2 * (Iv / N) * Sh - gamma * Ih
        dRh = gamma * Ih

        dSv = b - a * b1 * (Ih / N) * Sv - mu * Sv

        dEv = a * b1 * (Ih / N) * Sv - mu * Ev - b3 * Ev - l * Ev

        dIv = b3 * Ev - mu * Iv

        return [dSh, dIh, dRh, dSv, dEv, dIv]


class SIR2Strain(ContinuousModel):
    """
    SIR (Susceptible-Infectious-Removed) Model with two strains.

    A compartmental model for infectious disease dynamics with two strains.
    Adapted from: https://doi.org/10.1016/j.jtbi.2011.08.043

    State Variables:
        - S: Susceptible individuals
        - S1: Susceptible individuals with a previous infection with strain 1
        - S2: Susceptible individuals with a previous infection with strain 2
        - I1: Infectious individuals (first infection) with strain 1
        - I21: Infectious individuals (second infection) with strain 1
        - I2: Infectious individuals (first infection) with strain 2
        - I12: Infectious individuals (second infection) with strain 2
        - R1: Recovered individuals from the first infection with strain 1
        - R2: Recovered individuals from the first infection with strain 2
        - R: Recovered individuals from the secondary infection

    Parameters:
        - beta (β): Infection rate
        - gamma (γ): Recovery rate (1 / average infectious period)
        - mu (μ): Birth and death rate
        - rho (ρ): Ratio of secondary infections contributing to force of infection (adimensional)
        - phi (Φ): Import parameter (adimensional)
        - alpha (α): Temporary cross-immunity rate (1 / average corss-immunity duration period)

    Equations:
        dS/dt = - β/N S (I1 + ρ N + Φ I21) - β/N S(I2 + ρ N + Φ I12) + μ (N-S)
        dI1/dt = β/N S (I1 + ρ N + Φ I21) - (γ + μ) I1
        dI2/dt = β/N S (I2 + ρ N + Φ I12) - (γ + μ) I2
        dR1/dt = γ I1 - (α + μ) R1
        dR2/dt = γ I2 - (α + μ) R2
        dS1/dt = - β/N S1 (I2 + ρ N + Φ I12) + α R1 - μ S1
        dS2/dt = - β/N S2 (I1 + ρ N + Φ I21) + α R2 - μ S2
        dI12/dt = β/N S1 (I2 + ρ N + Φ I12) - (γ + μ) I12
        dI21/dt = β/N S2 (I1 + ρ N + Φ I21) - (γ + μ) I21
        dR/dt = γ (I12 + I21) - μ R

    Example:
    model = SIR2Strain()
    model([9990, 4, 6, 0, 0, 0, 0, 0, 0, 0], [0, 100], 10000, {'beta': 2, 'gamma': 1/52, 'mu': 1/65, 'rho': 0.001, 'phi': 0.2, 'alpha': 1/2})
    model.plot_traces()

    """

    def __init__(self):
        super().__init__()
        self.state_variables = OrderedDict(
            {
                "S": "Susceptible",
                "I_1": "Infectious 1",
                "I_2": "Infectious 2",
                "R_1": "Removed 1",
                "R_2": "Removed 2",
                "S1": "Susceptible with a previous infection with strain 1, i.e., susceptible only to strain 2",
                "S2": "Susceptible with a previous infection with strain 2, i.e., susceptible only to strain 1",
                "I_12": "Infectious 1 after 2",
                "I_21": "Infectious 2 after 1",
                "R": "Removed 1 and 2",
            }
        )
        self.parameters = OrderedDict(
            {
                "beta": r"$\beta$",  #  infection rate
                "gamma": r"$\gamma",  # recovery rate
                "mu": r"$\mu",  # birth and death rate
                "rho": r"$\rho$",  # ratio of secondary infections contributing
                "phi": r"$\phi$",  # import parameter
                "alpha": r"$\alpha$",  # temporary cross-immunity
            }
        )
        self.model_type = "SIR2Strain"

    @property
    def diagram(self) -> str:
        """Mermaid diagram of the compartmental model"""
        return r"""flowchart LR
    S(Susceptible) -->|"$$\frac{\beta}{N} (I1 + \rho N + \phi I_{21})$$"| I1(I1)
    S -->|"$$\frac{\beta}{N} (I2 + \rho N + \phi I_{12})$$"| I2(I2)

    I1 -->|"$$\gamma$$"| R1(R1)

    I2 -->|"$$\gamma$$"| R2(R2)

    R1 -->|"$$\alpha$$"| S1(S1)

    R2 -->|"$$\alpha$$"| S2(S2)

    S1 -->|"$$\frac{\beta}{N} (I2 + \rho N + \phi I_{12})$$"| I12(I12)

    S2 -->|"$$\frac{\beta}{N} (I1 + \rho N + \phi I_{21})$$"| I21(I21)

    I12 -->|"$$\gamma$$"| R(R)

    I21 -->|"$$\gamma$$"| R

    R -->|"$$\mu$$"| OUT
    
    classDef strain1 fill:#ffcccc,stroke:#ff0000
    classDef strain2 fill:#ccffcc,stroke:#00ff00
    classDef invisible fill:none,stroke:none,color:none
    
    class I1,I21 strain1
    class I2,I12 strain2

    """

    def _model(self, t: float, y: list[float], params: dict[str, float]) -> list[float]:
        (S, I1, I2, R1, R2, S1, S2, I12, I21, R) = y
        beta, gamma, mu, rho, phi, alpha, N = (
            params["beta"],
            params["gamma"],
            params["mu"],
            params["rho"],
            params["phi"],
            params["alpha"],
            params["N"],
        )
        return [
            -beta / N * S * (I1 + rho * N + phi * I21)
            - beta / N * S * (I2 + rho * N + phi * I12)
            + mu * (N - S),
            beta / N * S * (I1 + rho * N + phi * I21) - (gamma + mu) * I1,
            beta / N * S * (I2 + rho * N + phi * I12) - (gamma + mu) * I2,
            gamma * I1 - (alpha + mu) * R1,
            gamma * I2 - (alpha + mu) * R2,
            -beta / N * S1 * (I2 + rho * N + phi * I12) + alpha * R1 - mu * S1,
            -beta / N * S2 * (I1 + rho * N + phi * I21) + alpha * R2 - mu * S2,
            beta / N * S1 * (I2 + rho * N + phi * I12) - (gamma + mu) * I12,
            beta / N * S2 * (I1 + rho * N + phi * I21) - (gamma + mu) * I21,
            gamma * (I12 + I21) - mu * R,
        ]


class SISLogistic(ContinuousModel):
    """
    SIS model with logistic growth.

    State Variables:
        - S: Susceptible individuals
        - I: Infectious individuals

    Parameters:
        - beta (β): Transmission rate
        - gamma (γ): Recovery rate
        - r: Population growth rate
        - k: Carrying capacity

    Equations:

        dS/dt = rS(1 - N/k) - βSI/N + γI
        dI/dt = βSI/N - γI

    Basic Reproduction Number:
        R₀ = β/γ

    Note:
        N = S + I
    """

    def __init__(self):
        super().__init__()
        self.state_variables = OrderedDict({"S": "Susceptible", "I": "Infectious"})
        self.parameters = {"beta": r"\beta", "gamma": r"\gamma", "r": r"r", "k": r"k"}
        self.model_type = "SIS_logistic"

    @property
    def diagram(self) -> str:
        return r"""flowchart LR
         
S(Susceptible) -->|$$\beta$$| I(Infectious)
I -->|$$\gamma$$| S
S --> |$$r(1-N/k)$$| S
"""

    @property
    def R0(self) -> float | None:
        """
        Basic reproduction number for SIS logistc model.

        R0 = β / γ

        :return: Basic reproduction number, or None if parameters not set
        """
        if self.param_values and "beta" in self.param_values and "gamma" in self.param_values:
            return float(self.param_values["beta"] / self.param_values["gamma"])
        return None

    def _model(self, t: float, y: list[float], params: dict[str, float]) -> list[float]:

        S, I = y

        beta = params["beta"]
        gamma = params["gamma"]
        r = params["r"]
        k = params["k"]

        N = S + I

        dS = r * S * (1 - N / k) - beta * S * I / N + gamma * I
        dI = beta * S * I / N - gamma * I

        return [dS, dI]
class SIRSNonAutonomous(ContinuousModel):
    """
    SIRS model with time-dependent parameters.

    Includes waning immunity: R -> S with rate alpha(t)
    """

    def __init__(self):
        super().__init__()

        # Fórmulas simbólicas para compatibilidade com os testes avançados
        I, tau = sp.symbols("I tau")
        beta, gamma, alpha, I0, N = sp.symbols("beta gamma alpha I0 N")

        self._formulas = {
            "I": I * beta * (1 - I0 / N) * (1 + tau / alpha) ** (-(alpha + 1)) - gamma * I,
            "tau": beta * I / N,
        }


    def removed(self, I: float, tau: float, N: float, I0: float, alpha: float) -> float:
        S = self.susceptible(tau, N, I0, alpha)
        return N - S - I

    def _model(self, t: float, y: list[float], params: dict[str, float]) -> list[float]:
        I, tau = y
        beta = params["beta"]
        gamma = params["gamma"]
        alpha = params["alpha"]
        I0 = params["I0"]
        N = params["N"]

        dI = I * beta * (1 - I0 / N) * (1 + tau / alpha) ** (-(alpha + 1)) - gamma * I
        dtau = beta * I / N

        return [dI, dtau]
class NeipelHeterogeneousSIR(ContinuousModel):
    """
    Heterogeneous SIR model based on Neipel et al. (2020).

    State Variables:
        - I: Infectious individuals
        - tau: epidemic progress variable

    Derived quantities:
        - S(t) = (N - I0) * (1 + tau/alpha)^(-alpha)
        - R(t) = N - S(t) - I(t)

    Parameters:
        - beta: transmission rate
        - gamma: recovery rate
        - alpha: susceptibility heterogeneity exponent
        - I0: initial number of infectious individuals
    """

    def __init__(self):
        super().__init__()
        self.state_variables = OrderedDict(
            {"I": "Infectious", "tau": "Epidemic progress"}
        )
        self.parameters = OrderedDict(
            {
                "beta": r"$\beta$",
                "gamma": r"$\gamma$",
                "alpha": r"$\alpha$",
                "I0": r"$I_0$",
            }
        )
        self.model_type = "NeipelHeterogeneousSIR"

    @property
    def diagram(self) -> str:
        return r"""flowchart LR

S(Susceptible heterogeneous) -->|$$\beta, \alpha$$| I(Infectious)
I -->|$$\gamma$$| R(Removed)
"""

    @property
    def R0(self) -> float | None:
        if self.param_values and "beta" in self.param_values and "gamma" in self.param_values:
            return float(self.param_values["beta"] / self.param_values["gamma"])
        return None

    def susceptible(self, tau: float, N: float, I0: float, alpha: float) -> float:
        return (N - I0) * (1 + tau / alpha) ** (-alpha)

        self.model_type = "SIRS Non-Autonomous"

    def _model(self, t: float, y: list[float], params: dict):
        S, I, R = y
        N = params["N"]

        # parâmetros dependentes do tempo
        alpha = params["alpha"](t)
        beta = params["beta"](t)
        gamma = params["gamma"](t)

        dSdt = -beta * S * I / N + alpha * R/N
        dIdt = beta * S * I / N - gamma * I/N
        dRdt = gamma * I/N - alpha * R


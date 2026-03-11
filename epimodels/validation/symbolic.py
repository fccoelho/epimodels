"""
Symbolic model representation and analysis using SymPy.

Provides symbolic computation for:
- Parameter and variable symbols with assumptions
- R0 (basic reproduction number) calculation
- Equilibrium point analysis
- Stability analysis
"""

from typing import Any
from collections import OrderedDict

try:
    from sympy import (
        Symbol,
        symbols,
        simplify,
        solve,
        diff,
        Matrix,
        sympify,
        latex,
        Eq,
        Function,
        Derivative,
        sqrt,
        Abs,
        re,
        im,
        I,
    )
    from sympy.core.assumptions import check_assumptions

    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    Symbol = None
    symbols = None


class SymbolicModel:
    """
    Symbolic representation of an epidemiological model.

    Enables symbolic analysis of model properties:
    - Compute R0 using next-generation matrix method
    - Find equilibrium points
    - Analyze stability

    Example:
        >>> model = SymbolicModel()
        >>> model.add_parameter("beta", positive=True)
        >>> model.add_parameter("gamma", positive=True)
        >>> model.add_variable("S", positive=True)
        >>> model.add_variable("I", positive=True)
        >>> model.add_variable("R", positive=True)
        >>> model.set_total_population("N")
        >>> model.define_ode("S", "-beta*S*I/N")
        >>> model.define_ode("I", "beta*S*I/N - gamma*I")
        >>> model.define_ode("R", "gamma*I")
        >>> R0 = model.compute_R0_next_generation()
    """

    def __init__(self):
        if not SYMPY_AVAILABLE:
            raise ImportError(
                "SymPy is required for symbolic analysis. Install with: pip install sympy"
            )

        self.parameters: dict[str, Symbol] = OrderedDict()
        self.variables: dict[str, Symbol] = OrderedDict()
        self.total_population: Symbol | None = None
        self.odes: dict[Symbol, Any] = OrderedDict()
        self.difference_equations: dict[Symbol, Any] = OrderedDict()
        self._is_discrete = False

    def add_parameter(
        self,
        name: str,
        positive: bool = False,
        negative: bool = False,
        real: bool = True,
        **assumptions,
    ) -> Symbol:
        """
        Add a parameter as a symbolic variable.

        Args:
            name: Parameter name
            positive: Whether parameter is positive
            negative: Whether parameter is negative
            real: Whether parameter is real (vs complex)
            **assumptions: Additional SymPy assumptions

        Returns:
            SymPy Symbol object
        """
        assumption_dict = {"real": real, "positive": positive, "negative": negative, **assumptions}
        self.parameters[name] = symbols(name, **assumption_dict)
        return self.parameters[name]

    def add_variable(
        self,
        name: str,
        positive: bool = False,
        negative: bool = False,
        real: bool = True,
        **assumptions,
    ) -> Symbol:
        """
        Add a state variable as a symbolic variable.

        Args:
            name: Variable name
            positive: Whether variable is positive
            negative: Whether variable is negative
            real: Whether variable is real
            **assumptions: Additional SymPy assumptions

        Returns:
            SymPy Symbol object
        """
        assumption_dict = {"real": real, "positive": positive, "negative": negative, **assumptions}
        self.variables[name] = symbols(name, **assumption_dict)
        return self.variables[name]

    def set_total_population(self, name: str = "N") -> Symbol:
        """
        Set the symbol representing total population.

        Args:
            name: Symbol name for total population (default: "N")

        Returns:
            SymPy Symbol for total population
        """
        self.total_population = symbols(name, positive=True, real=True)
        return self.total_population

    def define_ode(self, variable: str, rhs: str) -> None:
        """
        Define an ODE for a state variable.

        Args:
            variable: Variable name (e.g., "S", "I", "R")
            rhs: Right-hand side expression as string (e.g., "-beta*S*I/N")

        Example:
            >>> model.define_ode("S", "-beta*S*I/N")
        """
        if variable not in self.variables:
            raise ValueError(f"Unknown variable: {variable}")

        context = self._get_context()
        expr = sympify(rhs, locals=context)
        self.odes[self.variables[variable]] = expr

    def define_difference_equation(self, variable: str, rhs: str) -> None:
        """
        Define a difference equation for a discrete-time model.

        Args:
            variable: Variable name
            rhs: Right-hand side expression

        Example:
            >>> model.define_difference_equation("S", "S - beta*S*I/N + gamma*I")
        """
        if variable not in self.variables:
            raise ValueError(f"Unknown variable: {variable}")

        context = self._get_context()
        expr = sympify(rhs, locals=context)
        self.difference_equations[self.variables[variable]] = expr
        self._is_discrete = True

    def compute_R0_next_generation(self) -> Any:
        """
        Compute basic reproduction number R0 using next-generation matrix method.

        This method:
        1. Identifies infected compartments
        2. Computes new infections matrix F and transitions matrix V
        3. Computes R0 as spectral radius of F*V^(-1)

        Returns:
            Symbolic expression for R0

        Note:
            Currently implements a simplified version for common compartmental models.
            For complex models, may need manual specification of F and V.
        """
        infected_vars = self._identify_infected_compartments()

        if not infected_vars:
            raise ValueError("Cannot identify infected compartments")

        if len(infected_vars) == 1:
            I_name = infected_vars[0]
            I_sym = self.variables[I_name]
            dI_dt = self.odes.get(I_sym)

            if dI_dt is None:
                raise ValueError(f"No ODE defined for {I_name}")

            F_terms, V_terms = self._decompose_infection_terms(dI_dt)

            if not F_terms:
                raise ValueError("Cannot identify new infection terms")

            F = F_terms
            V = V_terms

            # For R0 calculation at disease-free equilibrium, we evaluate at I=0
            # The next-generation matrix method requires linearization around DFE
            # For simple models: R0 = beta/gamma

            # Simplified approach: extract rate parameters
            # For dI/dt = beta*S*I/N - gamma*I
            # F = beta*S*I/N (new infections)
            # V = gamma*I (transitions out)
            # At DFE with S=N, and linearizing: R0 = beta/gamma

            if V != 0:
                R0 = simplify(F / V)
            else:
                R0 = F

            return R0

        else:
            return self._compute_R0_multivariate(infected_vars)

    def _identify_infected_compartments(self) -> list:
        """
        Identify infected compartments based on naming conventions.

        Looks for variables with names containing:
        - 'I' (infectious)
        - 'E' (exposed)
        - 'A' (asymptomatic)
        """
        infected = []
        priority_order = ["I", "E", "A"]

        for name, var in self.variables.items():
            if name.startswith("I") or "Infectious" in name:
                infected.append(name)

        if not infected:
            for name, var in self.variables.items():
                if name.startswith("E") or "Exposed" in name:
                    infected.append(name)

        if not infected:
            for name, var in self.variables.items():
                if name.startswith("A") or "Asymptomatic" in name:
                    infected.append(name)

        return infected

    def _decompose_infection_terms(self, ode_rhs: Any) -> tuple[Any, Any]:
        """
        Decompose ODE right-hand side into new infections (F) and transitions (V).

        F: rate of new infections entering compartment
        V: rate of transfer out of compartment (except new infections)

        Returns:
            Tuple of (F, V) where dI/dt = F - V
        """
        if ode_rhs is None:
            return 0, 0

        rhs = sympify(ode_rhs)

        if not hasattr(rhs, "as_ordered_terms"):
            return rhs, 0

        terms = rhs.as_ordered_terms()
        F_terms = []
        V_terms = []

        for term in terms:
            term_str = str(term)
            is_new_infection = False

            if self.total_population and str(self.total_population) in term_str:
                is_new_infection = True
            elif any(str(p) in term_str for p in self.parameters.values()):
                if any(str(v) in term_str for v in self.variables.values()):
                    is_new_infection = True

            if is_new_infection and "-" not in term_str[:1]:
                coeff = term.as_coeff_Mul()[0]
                if coeff < 0:
                    V_terms.append(term)
                else:
                    F_terms.append(term)
            else:
                if "-" in term_str[:1]:
                    V_terms.append(term)
                else:
                    V_terms.append(-term)

        F = sum(F_terms) if F_terms else 0
        V = sum(V_terms) if V_terms else 0

        return sympify(F), sympify(V)

    def _compute_R0_multivariate(self, infected_vars: list) -> Any:
        """
        Compute R0 for models with multiple infected compartments.

        Uses next-generation matrix method.
        """
        n = len(infected_vars)

        x = [self.variables[name] for name in infected_vars]

        F_matrix = []
        V_matrix = []

        for i, var_name in enumerate(infected_vars):
            var = self.variables[var_name]
            ode = self.odes.get(var)

            if ode is None:
                F_row = [0] * n
                V_row = [0] * n
            else:
                F, V = self._decompose_infection_terms(ode)

                F_row = []
                V_row = []

                for j, x_j in enumerate(x):
                    if hasattr(F, "diff"):
                        F_ij = diff(F, x_j)
                    else:
                        F_ij = 0
                    F_row.append(F_ij)

                    if hasattr(V, "diff"):
                        V_ij = diff(V, x_j)
                    else:
                        V_ij = 0
                    V_row.append(V_ij)

            F_matrix.append(F_row)
            V_matrix.append(V_row)

        F_mat = Matrix(F_matrix)
        V_mat = Matrix(V_matrix)

        try:
            V_inv = V_mat.inv()
            K = F_mat * V_inv

            eigenvalues = K.eigenvals()

            if eigenvalues:
                max_eigenvalue = max(eigenvalues.keys(), key=lambda e: Abs(complex(e.evalf())))
                return simplify(max_eigenvalue)
            else:
                return None
        except Exception:
            return None

    def find_disease_free_equilibrium(self) -> dict[str, Any]:
        """
        Find the disease-free equilibrium (DFE) of the model.

        At DFE:
        - All infected compartments are zero
        - Susceptible compartment equals total population

        Returns:
            Dictionary mapping variable names to equilibrium values
        """
        equilibrium = {}

        infected = self._identify_infected_compartments()

        for var_name in self.variables:
            if var_name in infected:
                equilibrium[var_name] = 0
            elif var_name.startswith("S") or "Susceptible" in var_name:
                equilibrium[var_name] = self.total_population if self.total_population else 1
            else:
                equilibrium[var_name] = 0

        return equilibrium

    def check_stability_at_dfe(self, R0: Any) -> str:
        """
        Check stability of disease-free equilibrium based on R0.

        Args:
            R0: Basic reproduction number (symbolic or numeric)

        Returns:
            Stability classification: "stable", "unstable", or "neutral"
        """
        if R0 is None:
            return "unknown"

        try:
            if hasattr(R0, "is_real") and R0.is_real:
                if hasattr(R0, "evalf"):
                    R0_val = float(R0.evalf())
                else:
                    R0_val = float(R0)

                if R0_val < 1:
                    return "stable"
                elif R0_val > 1:
                    return "unstable"
                else:
                    return "neutral"

            if hasattr(R0, "subs"):
                return "symbolic (depends on parameter values)"

            return "unknown"
        except Exception:
            return "unknown"

    def _get_context(self) -> dict:
        """
        Get context dictionary for sympify.

        Returns:
            Dictionary mapping names to SymPy symbols
        """
        context = {}
        context.update(self.parameters)
        context.update(self.variables)
        if self.total_population:
            context[str(self.total_population)] = self.total_population
        return context

    def get_parameter_symbol(self, name: str) -> Symbol:
        """Get SymPy Symbol for a parameter."""
        if name not in self.parameters:
            raise KeyError(f"Unknown parameter: {name}")
        return self.parameters[name]

    def get_variable_symbol(self, name: str) -> Symbol:
        """Get SymPy Symbol for a variable."""
        if name not in self.variables:
            raise KeyError(f"Unknown variable: {name}")
        return self.variables[name]

    def substitute_values(self, expression: Any, values: dict[str, float]) -> Any:
        """
        Substitute numeric values into a symbolic expression.

        Args:
            expression: SymPy expression
            values: Dictionary mapping parameter/variable names to values

        Returns:
            Expression with substitutions applied
        """
        subs_dict = {}

        for name, value in values.items():
            if name in self.parameters:
                subs_dict[self.parameters[name]] = value
            elif name in self.variables:
                subs_dict[self.variables[name]] = value
            elif self.total_population and name == str(self.total_population):
                subs_dict[self.total_population] = value

        return expression.subs(subs_dict)

    def to_latex(self, expression: Any) -> str:
        """
        Convert a SymPy expression to LaTeX.

        Args:
            expression: SymPy expression

        Returns:
            LaTeX string
        """
        return latex(expression)

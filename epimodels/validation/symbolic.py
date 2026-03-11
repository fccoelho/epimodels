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
        S,
        nsolve,
    )
    from sympy.core.assumptions import check_assumptions
    import numpy as np

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

    def find_all_equilibria(
        self,
        params: dict[str, float] | None = None,
        numeric_fallback: bool = True,
        max_solutions: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Find all equilibrium points of the model.

        Solves the system of equations f(x) = 0 where f is the vector field.
        Includes both disease-free and endemic equilibria.

        Args:
            params: Parameter values for numeric solving (optional)
            numeric_fallback: If True, use numerical solver when symbolic fails
            max_solutions: Maximum number of solutions to return

        Returns:
            List of equilibrium dictionaries. Each dictionary contains:
            - Variable names mapped to equilibrium values
            - 'type': 'dfe' or 'endemic'
            - 'method': 'symbolic' or 'numeric'

        Example:
            >>> model.find_all_equilibria(params={'beta': 0.3, 'gamma': 0.1, 'N': 1000})
            [
                {'S': N, 'I': 0, 'R': 0, 'type': 'dfe', 'method': 'symbolic'},
                {'S': N*gamma/beta, 'I': ..., 'R': ..., 'type': 'endemic', 'method': 'symbolic'}
            ]
        """
        equilibria = []

        # 1. Always include DFE
        dfe = self.find_disease_free_equilibrium()
        dfe["type"] = "dfe"
        dfe["method"] = "analytical"
        equilibria.append(dfe)

        # 2. Try symbolic solving for all equilibria
        symbolic_eqs = self._find_equilibria_symbolic(max_solutions)

        for eq in symbolic_eqs:
            if not self._is_equilibrium_duplicate(eq, equilibria):
                eq_type = self._classify_equilibrium(eq)
                eq["type"] = eq_type
                eq["method"] = "symbolic"
                equilibria.append(eq)

        # 3. If requested and params provided, try numeric solving
        if numeric_fallback and params and len(equilibria) < max_solutions:
            numeric_eqs = self._find_equilibria_numeric(params, max_solutions - len(equilibria))

            for eq in numeric_eqs:
                if not self._is_equilibrium_duplicate(eq, equilibria):
                    eq_type = self._classify_equilibrium(eq)
                    eq["type"] = eq_type
                    eq["method"] = "numeric"
                    equilibria.append(eq)

        return equilibria[:max_solutions]

    def find_endemic_equilibrium(
        self, params: dict[str, float] | None = None, numeric_fallback: bool = True
    ) -> dict[str, Any] | None:
        """
        Find endemic equilibrium point.

        At endemic equilibrium, disease persists (I* > 0).
        Only exists when R0 > 1.

        Args:
            params: Parameter values for numeric solving and R0 calculation
            numeric_fallback: If True, use numerical solver when symbolic fails

        Returns:
            Equilibrium dictionary or None if no endemic equilibrium exists

        Example:
            >>> model.find_endemic_equilibrium({'beta': 0.3, 'gamma': 0.1, 'N': 1000})
            {'S': 333.33, 'I': 666.67, 'R': 0, 'type': 'endemic', 'method': 'symbolic'}
        """
        # Check if endemic equilibrium exists (R0 > 1)
        if params:
            try:
                R0_expr = self.compute_R0_next_generation()
                R0_val = float(self.substitute_values(R0_expr, params))
                if R0_val <= 1:
                    return None  # No endemic equilibrium when R0 <= 1
            except Exception:
                pass

        # Find all equilibria and filter for endemic
        equilibria = self.find_all_equilibria(params, numeric_fallback)

        for eq in equilibria:
            if eq.get("type") == "endemic":
                return eq

        return None

    def _find_equilibria_symbolic(self, max_solutions: int) -> list[dict[str, Any]]:
        """
        Find equilibria using symbolic solving.

        Args:
            max_solutions: Maximum number of solutions to find

        Returns:
            List of equilibrium dictionaries
        """
        equilibria = []

        if not self.odes:
            return equilibria

        try:
            # Build system of equations: dx/dt = 0 for all variables
            equations = []
            for var_sym, ode_rhs in self.odes.items():
                equations.append(Eq(ode_rhs, 0))

            # Solve the system
            variables_list = list(self.variables.values())
            solutions = solve(equations, variables_list, dict=True)

            # Handle single solution case
            if solutions and not isinstance(solutions, list):
                solutions = [solutions]

            for solution in solutions[:max_solutions]:
                eq = {}

                # Extract values for each variable
                for var_name, var_sym in self.variables.items():
                    if var_sym in solution:
                        eq[var_name] = solution[var_sym]
                    else:
                        eq[var_name] = var_sym

                # Validate the equilibrium
                if self._validate_equilibrium(eq):
                    equilibria.append(eq)

        except Exception as e:
            # Symbolic solving failed, will fall back to numeric
            pass

        return equilibria

    def _find_equilibria_numeric(
        self, params: dict[str, float], max_solutions: int
    ) -> list[dict[str, Any]]:
        """
        Find equilibria using numerical solving.

        Args:
            params: Parameter values
            max_solutions: Maximum number of solutions to find

        Returns:
            List of equilibrium dictionaries
        """
        equilibria = []

        if not self.odes or not params:
            return equilibria

        try:
            import numpy as np
            from scipy.optimize import fsolve

            # Substitute parameter values into ODEs
            odes_numeric = {}
            for var_sym, ode_rhs in self.odes.items():
                odes_numeric[var_sym] = self.substitute_values(ode_rhs, params)

            # Define system of equations for fsolve
            def equations(x):
                result = []
                for i, (var_sym, ode_rhs) in enumerate(odes_numeric.items()):
                    # Substitute variable values
                    subs_dict = {}
                    for j, (v_name, v_sym) in enumerate(self.variables.items()):
                        subs_dict[v_sym] = x[j]

                    val = float(ode_rhs.subs(subs_dict))
                    result.append(val)
                return result

            # Try multiple initial guesses
            n_vars = len(self.variables)
            initial_guesses = self._generate_initial_guesses(params, n_vars, max_solutions)

            found_solutions = set()

            for x0 in initial_guesses:
                try:
                    solution, info, ier, msg = fsolve(equations, x0, full_output=True)

                    if ier == 1:  # Solution found
                        # Round to avoid duplicates
                        solution_key = tuple(round(x, 6) for x in solution)

                        if solution_key not in found_solutions:
                            found_solutions.add(solution_key)

                            eq = {}
                            for i, (var_name, var_sym) in enumerate(self.variables.items()):
                                eq[var_name] = float(solution[i])

                            # Validate
                            if self._validate_equilibrium(eq, tolerance=1e-6):
                                equilibria.append(eq)

                                if len(equilibria) >= max_solutions:
                                    break
                except Exception:
                    continue

        except ImportError:
            # scipy not available
            pass
        except Exception:
            pass

        return equilibria

    def _generate_initial_guesses(
        self, params: dict[str, float], n_vars: int, n_guesses: int
    ) -> list[list[float]]:
        """
        Generate initial guesses for numeric equilibrium finding.

        Args:
            params: Parameter values
            n_vars: Number of variables
            n_guesses: Number of guesses to generate

        Returns:
            List of initial guess vectors
        """
        guesses = []
        N = params.get("N", params.get(str(self.total_population), 1000))

        # 1. DFE
        dfe_guess = [0.0] * n_vars
        for i, var_name in enumerate(self.variables.keys()):
            if var_name.startswith("S") or "Susceptible" in var_name:
                dfe_guess[i] = N
        guesses.append(dfe_guess)

        # 2. Endemic-like (small I, large S)
        endemic_guess = [0.0] * n_vars
        for i, var_name in enumerate(self.variables.keys()):
            if var_name.startswith("S") or "Susceptible" in var_name:
                endemic_guess[i] = N * 0.8
            elif var_name.startswith("I") or "Infectious" in var_name:
                endemic_guess[i] = N * 0.1
            elif var_name.startswith("R") or "Removed" in var_name:
                endemic_guess[i] = N * 0.1
        guesses.append(endemic_guess)

        # 3. Random guesses
        np.random.seed(42)  # Reproducibility
        for _ in range(n_guesses - 2):
            guess = np.random.uniform(0, N, n_vars)
            # Ensure sum equals N for conserved models
            if self.total_population:
                guess = guess / guess.sum() * N
            guesses.append(guess.tolist())

        return guesses[:n_guesses]

    def _classify_equilibrium(self, eq: dict[str, Any]) -> str:
        """
        Classify equilibrium as disease-free or endemic.

        Args:
            eq: Equilibrium dictionary

        Returns:
            'dfe' or 'endemic'
        """
        infected_vars = self._identify_infected_compartments()

        for var_name in infected_vars:
            value = eq.get(var_name, 0)

            # Convert symbolic to numeric if needed
            if hasattr(value, "evalf"):
                try:
                    value = float(value.evalf())
                except Exception:
                    # Symbolic and non-zero, likely endemic
                    return "endemic"

            if value != 0:
                return "endemic"

        return "dfe"

    def _validate_equilibrium(self, eq: dict[str, Any], tolerance: float = 1e-10) -> bool:
        """
        Validate that a point is actually an equilibrium.

        Args:
            eq: Equilibrium dictionary
            tolerance: Tolerance for checking dx/dt ≈ 0

        Returns:
            True if valid equilibrium, False otherwise
        """
        if not self.odes:
            return True

        try:
            # Substitute equilibrium values into ODEs
            subs_dict = {}
            for var_name, value in eq.items():
                if var_name in self.variables:
                    subs_dict[self.variables[var_name]] = value

            for var_sym, ode_rhs in self.odes.items():
                residual = ode_rhs.subs(subs_dict)

                # Evaluate numerically
                if hasattr(residual, "evalf"):
                    residual = float(residual.evalf())

                if abs(residual) > tolerance:
                    return False

            return True
        except Exception:
            return False

    def _is_equilibrium_duplicate(
        self, eq: dict[str, Any], existing: list[dict[str, Any]], tolerance: float = 1e-6
    ) -> bool:
        """
        Check if equilibrium is duplicate of existing ones.

        Args:
            eq: Equilibrium to check
            existing: List of existing equilibria
            tolerance: Tolerance for comparison

        Returns:
            True if duplicate, False otherwise
        """
        for existing_eq in existing:
            is_duplicate = True

            for var_name in self.variables.keys():
                val1 = eq.get(var_name, 0)
                val2 = existing_eq.get(var_name, 0)

                # Convert to float for comparison
                try:
                    if hasattr(val1, "evalf"):
                        val1 = float(val1.evalf())
                    elif hasattr(val1, "__float__"):
                        val1 = float(val1)
                except (ValueError, TypeError):
                    # Cannot convert to float, treat as different
                    is_duplicate = False
                    break

                try:
                    if hasattr(val2, "evalf"):
                        val2 = float(val2.evalf())
                    elif hasattr(val2, "__float__"):
                        val2 = float(val2)
                except (ValueError, TypeError):
                    # Cannot convert to float, treat as different
                    is_duplicate = False
                    break

                if abs(val1 - val2) > tolerance:
                    is_duplicate = False
                    break

            if is_duplicate:
                return True

        return False

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

    def compute_jacobian(
        self, equilibrium: dict[str, float | Symbol], substitute_values: bool = False
    ) -> Matrix:
        """
        Compute Jacobian matrix at an equilibrium point.

        The Jacobian matrix J has entries J_ij = ∂f_i/∂x_j where:
        - f_i is the ODE for variable i
        - x_j is the j-th state variable

        Args:
            equilibrium: Dictionary mapping variable names to values.
                        Can be symbolic or numeric values.
            substitute_values: If True, substitute equilibrium values into Jacobian.
                             If False, keep symbolic form.

        Returns:
            SymPy Matrix representing the Jacobian

        Example:
            >>> dfe = model.find_disease_free_equilibrium()
            >>> J = model.compute_jacobian(dfe)
            >>> J
            Matrix([
                [0, -beta, 0],
                [0, beta - gamma, 0],
                [0, gamma, 0]
            ])
        """
        if not self.odes:
            raise ValueError("No ODEs defined for this model")

        # Build Jacobian matrix
        n_vars = len(self.variables)
        var_names = list(self.variables.keys())
        var_syms = list(self.variables.values())

        jacobian_entries = []

        for i, var_name_i in enumerate(var_names):
            var_sym_i = self.variables[var_name_i]
            ode_i = self.odes.get(var_sym_i)

            if ode_i is None:
                # No ODE for this variable - all zeros
                jacobian_entries.append([0] * n_vars)
                continue

            row = []
            for j, var_name_j in enumerate(var_names):
                var_sym_j = self.variables[var_name_j]

                # Compute partial derivative ∂f_i/∂x_j
                try:
                    partial = diff(ode_i, var_sym_j)
                    partial = simplify(partial)
                except Exception:
                    partial = 0

                row.append(partial)

            jacobian_entries.append(row)

        J = Matrix(jacobian_entries)

        # Substitute equilibrium values if requested
        if substitute_values and equilibrium:
            subs_dict = {}
            for var_name, value in equilibrium.items():
                if var_name in self.variables:
                    subs_dict[self.variables[var_name]] = value

            J = J.subs(subs_dict)
            J = simplify(J)

        return J

    def compute_eigenvalues(
        self, jacobian: Matrix, numeric: bool = False, params: dict[str, float] | None = None
    ) -> list[Any]:
        """
        Compute eigenvalues of the Jacobian matrix.

        Eigenvalues determine local stability:
        - All Re(λ) < 0: stable equilibrium
        - Any Re(λ) > 0: unstable equilibrium
        - Re(λ) = 0: neutral or bifurcation point

        Args:
            jacobian: Jacobian matrix (from compute_jacobian)
            numeric: Force numeric evaluation of eigenvalues
            params: Parameter values for numeric evaluation

        Returns:
            List of eigenvalues (symbolic or numeric complex numbers)

        Example:
            >>> J = model.compute_jacobian(dfe, substitute_values=True)
            >>> eigenvalues = model.compute_eigenvalues(J)
            >>> eigenvalues
            [0, -gamma, -beta + gamma]
        """
        eigenvalues = []

        try:
            if numeric or params:
                # Substitute parameter values if provided
                if params:
                    subs_dict = {}
                    for param_name, value in params.items():
                        if param_name in self.parameters:
                            subs_dict[self.parameters[param_name]] = value
                    jacobian = jacobian.subs(subs_dict)

                # Try numeric eigenvalue computation
                try:
                    # Convert to numpy array for numeric computation
                    jac_np = np.array(jacobian.tolist(), dtype=float)
                    eigenvalues_np = np.linalg.eigvals(jac_np)
                    eigenvalues = [complex(ev) for ev in eigenvalues_np]
                except Exception:
                    # Fall back to symbolic
                    pass

            if not eigenvalues:
                # Try symbolic eigenvalue computation
                eigenvalue_dict = jacobian.eigenvals()

                if eigenvalue_dict:
                    for eigenvalue, multiplicity in eigenvalue_dict.items():
                        eigenvalue = simplify(eigenvalue)
                        # Add according to multiplicity
                        for _ in range(multiplicity):
                            eigenvalues.append(eigenvalue)
                else:
                    # If eigenvals() fails, try more robust method
                    n = jacobian.shape[0]
                    char_poly = jacobian.charpoly()
                    roots = solve(char_poly.as_expr())
                    eigenvalues = roots if isinstance(roots, list) else [roots]

        except Exception as e:
            # Last resort: return empty list
            pass

        return eigenvalues

    def analyze_stability_full(
        self,
        equilibrium: dict[str, float | Symbol],
        params: dict[str, float] | None = None,
        tolerance: float = 1e-10,
    ) -> dict[str, Any]:
        """
        Perform full stability analysis at an equilibrium point.

        Computes and analyzes:
        - Jacobian matrix
        - Eigenvalues
        - Stability classification
        - Bifurcation indicators
        - Detailed classification (node, focus, saddle, etc.)

        Args:
            equilibrium: Equilibrium point (variable names to values)
            params: Parameter values for numeric evaluation
            tolerance: Tolerance for eigenvalue zero detection

        Returns:
            Dictionary with comprehensive stability information:
            - 'jacobian': Jacobian matrix (symbolic or numeric)
            - 'eigenvalues': List of eigenvalues
            - 'eigenvalues_numeric': Numeric eigenvalues (if params provided)
            - 'stability': 'stable', 'unstable', 'neutral', or 'saddle'
            - 'classification': Detailed type (e.g., 'stable_node', 'unstable_focus')
            - 'max_real_part': Maximum real part of eigenvalues
            - 'min_real_part': Minimum real part of eigenvalues
            - 'has_complex': Boolean indicating complex eigenvalues
            - 'near_bifurcation': Boolean indicating proximity to bifurcation
            - 'bifurcation_type': Type of bifurcation if detected

        Example:
            >>> result = model.analyze_stability_full(dfe, params={'beta': 0.3, 'gamma': 0.1})
            >>> result['stability']
            'unstable'
            >>> result['classification']
            'unstable_node'
            >>> result['max_real_part']
            0.2
        """
        result = {
            "jacobian": None,
            "eigenvalues": [],
            "eigenvalues_numeric": [],
            "stability": "unknown",
            "classification": "unknown",
            "max_real_part": None,
            "min_real_part": None,
            "has_complex": False,
            "near_bifurcation": False,
            "bifurcation_type": None,
        }

        try:
            # 1. Compute Jacobian
            J = self.compute_jacobian(equilibrium, substitute_values=False)
            result["jacobian"] = J

            # 2. Compute symbolic eigenvalues
            eigenvalues_sym = self.compute_eigenvalues(J, numeric=False)
            result["eigenvalues"] = eigenvalues_sym

            # 3. Compute numeric eigenvalues if params provided
            if params:
                J_numeric = self.compute_jacobian(equilibrium, substitute_values=True)
                eigenvalues_num = self.compute_eigenvalues(J_numeric, numeric=True, params=params)
                result["eigenvalues_numeric"] = eigenvalues_num

                # Use numeric eigenvalues for analysis
                eigenvalues_for_analysis = eigenvalues_num
            else:
                eigenvalues_for_analysis = eigenvalues_sym

            # 4. Analyze eigenvalue spectrum
            if eigenvalues_for_analysis:
                real_parts = []
                imag_parts = []

                for ev in eigenvalues_for_analysis:
                    if hasattr(ev, "evalf"):
                        try:
                            ev_complex = complex(ev.evalf())
                            real_parts.append(ev_complex.real)
                            imag_parts.append(ev_complex.imag)
                        except Exception:
                            # Symbolic eigenvalue
                            pass
                    elif isinstance(ev, complex):
                        real_parts.append(ev.real)
                        imag_parts.append(ev.imag)
                    elif isinstance(ev, (int, float)):
                        real_parts.append(ev)
                        imag_parts.append(0.0)

                if real_parts:
                    result["max_real_part"] = max(real_parts)
                    result["min_real_part"] = min(real_parts)
                    result["has_complex"] = any(abs(im) > tolerance for im in imag_parts)

                    # 5. Classify stability
                    result["stability"] = self._classify_stability(
                        real_parts, imag_parts, tolerance
                    )

                    # 6. Detailed classification
                    result["classification"] = self._classify_stability_detailed(
                        real_parts, imag_parts, tolerance
                    )

                    # 7. Detect bifurcations
                    result["near_bifurcation"], result["bifurcation_type"] = (
                        self._detect_bifurcation(real_parts, imag_parts, tolerance)
                    )

        except Exception as e:
            result["error"] = str(e)

        return result

    def _classify_stability(
        self, real_parts: list[float], imag_parts: list[float], tolerance: float
    ) -> str:
        """
        Classify stability based on eigenvalue real parts.

        Args:
            real_parts: Real parts of eigenvalues
            imag_parts: Imaginary parts of eigenvalues
            tolerance: Tolerance for zero detection

        Returns:
            'stable', 'unstable', 'saddle', or 'neutral'
        """
        max_real = max(real_parts)
        min_real = min(real_parts)

        # Check for zero eigenvalues (neutral)
        if abs(max_real) < tolerance and abs(min_real) < tolerance:
            return "neutral"

        # Check for mixed signs (saddle point)
        if max_real > tolerance and min_real < -tolerance:
            return "saddle"

        # Check for all negative (stable)
        if max_real < -tolerance:
            return "stable"

        # Check for all positive (unstable)
        if min_real > tolerance:
            return "unstable"

        # Some eigenvalues near zero
        if abs(max_real) < tolerance or abs(min_real) < tolerance:
            return "neutral"

        return "unknown"

    def _classify_stability_detailed(
        self, real_parts: list[float], imag_parts: list[float], tolerance: float
    ) -> str:
        """
        Detailed stability classification.

        Args:
            real_parts: Real parts of eigenvalues
            imag_parts: Imaginary parts of eigenvalues
            tolerance: Tolerance for zero detection

        Returns:
            Detailed classification string
        """
        max_real = max(real_parts)
        min_real = min(real_parts)
        has_complex = any(abs(im) > tolerance for im in imag_parts)

        # Determine base stability
        if max_real < -tolerance:
            base = "stable"
        elif min_real > tolerance:
            base = "unstable"
        elif max_real > tolerance and min_real < -tolerance:
            base = "saddle"
        elif abs(max_real) < tolerance:
            base = "neutral"
        else:
            base = "unknown"

        # Determine type (node vs focus)
        if has_complex:
            type_str = "focus"
        else:
            type_str = "node"

        # Special case: saddle
        if base == "saddle":
            return f"saddle_point"

        # Combine
        if base in ["stable", "unstable"]:
            return f"{base}_{type_str}"
        elif base == "neutral":
            if has_complex:
                return "center"
            else:
                return "neutral"

        return base

    def _detect_bifurcation(
        self, real_parts: list[float], imag_parts: list[float], tolerance: float
    ) -> tuple[bool, str | None]:
        """
        Detect if system is near a bifurcation point.

        Bifurcations occur when eigenvalues cross the imaginary axis.

        Args:
            real_parts: Real parts of eigenvalues
            imag_parts: Imaginary parts of eigenvalues
            tolerance: Tolerance for detection

        Returns:
            Tuple of (is_near_bifurcation, bifurcation_type)
        """
        # Check if any eigenvalue is near imaginary axis
        near_bifurcation = False
        bifurcation_type = None

        for i, (re, im) in enumerate(zip(real_parts, imag_parts)):
            if abs(re) < tolerance * 10:  # Near zero real part
                near_bifurcation = True

                if abs(im) < tolerance:
                    # Real eigenvalue crossing zero
                    bifurcation_type = "transcritical_or_saddle_node"
                else:
                    # Complex pair crossing imaginary axis
                    bifurcation_type = "hopf"

                break

        return near_bifurcation, bifurcation_type

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

"""
Symbolic formula extraction for epimodels.

This module provides automatic extraction of ODE formulas from ContinuousModel
instances via symbolic execution using SymPy.
"""

from __future__ import annotations

import ast
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import sympy as sp
    from epimodels.continuous import ContinuousModel

from epimodels import FormulaExtractionError


@dataclass
class ModelIssue:
    """Represents a potential issue found in a model's _model method."""

    category: str
    message: str
    lineno: int | None = None
    is_critical: bool = False


def validate_model_method(model: ContinuousModel) -> list[ModelIssue]:
    """
    Inspect _model method for patterns that may fail symbolic execution.

    Args:
        model: A ContinuousModel instance

    Returns:
        List of ModelIssue objects describing potential problems
    """
    import textwrap

    issues = []

    try:
        source = inspect.getsource(model._model)
    except (TypeError, OSError) as e:
        issues.append(
            ModelIssue(
                category="source", message=f"Could not inspect _model source: {e}", is_critical=True
            )
        )
        return issues

    # Dedent the source to fix indentation issues
    source = textwrap.dedent(source)

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        issues.append(
            ModelIssue(
                category="syntax", message=f"Could not parse _model source: {e}", is_critical=True
            )
        )
        return issues

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        issues.append(
            ModelIssue(
                category="syntax", message=f"Could not parse _model source: {e}", is_critical=True
            )
        )
        return issues

    for node in ast.walk(tree):
        # Detect if statements
        if isinstance(node, ast.If):
            issues.append(
                ModelIssue(
                    category="conditional",
                    message="Model contains 'if' statements which may not work "
                    "with symbolic execution. Consider using sympy.Piecewise "
                    "or defining formulas manually.",
                    lineno=getattr(node, "lineno", None),
                    is_critical=False,
                )
            )

        # Detect while loops
        if isinstance(node, ast.While):
            issues.append(
                ModelIssue(
                    category="loop",
                    message="Model contains 'while' loops which cannot be symbolically executed. "
                    "Define formulas manually via _formulas attribute.",
                    lineno=getattr(node, "lineno", None),
                    is_critical=True,
                )
            )

        # Detect for loops
        if isinstance(node, ast.For):
            issues.append(
                ModelIssue(
                    category="loop",
                    message="Model contains 'for' loops which cannot be symbolically executed. "
                    "Define formulas manually via _formulas attribute.",
                    lineno=getattr(node, "lineno", None),
                    is_critical=True,
                )
            )

        # Detect lambda definitions
        if isinstance(node, ast.Lambda):
            issues.append(
                ModelIssue(
                    category="lambda",
                    message="Model contains lambda functions. Use sympy.Piecewise "
                    "for conditional expressions or define _formulas manually.",
                    lineno=getattr(node, "lineno", None),
                    is_critical=False,
                )
            )

    return issues


def extract_formulas(model: ContinuousModel) -> dict[str, sp.Expr]:
    """
    Extract ODE formulas from a ContinuousModel via symbolic execution.

    This function attempts to call the model's _model method with SymPy
    symbolic variables instead of numeric values, producing symbolic
    expressions for each state variable's derivative.

    Args:
        model: A ContinuousModel instance with a _model method

    Returns:
        Dict mapping state variable names to SymPy expressions

    Raises:
        FormulaExtractionError: If extraction fails

    Example:
        >>> from epimodels.continuous import SIR
        >>> model = SIR()
        >>> formulas = extract_formulas(model)
        >>> formulas['S']
        -beta*S*I/N
    """
    try:
        import sympy as sp
    except ImportError as e:
        raise FormulaExtractionError(
            model_name=getattr(model, "model_type", type(model).__name__),
            reason="SymPy is required for formula extraction",
            suggestion="Install with: pip install sympy",
        ) from e

    model_name = getattr(model, "model_type", type(model).__name__)

    # Check for _model method
    if not hasattr(model, "_model") or not callable(model._model):
        raise FormulaExtractionError(
            model_name=model_name,
            reason="Model does not have a callable _model method",
            suggestion="Ensure the model is a ContinuousModel subclass with _model defined",
        )

    # Pre-flight validation
    issues = validate_model_method(model)
    critical_issues = [i for i in issues if i.is_critical]

    if critical_issues:
        issue = critical_issues[0]
        raise FormulaExtractionError(
            model_name=model_name,
            reason=issue.message,
            suggestion=f"Define formulas manually by setting model._formulas = {{...}}",
        )

    # Issue warnings for non-critical issues
    for issue in issues:
        if not issue.is_critical:
            warnings.warn(f"{model_name}: {issue.message}", UserWarning, stacklevel=3)

    # Create symbols for state variables
    state_syms = {name: sp.Symbol(name) for name in model.state_variables}

    # Create symbols for parameters
    param_syms = {name: sp.Symbol(name) for name in model.parameters}

    # Add N (population) as it's commonly used
    param_syms["N"] = sp.Symbol("N")

    # Add time symbol
    t_sym = sp.Symbol("t")

    # Map numpy functions to sympy equivalents
    numpy_to_sympy_globals = {
        "exp": sp.exp,
        "log": sp.log,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "tanh": sp.tanh,
        "sinh": sp.sinh,
        "cosh": sp.cosh,
        "sqrt": sp.sqrt,
        "abs": sp.Abs,
        "Abs": sp.Abs,
        "pi": sp.pi,
        "Pi": sp.pi,
        "e": sp.E,
        "E": sp.E,
    }

    # Prepare state variable list in order
    y_list = [state_syms[name] for name in model.state_variables]

    try:
        # Execute _model with symbols (type ignore: intentional for symbolic execution)
        result = model._model(t_sym, y_list, param_syms)  # type: ignore
    except TypeError as e:
        raise FormulaExtractionError(
            model_name=model_name,
            reason=f"Symbolic execution failed with TypeError: {e}",
            suggestion="The _model method may use features incompatible with symbolic execution. "
            "Define _formulas manually.",
        ) from e
    except NameError as e:
        raise FormulaExtractionError(
            model_name=model_name,
            reason=f"Symbolic execution failed with NameError: {e}",
            suggestion="The _model method uses undefined variables. "
            "Ensure all variables are parameters or state variables.",
        ) from e
    except Exception as e:
        raise FormulaExtractionError(
            model_name=model_name,
            reason=f"Symbolic execution failed: {type(e).__name__}: {e}",
            suggestion="Define formulas manually via model._formulas = {...}",
        ) from e

    # Validate results
    if result is None:
        raise FormulaExtractionError(
            model_name=model_name,
            reason="_model method returned None",
            suggestion="Ensure _model returns a list of expressions",
        )

    if not isinstance(result, (list, tuple)):
        raise FormulaExtractionError(
            model_name=model_name,
            reason=f"_model returned {type(result).__name__}, expected list or tuple",
            suggestion="Ensure _model returns a list of derivative expressions",
        )

    if len(result) != len(model.state_variables):
        raise FormulaExtractionError(
            model_name=model_name,
            reason=f"_model returned {len(result)} expressions, "
            f"expected {len(model.state_variables)} (one per state variable)",
            suggestion="Ensure _model returns one expression per state variable",
        )

    # Convert results to SymPy expressions
    formulas = {}
    for name, expr in zip(model.state_variables.keys(), result):
        try:
            if isinstance(expr, (sp.Expr, sp.Number)):
                formulas[name] = expr
            elif isinstance(expr, (int, float)):
                formulas[name] = sp.sympify(expr)
            else:
                # Try to convert to SymPy expression
                formulas[name] = sp.sympify(expr)
        except Exception as e:
            raise FormulaExtractionError(
                model_name=model_name,
                reason=f"Could not convert expression for '{name}' to SymPy: {type(expr).__name__}",
                suggestion=f"Ensure the formula for '{name}' uses only mathematical operations",
            ) from e

    return formulas


def validate_formulas(model: ContinuousModel, formulas: dict[str, sp.Expr]) -> None:
    """
    Validate manually defined formulas.

    Args:
        model: The model to validate against
        formulas: Dict of state variable names to SymPy expressions

    Raises:
        ValueError: If formulas are missing or invalid
        TypeError: If formulas are not SymPy expressions
    """
    try:
        import sympy as sp
    except ImportError:
        raise ImportError("SymPy is required for formula validation")

    # Check for missing state variables
    missing_states = set(model.state_variables) - set(formulas)
    if missing_states:
        raise ValueError(f"Formulas missing for state variables: {missing_states}")

    # Check for extra formulas (warning, not error)
    extra_states = set(formulas) - set(model.state_variables)
    if extra_states:
        warnings.warn(f"Formulas defined for unknown state variables: {extra_states}", UserWarning)

    # Validate each formula is a SymPy expression
    for name, expr in formulas.items():
        if not isinstance(expr, (sp.Expr, sp.Number, int, float)):
            raise TypeError(
                f"Formula for '{name}' must be a SymPy expression, " f"got {type(expr).__name__}"
            )


def sympy_to_vfgen(expr: sp.Expr) -> str:
    """
    Convert a SymPy expression to vfgen formula string.

    Vfgen uses standard mathematical notation with some differences:
    - Power: ** becomes ^
    - Functions: exp, sin, cos, tan, tanh, etc. are standard

    Args:
        expr: A SymPy expression

    Returns:
        String representation compatible with vfgen
    """
    import sympy as sp

    # Convert to string
    # sympy uses ** for power, vfgen uses ^
    formula_str = str(expr)

    # Replace Python power operator with vfgen's ^
    # Need to be careful not to replace ** inside function names
    # Simple approach: replace ** with ^
    formula_str = formula_str.replace("**", "^")

    return formula_str


def get_free_symbols(formulas: dict[str, sp.Expr]) -> set[sp.Symbol]:
    """
    Get all free symbols used in formulas.

    Args:
        formulas: Dict of state variable names to SymPy expressions

    Returns:
        Set of SymPy Symbols used in any formula
    """
    all_symbols = set()
    for expr in formulas.values():
        if hasattr(expr, "free_symbols"):
            all_symbols.update(expr.free_symbols)
    return all_symbols

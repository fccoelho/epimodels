"""
Validation functions for parameter values, initial conditions, and constraints.
"""

import re
import ast
import operator
from typing import Any
from epimodels.exceptions import ValidationError
from epimodels.validation.specs import ParameterSpec, VariableSpec


COMPARISON_OPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.And: lambda a, b: a and b,
    ast.Or: lambda a, b: a or b,
}


def validate_parameter_value(
    name: str, value: Any, spec: ParameterSpec, all_params: dict[str, Any] | None = None
) -> list[str]:
    """
    Validate a parameter value against its specification.

    Args:
        name: Parameter name
        value: Parameter value to validate
        spec: Parameter specification
        all_params: All parameter values (for cross-parameter validation)

    Returns:
        List of error messages (empty if valid)

    Example:
        >>> spec = ParameterSpec(name="beta", symbol="β", bounds=(0, None))
        >>> errors = validate_parameter_value("beta", -0.5, spec)
        >>> len(errors)
        1
    """
    errors = []

    if value is None:
        if spec.required:
            errors.append(f"Required parameter '{name}' is None")
        return errors

    if not isinstance(value, spec.dtype):
        if spec.dtype == float and isinstance(value, int):
            value = float(value)
        elif spec.dtype == int and isinstance(value, float) and value.is_integer():
            value = int(value)
        else:
            errors.append(
                f"Parameter '{name}' has wrong type: expected {spec.dtype.__name__}, "
                f"got {type(value).__name__}"
            )
            return errors

    if spec.bounds is not None and isinstance(value, (int, float)):
        min_val, max_val = spec.bounds
        if min_val is not None and value < min_val:
            errors.append(f"Parameter '{name}' value {value} is below minimum bound {min_val}")
        if max_val is not None and value > max_val:
            errors.append(f"Parameter '{name}' value {value} exceeds maximum bound {max_val}")

    for constraint_expr in spec.constraints:
        try:
            constraint_errors = _validate_single_constraint(
                constraint_expr, name, value, all_params or {}
            )
            errors.extend(constraint_errors)
        except Exception as e:
            errors.append(
                f"Failed to evaluate constraint '{constraint_expr}' for parameter '{name}': {e}"
            )

    return errors


def validate_initial_condition(
    name: str, value: float, spec: VariableSpec, all_values: dict[str, float] | None = None
) -> list[str]:
    """
    Validate an initial condition value against its specification.

    Args:
        name: Variable name
        value: Initial condition value
        spec: Variable specification
        all_values: All initial condition values (for cross-variable validation)

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    if spec.non_negative and value < 0:
        errors.append(f"Initial condition '{name}' must be non-negative, got {value}")

    if spec.bounds is not None:
        min_val, max_val = spec.bounds
        if min_val is not None and value < min_val:
            errors.append(f"Initial condition '{name}' value {value} is below minimum {min_val}")
        if max_val is not None and value > max_val:
            errors.append(f"Initial condition '{name}' value {value} exceeds maximum {max_val}")

    for constraint_expr in spec.constraints:
        try:
            constraint_errors = _validate_single_constraint(
                constraint_expr, name, value, all_values or {}
            )
            errors.extend(constraint_errors)
        except Exception as e:
            errors.append(
                f"Failed to evaluate constraint '{constraint_expr}' for variable '{name}': {e}"
            )

    return errors


def evaluate_constraint(expression: str, context: dict[str, Any]) -> tuple[bool, str | None]:
    """
    Evaluate a constraint expression in the given context.

    Args:
        expression: Constraint expression (e.g., "beta > gamma")
        context: Dictionary mapping names to values

    Returns:
        Tuple of (is_satisfied, error_message)

    Example:
        >>> satisfied, msg = evaluate_constraint("x > y", {"x": 5, "y": 3})
        >>> satisfied
        True
    """
    try:
        result = _safe_eval_expression(expression, context)
        if isinstance(result, bool):
            return result, None
        else:
            return False, f"Expression '{expression}' did not evaluate to boolean"
    except Exception as e:
        return False, f"Failed to evaluate expression: {e}"


def _validate_single_constraint(
    constraint_expr: str, param_name: str, value: Any, all_params: dict[str, Any]
) -> list[str]:
    """
    Validate a single constraint expression for a parameter.

    The expression can use 'value' to refer to the current parameter value,
    or use the parameter name directly.
    """
    errors = []

    context = dict(all_params)
    context[param_name] = value
    context["value"] = value

    expr = constraint_expr.strip()

    if expr.startswith("value ") or " value " in expr:
        pass
    else:
        expr = expr.replace(param_name, "value", 1)

    satisfied, error_msg = evaluate_constraint(expr, context)

    if not satisfied:
        errors.append(
            f"Parameter '{param_name}' value {value} violates constraint: {constraint_expr}"
            + (f" ({error_msg})" if error_msg else "")
        )

    return errors


def _safe_eval_expression(expression: str, context: dict[str, Any]) -> Any:
    """
    Safely evaluate a constraint expression.

    Uses AST parsing to allow only safe operations.

    Args:
        expression: Expression to evaluate
        context: Variable bindings

    Returns:
        Result of evaluation

    Raises:
        ValueError: If expression contains unsafe operations
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {expression}") from e

    return _eval_node(tree.body, context)


def _eval_node(node: ast.AST, context: dict[str, Any]) -> Any:
    """
    Recursively evaluate an AST node.

    Only allows:
    - Numbers and strings
    - Variable names (looked up in context)
    - Comparison operators (==, !=, <, <=, >, >=)
    - Boolean operators (and, or)
    - Arithmetic operators (+, -, *, /, **)
    - Unary operators (+, -, not)
    """
    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.Name):
        name = node.id
        if name in context:
            return context[name]
        raise ValueError(f"Unknown variable: {name}")

    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, context)

        result = True
        prev_val = left

        for op, comparator in zip(node.ops, node.comparators):
            right = _eval_node(comparator, context)
            op_func = COMPARISON_OPS.get(type(op))

            if op_func is None:
                raise ValueError(f"Unsupported comparison operator: {type(op).__name__}")

            if not op_func(prev_val, right):
                return False

            prev_val = right

        return True

    if isinstance(node, ast.BoolOp):
        op_func = COMPARISON_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported boolean operator: {type(node.op).__name__}")

        values = [_eval_node(v, context) for v in node.values]
        result = values[0]
        for v in values[1:]:
            result = op_func(result, v)
        return result

    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, context)
        right = _eval_node(node.right, context)

        if isinstance(node.op, ast.Add):
            return left + right
        elif isinstance(node.op, ast.Sub):
            return left - right
        elif isinstance(node.op, ast.Mult):
            return left * right
        elif isinstance(node.op, ast.Div):
            return left / right
        elif isinstance(node.op, ast.Pow):
            return left**right
        elif isinstance(node.op, ast.FloorDiv):
            return left // right
        elif isinstance(node.op, ast.Mod):
            return left % right
        else:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")

    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand, context)

        if isinstance(node.op, ast.UAdd):
            return +operand
        elif isinstance(node.op, ast.USub):
            return -operand
        elif isinstance(node.op, ast.Not):
            return not operand
        else:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")

    raise ValueError(f"Unsupported expression type: {type(node).__name__}")

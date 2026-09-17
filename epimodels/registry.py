"""
Model registry: look up models by name across model families.

The library ships three model families (continuous, discrete, stochastic)
whose class names collide (e.g. ``SIR`` exists in all three). The registry
provides a string-based, collision-aware API:

Example:
    >>> from epimodels.registry import get_model, list_models
    >>> SIR = get_model("SIR", family="continuous")
    >>> model = SIR()
    >>> list_models(family="discrete")
    {'discrete': ['Influenza', 'SIR', ...]}

Custom models can be registered with the :func:`register_model` decorator:

    >>> from epimodels.registry import register_model
    >>> from epimodels.continuous import ContinuousModel
    >>> @register_model("MySIR", family="custom")
    ... class MySIR(ContinuousModel):
    ...     ...
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from epimodels import BaseModel

__all__ = [
    "get_model",
    "list_models",
    "register_model",
    "unregister_model",
]

# Lookup precedence when family="any" and the name exists in several families.
FAMILY_PRECEDENCE = ("continuous", "discrete", "stochastic", "network", "custom")

# Base classes that should not be registered as models themselves.
_BASE_CLASS_NAMES = {"BaseModel", "ContinuousModel", "DiscreteModel", "CTMCModel", "NetworkModel"}

_BUILTIN_MODULES: dict[str, list[str]] = {
    "continuous": ["epimodels.continuous.models"],
    "discrete": ["epimodels.discrete.models"],
    "stochastic": ["epimodels.stochastic.CTMC.models"],
    "network": ["epimodels.network"],
}

_REGISTRY: dict[str, dict[str, type[BaseModel]]] = {}
_discovered = False

M = TypeVar("M", bound=type)


def _discover_builtins() -> None:
    """Import built-in model modules and register their classes (once)."""
    global _discovered
    if _discovered:
        return
    from epimodels import BaseModel

    for family, modules in _BUILTIN_MODULES.items():
        for module_name in modules:
            module = importlib.import_module(module_name)
            exported = getattr(module, "__all__", None)
            if exported is None:
                exported = (
                    name
                    for name, obj in vars(module).items()
                    if inspect.isclass(obj)
                    and obj.__module__ == module.__name__
                )
            for name in exported:
                obj = getattr(module, name, None)
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, BaseModel)
                    and not inspect.isabstract(obj)
                    and name not in _BASE_CLASS_NAMES
                ):
                    _REGISTRY.setdefault(family, {})[name] = obj
    _discovered = True


def register_model(
    name: str | None = None,
    family: str = "custom",
) -> Callable[[M], M]:
    """
    Class decorator to register a model in the registry.

    Args:
        name: Registry name (defaults to the class name)
        family: Family label (default: "custom")

    Example:
        >>> @register_model("MySIR", family="custom")
        ... class MySIR(ContinuousModel):
        ...     ...
    """

    def decorator(cls: M) -> M:
        _REGISTRY.setdefault(family, {})[name or cls.__name__] = cls
        return cls

    return decorator


def unregister_model(name: str, family: str) -> None:
    """Remove a model from the registry (mainly useful for tests)."""
    _REGISTRY.get(family, {}).pop(name, None)


def get_model(name: str, family: str = "any") -> type[BaseModel]:
    """
    Look up a model class by name.

    Args:
        name: Model class name, e.g. ``"SIR"``
        family: Family to search: ``"continuous"``, ``"discrete"``,
            ``"stochastic"``, ``"custom"`` or ``"any"`` (default). When
            ``"any"`` and the name exists in multiple families, the first
            match in precedence order (continuous, discrete, stochastic,
            custom) is returned; pass an explicit ``family`` to disambiguate.

    Returns:
        The model class (not an instance).

    Raises:
        KeyError: If the model is not found (message lists available models).
        ValueError: If an unknown family is requested.
    """
    _discover_builtins()

    if family == "any":
        families = [f for f in FAMILY_PRECEDENCE if f in _REGISTRY] + [
            f for f in _REGISTRY if f not in FAMILY_PRECEDENCE
        ]
    elif family in _REGISTRY or family in _BUILTIN_MODULES:
        families = [family]
    else:
        raise ValueError(
            f"Unknown family '{family}'. Available families: "
            f"{sorted(set(_REGISTRY) | set(_BUILTIN_MODULES))}"
        )

    for fam in families:
        cls = _REGISTRY.get(fam, {}).get(name)
        if cls is not None:
            return cls

    available = {
        fam: sorted(classes) for fam, classes in _REGISTRY.items()
    }
    raise KeyError(
        f"Unknown model '{name}' (family={family!r}). "
        f"Available models: {available}"
    )


def list_models(family: str | None = None) -> dict[str, list[str]]:
    """
    List registered model names.

    Args:
        family: Restrict to one family (default: all families).

    Returns:
        Dict mapping family name to a sorted list of model names.
    """
    _discover_builtins()
    if family is not None:
        if family not in _REGISTRY:
            raise ValueError(
                f"Unknown family '{family}'. Available: {sorted(_REGISTRY)}"
            )
        return {family: sorted(_REGISTRY[family])}
    return {fam: sorted(classes) for fam, classes in _REGISTRY.items()}


def _reset_registry() -> None:
    """Test helper: clear all registrations."""
    global _discovered
    _REGISTRY.clear()
    _discovered = False

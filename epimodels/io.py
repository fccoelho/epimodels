"""
Save/load epidemic models to JSON or YAML.

Serializes the model *specification* — family, class name, state
variables, parameter symbols and the last-run parameter values, and
optionally the simulation traces — so models can be stored, shared and
reconstructed. Class lookup goes through the model registry
(:func:`epimodels.registry.get_model`), falling back to the module path
recorded at save time for custom models.

Example:
    >>> from epimodels.continuous import SIR
    >>> from epimodels.io import save_model, load_model
    >>>
    >>> model = SIR()
    >>> model([1000, 1, 0], [0, 50], 1001, {"beta": 2, "gamma": 0.1})
    >>> save_model(model, "sir_run.json", include_traces=True)
    >>> clone = load_model("sir_run.json")
    >>> clone.param_values["beta"]
    2
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from epimodels.registry import get_model

if TYPE_CHECKING:
    from epimodels import BaseModel

__all__ = ["model_to_dict", "model_from_dict", "save_model", "load_model"]

SCHEMA_KEY = "epimodels_model"


def _family_of(model: BaseModel) -> str:
    module = type(model).__module__
    for family in ("continuous", "discrete", "stochastic"):
        if f"epimodels.{family}" in module:
            return family
    return "custom"


def model_to_dict(model: BaseModel, include_traces: bool = False) -> dict[str, Any]:
    """
    Serialize a model specification to a plain dict.

    Args:
        model: The model instance.
        include_traces: Whether to embed the simulation traces (converted
            to nested lists).

    Returns:
        JSON/YAML-compatible dict describing the model.
    """
    data: dict[str, Any] = {
        SCHEMA_KEY: True,
        "family": _family_of(model),
        "model": type(model).__name__,
        "module": type(model).__module__,
        "state_variables": dict(model.state_variables),
        "parameters": dict(model.parameters),
        "param_values": {k: _plain(v) for k, v in model.param_values.items()},
    }
    if include_traces and model.traces:
        data["traces"] = {k: _plain(v) for k, v in model.traces.items()}
    return data


def model_from_dict(data: dict[str, Any]) -> BaseModel:
    """
    Reconstruct a model from its serialized specification.

    The registry is consulted first; if the model class is not registered
    (e.g. a custom model), the module recorded at save time is imported.

    Returns:
        A new model instance with parameter values (and traces, if present
        in the dict) restored.
    """
    if not data.get(SCHEMA_KEY):
        raise ValueError(
            f"Dict does not look like an epimodels model (missing '{SCHEMA_KEY}' flag)"
        )

    name = data["model"]
    family = data.get("family", "any")
    try:
        cls = get_model(name, family=family)
    except (KeyError, ValueError):
        module_path = data.get("module")
        if not module_path:
            raise
        cls = getattr(importlib.import_module(module_path), name)

    model = cls()

    if "state_variables" in data:
        model.state_variables = dict(data["state_variables"])
    if "parameters" in data:
        model.parameters = dict(data["parameters"])
    if "param_values" in data:
        model.param_values = dict(data["param_values"])
    if "traces" in data:
        model.traces = {k: _restore(v) for k, v in data["traces"].items()}
    return model


def _plain(value: Any) -> Any:
    """Convert numpy arrays / scalars to JSON-compatible Python objects."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def _restore(value: Any) -> Any:
    """Restore arrays from lists produced by :func:`_plain`."""
    if isinstance(value, dict):
        return {k: _restore(v) for k, v in value.items()}
    if isinstance(value, list):
        arr = None
        try:
            import numpy as np

            arr = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            pass
        if arr is not None and arr.size > 0:
            return arr
        return [_restore(v) for v in value]
    return value


def save_model(
    model: BaseModel, path: str | Path, include_traces: bool = False
) -> Path:
    """
    Save a model specification to a ``.json`` or ``.yaml``/``.yml`` file.

    YAML support requires pyyaml (``pip install pyyaml``).

    Returns:
        The path the model was written to.
    """
    path = Path(path)
    data = model_to_dict(model, include_traces=include_traces)

    suffix = path.suffix.lower()
    if suffix == ".json":
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    elif suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "pyyaml is required for YAML model files. Install with: pip install pyyaml"
            ) from e
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported model file extension: '{suffix}' (use .json/.yaml)")
    return path


def load_model(path: str | Path) -> BaseModel:
    """
    Load a model saved with :func:`save_model`.

    Returns:
        A reconstructed model instance.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    elif suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "pyyaml is required for YAML model files. Install with: pip install pyyaml"
            ) from e
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"Unsupported model file extension: '{suffix}' (use .json/.yaml)")

    return model_from_dict(data)

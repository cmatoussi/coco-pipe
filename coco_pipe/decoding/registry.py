"""
Decoding Registry
=================

Central registry for decoding estimators (classifiers, regressors, and FMs).
This allows instantiating models from string names in configuration files,
avoiding circular imports and simplifying the config layer.

Usage
-----
>>> from coco_pipe.decoding.registry import register_estimator, get_estimator_cls
>>>
>>> @register_estimator("MyModel")
>>> class MyModel: ...
>>>
>>> cls = get_estimator_cls("MyModel")
"""

import importlib
import pkgutil
import warnings
from typing import Callable, Dict, Type

from .capabilities import (
    EstimatorCapabilities,
    EstimatorSpec,
    get_estimator_capabilities,
    get_estimator_spec,
    list_estimator_specs,
    register_estimator_spec,
    resolve_estimator_capabilities,
)

__all__ = [
    "register_estimator",
    "register_spec",
    "get_estimator_cls",
    "list_estimators",
    "get_capabilities",
    "list_capabilities",
    "get_spec",
    "list_specs",
    "get_estimator_spec",
    "list_estimator_specs",
    "register_estimator_spec",
    "get_estimator_capabilities",
    "resolve_estimator_capabilities",
]

# Registry Storage
# Maps string alias -> class object
_ESTIMATOR_REGISTRY: Dict[str, Type] = {}
_INTERNAL_SCANNED = False


def _discover_entry_points():
    """
    Import 'coco_pipe.estimators' entry points.

    Plugins should call ``register_estimator_spec`` or ``register_estimator``
    when imported. We avoid inventing incomplete specs from string entry points.
    """
    try:
        from importlib.metadata import entry_points

        eps = entry_points(group="coco_pipe.estimators")
    except Exception:
        return

    for ep in eps:
        try:
            ep.load()
        except Exception:
            warnings.warn(f"Could not load estimator entry point '{ep.name}'")


def _discover_internal_modules():
    """
    Walk through the 'coco_pipe.decoding' subpackage and import all modules.
    This triggers the @register_estimator decorators.
    """
    package = importlib.import_module("coco_pipe.decoding")
    if not hasattr(package, "__path__"):
        return

    for _, name, ispkg in pkgutil.walk_packages(
        package.__path__, package.__name__ + "."
    ):
        try:
            importlib.import_module(name)
        except ImportError:
            # warn but continue - we don't want to crash if deep learning libs are
            # missing
            pass


# 1. Load Entry Points on startup (lazy map update only)
_discover_entry_points()


def register_estimator(name: str) -> Callable[[Type], Type]:
    """
    Decorator to register an estimator class under a specific name.

    Parameters
    ----------
    name : str
        The unique alias for the estimator (e.g., "RandomForestClassifier").
    """

    def decorator(cls: Type) -> Type:
        if name in _ESTIMATOR_REGISTRY:
            warnings.warn(f"Overwriting existing estimator registry for '{name}'")
        _ESTIMATOR_REGISTRY[name] = cls
        return cls

    return decorator


def register_spec(spec: EstimatorSpec) -> EstimatorSpec:
    """Register a typed estimator spec."""
    return register_estimator_spec(spec)


def get_estimator_cls(name: str) -> Type:
    """
    Retrieve an estimator class by name.

    Parameters
    ----------
    name : str
        Name of the estimator.

    Returns
    -------
    Type
        The class object.

    Raises
    ------
    ValueError
        If name is not found.
    """
    # 1. Check if already loaded
    if name in _ESTIMATOR_REGISTRY:
        return _ESTIMATOR_REGISTRY[name]

    # 2. Try typed spec lazy import target.
    try:
        spec = get_estimator_spec(name)
    except ValueError:
        spec = None

    if spec is not None:
        try:
            module = importlib.import_module(spec.module_path)
        except ImportError as e:
            raise ImportError(
                f"Could not load estimator '{name}' from '{spec.import_path}'. "
                f"Ensure optional dependencies are installed."
            ) from e

        if hasattr(module, spec.class_name):
            cls = getattr(module, spec.class_name)
            _ESTIMATOR_REGISTRY[name] = cls
            return cls

        # Check if the import triggered a decorator registration
        if name in _ESTIMATOR_REGISTRY:
            return _ESTIMATOR_REGISTRY[name]

    # 3. Last Ditch: Internal Discovery
    global _INTERNAL_SCANNED
    if not _INTERNAL_SCANNED:
        _discover_internal_modules()
        _INTERNAL_SCANNED = True
        if name in _ESTIMATOR_REGISTRY:
            return _ESTIMATOR_REGISTRY[name]

    if name not in _ESTIMATOR_REGISTRY:
        # Generate helpful error
        available = sorted(set(_ESTIMATOR_REGISTRY) | set(list_estimator_specs()))
        raise ValueError(
            f"Estimator '{name}' not found in registry.\n"
            f"Available estimators: {available}\n"
            f"Tip: Ensure the containing module is imported or registered via "
            f"entry points."
        )

    return _ESTIMATOR_REGISTRY[name]


def list_estimators() -> Dict[str, Type]:
    """Return a copy of the current registry."""
    # Ensure everything is discovered before listing
    global _INTERNAL_SCANNED
    if not _INTERNAL_SCANNED:
        _discover_internal_modules()
        _INTERNAL_SCANNED = True
    return dict(_ESTIMATOR_REGISTRY)


def get_capabilities(name: str) -> EstimatorCapabilities:
    """Return registered decoding capabilities for an estimator name."""
    return get_estimator_capabilities(name)


def list_capabilities() -> Dict[str, EstimatorCapabilities]:
    """Return capability metadata for known decoding estimators."""
    return {
        name: get_estimator_capabilities(name)
        for name in sorted(list_estimator_specs())
    }


def get_spec(name: str) -> EstimatorSpec:
    """Return the typed estimator spec for an estimator name."""
    return get_estimator_spec(name)


def list_specs() -> Dict[str, EstimatorSpec]:
    """Return typed estimator specs."""
    return list_estimator_specs()

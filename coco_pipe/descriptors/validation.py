"""Runtime input validation helpers for descriptor extraction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .configs import DescriptorConfig


def _normalize_channel_pooling(
    channel_pooling: str | Mapping[str, Sequence[str]],
    channel_names: list[str] | None,
) -> str | dict[str, list[str]]:
    if channel_pooling == "none":
        return "none"
    if channel_pooling == "all":
        return "all"
    if channel_names is None:
        raise ValueError(
            "`channel_names` must be passed explicitly when "
            "output.channel_pooling uses named groups."
        )
    if len(set(channel_names)) != len(channel_names):
        raise ValueError(
            "`channel_names` must be unique when output.channel_pooling uses "
            "named groups."
        )

    known_channels = set(channel_names)
    assigned: dict[str, str] = {}
    normalized: dict[str, list[str]] = {}
    for group_name, members in channel_pooling.items():
        normalized_members = [str(member) for member in members]
        for member in normalized_members:
            if member not in known_channels:
                raise ValueError(
                    f"output.channel_pooling['{group_name}'] references unknown "
                    f"channel '{member}'."
                )
            if member in assigned:
                raise ValueError(
                    f"Channel '{member}' is assigned to multiple channel_pooling "
                    "groups: "
                    f"'{assigned[member]}' and '{group_name}'."
                )
            assigned[member] = group_name
        normalized[str(group_name)] = normalized_members
    return normalized


def validate_runtime_inputs(
    config: DescriptorConfig,
    *,
    X: Any,
    ids: Sequence[Any] | np.ndarray | None = None,
    channel_names: Sequence[str] | np.ndarray | None = None,
    sfreq: float | None = None,
) -> dict[str, Any]:
    """Validate explicit runtime inputs against the descriptor contract.

    Parameters
    ----------
    config : DescriptorConfig
        Parsed descriptor config defining the expected runtime contract.
    X : Any
        Candidate signal array expected to coerce to shape
        ``(n_obs, n_channels, n_times)``.
    ids, channel_names, sfreq
        Optional runtime inputs aligned with the observation or channel axes.

    Returns
    -------
    dict[str, Any]
        Normalized runtime inputs ready for pipeline and extractor dispatch.

    Raises
    ------
    ValueError
        If array dimensionality, identifier alignment, sampling frequency, or
        explicit channel-name requirements are violated.
    """
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim != 3:
        raise ValueError(
            "Descriptors expect 3D input in 'obs_channel_time' layout; "
            f"got shape {X_arr.shape}."
        )

    n_obs, n_channels, _ = X_arr.shape

    sfreq_required = (
        config.input.require_sfreq
        or config.families.bands.enabled
        or config.families.parametric.enabled
        or (
            config.families.complexity.enabled
            and "spectral_entropy" in config.families.complexity.measures
        )
    )
    if sfreq_required:
        if sfreq is None:
            raise ValueError(
                "`sfreq` must be passed explicitly for the enabled descriptor families."
            )
        if sfreq <= 0:
            raise ValueError("`sfreq` must be positive.")

    channel_names_out = None
    channel_names_required = config.input.require_channel_names or (
        any(
            getattr(config.families, family_name).enabled
            for family_name in ("bands", "parametric", "complexity")
        )
        and config.output.channel_pooling != "all"
    )
    if channel_names is not None:
        channel_names_out = [str(name) for name in np.asarray(channel_names).tolist()]
        if len(channel_names_out) != n_channels:
            raise ValueError(
                f"`channel_names` must align with n_channels={n_channels}; "
                f"got {len(channel_names_out)}."
            )
    elif channel_names_required:
        raise ValueError(
            "`channel_names` must be passed explicitly for channel-resolved output."
        )

    ids_out = None
    if ids is not None:
        ids_out = np.asarray(ids)
        if ids_out.shape[0] != n_obs:
            raise ValueError(
                f"`ids` must align with n_obs={n_obs}; got shape {ids_out.shape}."
            )

    return {
        "X": X_arr,
        "ids": ids_out,
        "channel_names": channel_names_out,
        "channel_pooling": _normalize_channel_pooling(
            config.output.channel_pooling,
            channel_names_out,
        ),
        "sfreq": sfreq,
    }

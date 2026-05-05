"""Runtime input validation helpers for descriptor extraction."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .configs import DescriptorConfig


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
    channel_names_required = config.input.require_channel_names or any(
        getattr(config.families, family_name).enabled
        for family_name in ("bands", "parametric", "complexity")
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
        "sfreq": sfreq,
    }

"""
Utilities for descriptor extractors.

This module contains small pure helpers shared across descriptor extractors.
They cover:

- normalized failure-record creation
- descriptor-level channel pooling after per-channel values are computed

Notes
-----
Pooling helpers in this module are descriptor-level only:

- ``"none"`` keeps one descriptor column per input channel
- ``"all"`` averages descriptor values across all channels
- a mapping pools descriptor values within named groups and leaves ungrouped
  channels unchanged

Author: Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from __future__ import annotations

from typing import Any

import numpy as np


def make_failure_record(
    family: str,
    obs_index: int,
    obs_id: Any = None,
    channel_index: int | None = None,
    channel_name: str | None = None,
    exception_type: str | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    """Create one normalized extractor failure record.

    Parameters
    ----------
    family : str
        Canonical family name that raised or collected the failure.
    obs_index : int
        Global observation index in the original input array.
    obs_id : Any, optional
        Optional user-provided observation identifier aligned with
        ``obs_index``.
    channel_index : int, optional
        Channel index associated with the failure.
    channel_name : str, optional
        Explicit channel label associated with the failure.
    exception_type : str
        Exception class name or normalized failure type.
    message : str
        Stable human-readable failure description.

    Returns
    -------
    dict[str, Any]
        Failure record compatible with ``result["failures"]``.
    """
    return {
        "family": family,
        "obs_index": obs_index,
        "obs_id": obs_id,
        "channel_index": channel_index,
        "channel_name": channel_name,
        "exception_type": exception_type,
        "message": message,
    }


def average_channel_matrix(values: np.ndarray) -> np.ndarray:
    """Average a ``(n_obs, n_channels)`` descriptor matrix across channels.

    Parameters
    ----------
    values : np.ndarray
        Descriptor matrix with shape ``(n_obs, n_channels)``. A 1D vector is
        returned unchanged.

    Returns
    -------
    np.ndarray
        Vector with shape ``(n_obs,)`` containing the NaN-aware mean across the
        channel axis for each observation.

    Notes
    -----
    Rows containing no finite values yield ``NaN``.
    """
    if values.ndim == 1:
        return values
    out = np.empty(values.shape[0], dtype=float)
    for idx, row in enumerate(values):
        finite = row[np.isfinite(row)]
        out[idx] = np.nan if finite.size == 0 else float(finite.mean())
    return out


def pool_channel_descriptor_matrix(
    values: np.ndarray,
    channel_names: list[str],
    channel_pooling: str | dict[str, list[str]],
) -> tuple[np.ndarray, list[str]]:
    """Pool per-channel descriptor values into the public output layout.

    Parameters
    ----------
    values : np.ndarray
        Descriptor matrix with shape ``(n_obs, n_channels)``.
    channel_names : list of str
        Channel labels aligned with the columns of ``values``.
    channel_pooling : {"none", "all"} or dict of str to list of str
        Pooling specification coming from ``output.channel_pooling``.

    Returns
    -------
    tuple
        ``(X_pooled, scopes)`` where ``X_pooled`` is the pooled descriptor
        matrix and ``scopes`` is the aligned list of channel-scope tokens used
        by the extractor base class to build final descriptor names.

    Raises
    ------
    ValueError
        If ``values`` is not 2D.

    Notes
    -----
    Grouped outputs preserve first-sensor order: the first channel belonging to
    a group determines where that group appears in the output column order.
    Channels not assigned to a group remain as standalone outputs.
    Channel labels that already carry the public ``"ch-"`` scope prefix are
    preserved as-is instead of receiving a second prefix.

    Examples
    --------
    Given ``channel_names=["Fz", "Cz", "Pz"]``:

    - ``channel_pooling="none"`` returns scopes
      ``["ch-Fz", "ch-Cz", "ch-Pz"]``
    - ``channel_pooling="all"`` returns scopes ``["ch-all"]``
    - ``channel_pooling={"Frontal": ["Fz", "Cz"]}`` returns scopes
      ``["chgrp-Frontal", "ch-Pz"]``
    """
    if values.ndim != 2:
        raise ValueError("pool_channel_descriptor_matrix expects a 2D matrix.")

    def channel_scope(channel_name: str) -> str:
        return channel_name if channel_name.startswith("ch-") else f"ch-{channel_name}"

    if channel_pooling == "none":
        return values, [channel_scope(channel_name) for channel_name in channel_names]
    if channel_pooling == "all":
        return average_channel_matrix(values)[:, None], ["ch-all"]

    channel_to_index = {name: idx for idx, name in enumerate(channel_names)}
    member_to_group = {
        member: group_name
        for group_name, members in channel_pooling.items()
        for member in members
    }

    grouped_columns: list[np.ndarray] = []
    scopes: list[str] = []
    emitted_groups: set[str] = set()

    for channel_name in channel_names:
        group_name = member_to_group.get(channel_name)
        if group_name is None:
            grouped_columns.append(values[:, channel_to_index[channel_name]][:, None])
            scopes.append(channel_scope(channel_name))
            continue
        if group_name in emitted_groups:
            continue
        member_indices = [
            channel_to_index[member] for member in channel_pooling[group_name]
        ]
        grouped_columns.append(
            average_channel_matrix(values[:, member_indices])[:, None]
        )
        scopes.append(f"chgrp-{group_name}")
        emitted_groups.add(group_name)

    if not grouped_columns:
        return np.empty((values.shape[0], 0), dtype=float), []
    return np.concatenate(grouped_columns, axis=1), scopes

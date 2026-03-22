"""
Base interfaces for descriptor extraction backends.

This module defines the internal contracts shared by built-in descriptor
extractors. The module exposes:

- `BaseDescriptorExtractor` for families that consume validated raw signal
  batches
- `BasePSDDescriptorExtractor` for families that consume shared PSD batches
- `_DescriptorBlock` as the private family output payload

The surrounding descriptors stack uses these interfaces to provide:

- explicit runtime dispatch from `DescriptorPipeline`
- deterministic descriptor naming and channel reduction helpers
- family-wise metadata and failure collection
- safe merging of family outputs into one stable result dictionary

Notes
-----
`BaseDescriptorExtractor` is an internal extension point for descriptor
families. Unlike dim-reduction reducers, descriptor extractors are stateless at
runtime and do not expose `fit`, persistence, or model objects.

Examples
--------
The shared finalization helper converts per-channel descriptor values into the
public column naming convention based on ``output.channel_pooling``:

- ``channel_pooling="none"``:
  ``band_abs_alpha_ch-Fz``, ``band_abs_alpha_ch-Cz``
- ``channel_pooling="all"``:
  ``band_abs_alpha_ch-all``
- ``channel_pooling={"Frontal": ["Fz", "Cz"]}``:
  ``band_abs_alpha_chgrp-Frontal`` plus any ungrouped channels such as
  ``band_abs_alpha_ch-Pz``

Author: Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..configs import DescriptorRuntimeConfig
from .utils import pool_channel_descriptor_matrix

__all__ = ["BaseDescriptorExtractor", "BasePSDDescriptorExtractor"]


@dataclass(slots=True)
class _DescriptorBlock:
    """Private in-memory descriptor payload for one family.

    Attributes
    ----------
    family : str
        Canonical family name that produced the block.
    X : np.ndarray
        Family-specific descriptor matrix aligned on the observation axis.
    descriptor_names : list of str
        Deterministic column names aligned with the columns of ``X``.
    meta : dict
        Family-specific metadata to preserve under the merged result.
    failures : list of dict
        Normalized failure records collected during extraction.

    Notes
    -----
    ``X.shape[0]`` must always match the input observation count seen by the
    extractor, and ``len(descriptor_names)`` must always match ``X.shape[1]``.
    The pipeline depends on these alignment guarantees when merging family
    outputs.
    """

    family: str
    X: np.ndarray
    descriptor_names: list[str]
    meta: dict[str, Any] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)


class BaseDescriptorExtractor(ABC):
    """
    Abstract base class for descriptor extraction families.

    Subclasses receive already validated NumPy inputs and must return one
    `_DescriptorBlock` aligned on the observation axis. The base class keeps
    the extractor API narrow and provides a shared helper for channel
    finalization and deterministic descriptor naming.

    Parameters
    ----------
    config : Any
        Typed family configuration parsed by `DescriptorConfig`.

    Attributes
    ----------
    config : Any
        Stored family-specific configuration object.
    family_name : str
        Stable family identifier used in failure records and merged metadata.

    Notes
    -----
    Extractors are stateless at runtime. They do not learn parameters across
    calls; all runtime state is provided explicitly through `extract()`.

    Concrete extractors are expected to:

    1. compute family-specific values with shape ``(n_obs, n_channels)`` for
       each metric
    2. pass those values through :meth:`_finalize_descriptor`
    3. return one `_DescriptorBlock` with aligned names, metadata, and failures

    Examples
    --------
    A minimal concrete extractor typically looks like:

    >>> class MeanOverTimeExtractor(BaseDescriptorExtractor):
    ...     family_name = "toy"
    ...
    ...     def extract(
    ...         self,
    ...         X,
    ...         sfreq,
    ...         channel_names,
    ...         channel_pooling,
    ...         ids,
    ...         runtime,
    ...     ):
    ...         values = X.mean(axis=-1)
    ...         X_out, names = self._finalize_descriptor(
    ...             values,
    ...             family_prefix="toy",
    ...             metric_name="mean",
    ...             channel_names=channel_names,
    ...             channel_pooling=channel_pooling,
    ...         )
    ...         return _DescriptorBlock(
    ...             family=self.family_name,
    ...             X=X_out,
    ...             descriptor_names=names,
    ...         )
    """

    family_name = "base"

    def __init__(self, config: Any):
        """Store the typed family configuration."""
        self.config = config

    @property
    def capabilities(self) -> dict[str, Any]:
        """Return static extractor capability metadata.

        Returns
        -------
        dict[str, Any]
            Static metadata describing optional dependencies and general
            execution properties for the extractor.

        Notes
        -----
        The descriptors pipeline currently uses this mapping only as lightweight
        backend metadata. It is intentionally much smaller than the reducer
        capability surface in `dim_reduction`.
        """
        return {
            "requires_sfreq": False,
            "supports_batching": True,
            "supports_channelwise": True,
            "deterministic": True,
            "optional_dependencies": [],
        }

    @abstractmethod
    def extract(
        self,
        X: np.ndarray,
        sfreq: float | None,
        channel_names: list[str] | None,
        channel_pooling: str | dict[str, list[str]],
        ids: np.ndarray | None,
        runtime: DescriptorRuntimeConfig,
        obs_offset: int = 0,
    ) -> _DescriptorBlock:
        """Extract descriptors from a validated input array.

        Parameters
        ----------
        X : np.ndarray
            Input array with shape ``(n_obs, n_channels, n_times)``.
        sfreq : float, optional
            Sampling frequency in Hertz.
        channel_names : list of str, optional
            Explicit channel labels aligned with axis 1 of ``X``.
        channel_pooling : {"none", "all"} or dict
            Descriptor-level channel pooling policy applied after per-channel
            descriptors are computed.
        ids : np.ndarray, optional
            Observation identifiers aligned with axis 0 of ``X``.
        runtime : DescriptorRuntimeConfig
            Runtime execution controls shared across extractors.
        obs_offset : int, default=0
            Global observation offset applied to any collected failure records.

        Returns
        -------
        _DescriptorBlock
            Family-specific descriptor matrix plus metadata and failures.

        Raises
        ------
        ImportError
            If an optional backend required by the extractor is unavailable.
        ValueError
            If the extractor encounters an invalid runtime condition and the
            configured error policy requires raising.

        Notes
        -----
        The recommended pattern is to keep family-specific computation local to
        the extractor and delegate all public channel naming and channel pooling
        behavior to :meth:`_finalize_descriptor`.
        """

    def _finalize_descriptor(
        self,
        values: np.ndarray,
        family_prefix: str,
        metric_name: str,
        channel_names: list[str] | None,
        channel_pooling: str | dict[str, list[str]] = "none",
    ) -> tuple[np.ndarray, list[str]]:
        """Pool channels and build deterministic descriptor names.

        Parameters
        ----------
        values : np.ndarray
            Family metric values with shape ``(n_obs, n_channels)`` or
            ``(n_obs,)``.
        family_prefix : str
            Stable family prefix, for example ``"band"`` or ``"param"``.
        metric_name : str
            Family-local metric identifier used in the descriptor name.
        channel_names : list of str, optional
            Channel labels used when building channel-resolved descriptor names.
        channel_pooling : {"none", "all"} or dict, default="none"
            Descriptor-level channel pooling policy.

        Returns
        -------
        tuple
            ``(X_metric, names)`` where ``X_metric`` is the finalized metric
            matrix and ``names`` is the aligned list of descriptor names.

        Notes
        -----
        This helper assumes ``values`` already represents descriptor values, not
        raw signals. Pooling therefore always happens at the descriptor level:

        - ``"none"`` keeps one column per sensor
        - ``"all"`` averages descriptor values across all sensors
        - a mapping averages descriptor values within each named group and keeps
          ungrouped sensors as individual columns

        Examples
        --------
        Given ``channel_names=["Fz", "Cz", "Pz"]`` and
        ``metric_name="abs_alpha"``:

        - ``channel_pooling="none"`` yields
          ``["band_abs_alpha_ch-Fz", "band_abs_alpha_ch-Cz", "band_abs_alpha_ch-Pz"]``
        - ``channel_pooling="all"`` yields
          ``["band_abs_alpha_ch-all"]``
        - ``channel_pooling={"Frontal": ["Fz", "Cz"]}`` yields
          ``["band_abs_alpha_chgrp-Frontal", "band_abs_alpha_ch-Pz"]``
        """
        if values.ndim == 1:
            values = values[:, None]
        pooled_values, scopes = pool_channel_descriptor_matrix(
            values,
            channel_names=channel_names or [],
            channel_pooling=channel_pooling,
        )
        names = ["_".join((family_prefix, metric_name, scope)) for scope in scopes]
        return pooled_values, names


class BasePSDDescriptorExtractor(BaseDescriptorExtractor):
    """
    Abstract base class for descriptor families that consume PSD batches.

    PSD-consuming families still participate in the shared descriptor contract,
    but they expose one additional explicit entry point:

    - `extract_psd(...)` consumes precomputed `psds, freqs`
    - `psd_request()` tells the planner which PSD range and method is needed

    This keeps the generic raw-signal interface narrow while still giving the
    planner one formal PSD-consumer contract shared by spectral and parametric
    families.

    Notes
    -----
    PSD consumers may still expose `extract()` to satisfy the generic family
    interface, but the shared planner uses `psd_request()` and `extract_psd()`
    exclusively once PSD intermediates have been materialized.
    """

    @abstractmethod
    def psd_request(self) -> dict[str, Any]:
        """Describe the PSD requirements for the shared planner.

        Returns
        -------
        dict[str, Any]
            Minimal request payload containing the PSD method and the required
            frequency range for this family.
        """

    def parametric_fit_requirements(self) -> dict[str, Any]:
        """Describe whether this PSD consumer needs a shared parametric fit.

        Returns
        -------
        dict[str, Any]
            Shared-fit requirements with the keys:

            - `needed`
            - `metrics`
            - `periodic_psds`
            - `config`
        """
        return {
            "needed": False,
            "metrics": False,
            "periodic_psds": False,
            "config": None,
        }

    @abstractmethod
    def extract_psd(
        self,
        psds: np.ndarray,
        freqs: np.ndarray,
        channel_names: list[str] | None,
        channel_pooling: str | dict[str, list[str]],
        ids: np.ndarray | None,
        runtime: DescriptorRuntimeConfig,
        obs_offset: int = 0,
        fit_batch: Any | None = None,
    ) -> _DescriptorBlock:
        """Extract descriptors from explicit PSD intermediates.

        Parameters
        ----------
        psds : np.ndarray
            PSD batch with shape ``(n_obs, n_channels, n_freqs)``.
        freqs : np.ndarray
            Frequency grid aligned with the last axis of ``psds``.
        channel_names : list of str, optional
            Explicit channel labels aligned with the channel axis.
        channel_pooling : {"none", "all"} or dict
            Descriptor-level channel pooling policy.
        ids : np.ndarray, optional
            Observation identifiers aligned with the observation axis.
        runtime : DescriptorRuntimeConfig
            Runtime execution controls shared across extractors.
        obs_offset : int, default=0
            Global observation offset applied to collected failure records.
        fit_batch : Any, optional
            Additional shared fit payload required by some PSD consumers.

        Returns
        -------
        _DescriptorBlock
            Family-specific descriptor block aligned with the input PSD batch.
        """

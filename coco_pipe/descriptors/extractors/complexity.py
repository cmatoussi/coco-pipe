"""
Complexity descriptor extraction backend.

This module implements the built-in complexity family for
`coco_pipe.descriptors`. The extractor operates on already segmented NumPy
inputs with shape ``(n_obs, n_channels, n_times)`` and computes one or more
complexity measures per sensor, per observation.

Notes
-----
The complexity family prefers batched backend calls when the selected library
supports them. In the current implementation:

- `spectral_entropy`, `hjorth_mobility`, and `hjorth_complexity` use batched
  `antropy` calls over flattened observation-channel units
- `sample_entropy`, `perm_entropy`, and `lziv_complexity` are still evaluated
  one 1D signal at a time

Author: Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...utils import import_optional_dependency
from ..configs import ComplexityDescriptorConfig
from .base import BaseDescriptorExtractor, _DescriptorBlock
from .utils import make_failure_record


class ComplexityDescriptorExtractor(BaseDescriptorExtractor):
    """
    Complexity descriptor extractor.

    This extractor computes scalar complexity measures for each observation and
    sensor in a validated descriptor input array. It is intended for signals
    that are already segmented upstream, such as epochs, windows, or trial
    blocks.

    Parameters
    ----------
    config : ComplexityDescriptorConfig
        Parsed family configuration controlling the selected measures, backend,
        and any per-measure keyword arguments.

    Attributes
    ----------
    config : ComplexityDescriptorConfig
        Stored typed configuration for the complexity family.
    family_name : str
        Stable family identifier used in metadata and failure records.

    Notes
    -----
    The extractor always computes descriptor values per sensor first. Public
    output pooling, such as `channel_pooling="all"` or grouped channel pooling,
    is applied afterward through :meth:`BaseDescriptorExtractor._finalize_descriptor`.

    When `antropy` is selected, the extractor uses batched calls where the
    backend supports them and falls back to scalar loops for measures that are
    inherently one-signal-at-a-time in the current backend API.
    """

    family_name = "complexity"

    def __init__(self, config: ComplexityDescriptorConfig):
        super().__init__(config)
        self.config = config

    @property
    def capabilities(self) -> dict[str, Any]:
        """Return static complexity extractor capability metadata.

        Returns
        -------
        dict[str, Any]
            Capability metadata describing sampling-rate requirements and the
            optional backends used by the complexity family.

        Notes
        -----
        `spectral_entropy` requires an explicit sampling rate, while the other
        currently supported measures do not.
        """
        return {
            **super().capabilities,
            "requires_sfreq": "spectral_entropy" in self.config.measures,
            "optional_dependencies": ["antropy", "neurokit2"],
        }

    def _load_antropy(self):
        """Import `antropy` lazily when the configured backend needs it.

        Returns
        -------
        module
            Imported `antropy` module.

        Raises
        ------
        ImportError
            If `antropy` is not installed.
        """
        return import_optional_dependency(
            lambda: __import__("antropy"),
            feature="complexity descriptor extraction",
            dependency="antropy",
            install_hint="pip install coco-pipe[descriptors]",
        )

    def _load_neurokit(self):
        """Import `neurokit2` lazily when the configured backend needs it.

        Returns
        -------
        module
            Imported `neurokit2` module.

        Raises
        ------
        ImportError
            If `neurokit2` is not installed.
        """
        return import_optional_dependency(
            lambda: __import__("neurokit2"),
            feature="neurokit complexity descriptor extraction",
            dependency="neurokit2",
            install_hint="pip install coco-pipe[descriptors]",
        )

    def extract(
        self,
        X: np.ndarray,
        sfreq: float | None,
        channel_names: list[str] | None,
        channel_pooling: str | dict[str, list[str]],
        ids: np.ndarray | None,
        runtime,
        obs_offset: int = 0,
    ) -> _DescriptorBlock:
        """Extract complexity descriptors from segmented multi-channel data.

        Parameters
        ----------
        X : np.ndarray
            Input array with shape ``(n_obs, n_channels, n_times)``. Each row
            already represents one observation segment produced upstream.
        sfreq : float, optional
            Sampling frequency in Hertz. Required when
            `spectral_entropy` is requested.
        channel_names : list of str, optional
            Explicit channel labels aligned with axis 1 of ``X``. If omitted,
            fallback names ``"ch-0"``, ``"ch-1"``, ... are used internally.
        channel_pooling : {"none", "all"} or dict
            Descriptor-level channel pooling policy applied after per-sensor
            complexity values are computed.
        ids : np.ndarray, optional
            Observation identifiers aligned with axis 0 of ``X``.
        runtime : DescriptorRuntimeConfig
            Runtime execution controls shared across descriptor families.
        obs_offset : int, default=0
            Global observation offset added to any collected failure records
            when this extractor is called on one observation batch.

        Returns
        -------
        _DescriptorBlock
            Complexity-family descriptor block aligned with the input
            observation axis.

        Raises
        ------
        ImportError
            If the configured optional backend is unavailable.
        ValueError
            If a requested measure is unsupported by the selected backend, or
            if runtime error handling is configured to raise on a numerical or
            backend failure.

        Notes
        -----
        The extractor uses a mixed execution strategy:

        - batched `antropy` calls for `spectral_entropy`,
          `hjorth_mobility`, and `hjorth_complexity`
        - scalar calls for `sample_entropy`, `perm_entropy`, and
          `lziv_complexity`

        Non-finite outputs are converted to `NaN` and recorded under
        ``failures`` unless `runtime.on_error == "raise"`, in which case the
        extractor fails immediately.

        Examples
        --------
        With ``channel_pooling="none"`` and
        ``channel_names=["Fz", "Cz"]``, a requested measure such as
        ``perm_entropy`` yields channel-resolved names like
        ``complexity_perm_entropy_ch-Fz`` and
        ``complexity_perm_entropy_ch-Cz``.

        With ``channel_pooling="all"``, the same metric yields one pooled
        column named ``complexity_perm_entropy_ch-all``.
        """
        channel_names = channel_names or [f"ch-{idx}" for idx in range(X.shape[1])]

        descriptor_names: list[str] | None = None
        failures: list[dict[str, Any]] = []
        metric_arrays = {
            measure: np.full((X.shape[0], X.shape[1]), np.nan, dtype=float)
            for measure in self.config.measures
        }
        measure_kwargs = {
            measure: dict(self.config.measure_kwargs.get(measure, {}))
            for measure in self.config.measures
        }
        flat_signals = X.reshape(-1, X.shape[-1])
        batched_outputs: dict[str, np.ndarray] = {}
        scalar_dispatch: dict[str, Any] = {}

        if self.config.backend in {"antropy", "auto"}:
            ant = self._load_antropy()

            if "spectral_entropy" in self.config.measures:
                batched_outputs["spectral_entropy"] = np.asarray(
                    ant.spectral_entropy(
                        flat_signals,
                        sf=sfreq,
                        axis=-1,
                        **measure_kwargs["spectral_entropy"],
                    ),
                    dtype=float,
                )

            if (
                "hjorth_mobility" in self.config.measures
                or "hjorth_complexity" in self.config.measures
            ):
                mobility, complexity = ant.hjorth_params(
                    flat_signals,
                    axis=-1,
                )
                if "hjorth_mobility" in self.config.measures:
                    batched_outputs["hjorth_mobility"] = np.asarray(
                        mobility,
                        dtype=float,
                    )
                if "hjorth_complexity" in self.config.measures:
                    batched_outputs["hjorth_complexity"] = np.asarray(
                        complexity,
                        dtype=float,
                    )

            scalar_dispatch = {
                "sample_entropy": lambda signal, kwargs, sfreq: float(
                    ant.sample_entropy(signal, **kwargs)
                ),
                "perm_entropy": lambda signal, kwargs, sfreq: float(
                    ant.perm_entropy(signal, **kwargs)
                ),
                "lziv_complexity": lambda signal, kwargs, sfreq: float(
                    ant.lziv_complexity(
                        (signal > np.median(signal)).astype(int),
                        **kwargs,
                    )
                ),
            }
        else:
            nk = self._load_neurokit()
            scalar_dispatch = {
                "sample_entropy": lambda signal, kwargs, sfreq: float(
                    nk.entropy_sample(signal, **kwargs)[0]
                ),
                "perm_entropy": lambda signal, kwargs, sfreq: float(
                    nk.entropy_permutation(signal, **kwargs)[0]
                ),
                "spectral_entropy": lambda signal, kwargs, sfreq: float(
                    nk.entropy_spectral(
                        signal,
                        sampling_rate=sfreq,
                        **kwargs,
                    )[0]
                ),
            }

        for measure, flat_values in batched_outputs.items():
            values = np.asarray(flat_values, dtype=float).reshape(
                X.shape[0],
                X.shape[1],
            )
            metric_arrays[measure][:] = np.where(np.isfinite(values), values, np.nan)
            bad_positions = np.argwhere(~np.isfinite(values))
            if bad_positions.size == 0:
                continue

            message = f"Complexity measure '{measure}' produced a non-finite result."
            if runtime.on_error == "raise":
                raise ValueError(message)

            for obs_rel, unit_idx in bad_positions:
                failures.append(
                    make_failure_record(
                        family=self.family_name,
                        obs_index=obs_offset + int(obs_rel),
                        obs_id=None if ids is None else ids[int(obs_rel)],
                        channel_index=int(unit_idx),
                        channel_name=channel_names[int(unit_idx)],
                        exception_type="NumericalIssue",
                        message=message,
                    )
                )

        scalar_measures = [
            measure
            for measure in self.config.measures
            if measure not in batched_outputs
        ]
        unsupported = sorted(set(scalar_measures) - set(scalar_dispatch))
        if unsupported:
            raise ValueError(
                f"Measures {unsupported} are not supported by backend "
                f"'{self.config.backend}'."
            )

        for obs_rel in range(X.shape[0]):
            unit_signals = X[obs_rel]
            obs_id = None if ids is None else ids[obs_rel]

            for unit_idx, signal in enumerate(unit_signals):
                for measure in scalar_measures:
                    try:
                        value = scalar_dispatch[measure](
                            signal,
                            measure_kwargs[measure],
                            sfreq,
                        )
                        if np.isfinite(value):
                            metric_arrays[measure][obs_rel, unit_idx] = float(value)
                        else:
                            if runtime.on_error == "raise":
                                raise ValueError(
                                    "Complexity measure produced a non-finite result."
                                )
                            failures.append(
                                make_failure_record(
                                    family=self.family_name,
                                    obs_index=obs_offset + obs_rel,
                                    obs_id=obs_id,
                                    channel_index=unit_idx,
                                    channel_name=channel_names[unit_idx],
                                    exception_type="NumericalIssue",
                                    message=(
                                        "Complexity measure "
                                        f"'{measure}' produced a non-finite result."
                                    ),
                                )
                            )
                    except Exception as exc:  # pragma: no cover - hit via failure tests
                        if isinstance(exc, ImportError):
                            raise
                        if runtime.on_error == "raise":
                            raise
                        failures.append(
                            make_failure_record(
                                family=self.family_name,
                                obs_index=obs_offset + obs_rel,
                                obs_id=obs_id,
                                channel_index=unit_idx,
                                channel_name=channel_names[unit_idx],
                                exception_type=type(exc).__name__,
                                message=str(exc),
                            )
                        )

        chunk_features: list[np.ndarray] = []
        chunk_names: list[str] = []
        for measure in self.config.measures:
            feature, names = self._finalize_descriptor(
                metric_arrays[measure],
                family_prefix="complexity",
                metric_name=measure,
                channel_names=channel_names,
                channel_pooling=channel_pooling,
            )
            chunk_features.append(feature)
            chunk_names.extend(names)

        descriptor_names = chunk_names

        return _DescriptorBlock(
            family=self.family_name,
            X=np.concatenate(chunk_features, axis=1)
            if chunk_features
            else np.empty((X.shape[0], 0)),
            descriptor_names=descriptor_names or [],
            meta={
                "backend": self.config.backend,
                "measures": list(self.config.measures),
                "batched_measures": sorted(batched_outputs),
            },
            failures=failures,
        )

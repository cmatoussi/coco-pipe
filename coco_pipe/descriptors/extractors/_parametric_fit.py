"""
Shared specparam fitting for PSD-consuming descriptor paths.

This module holds the reusable fitting step used by the descriptors planner and
by extractors that consume explicit parametric-fit intermediates. It does not
define descriptor names or output pooling. It only:

- fit specparam models on PSD batches
- collect scalar fit metrics in aligned arrays
- optionally reconstruct periodic-only PSDs for corrected band outputs
- return one batch-scoped payload that downstream extractors can consume

Author: Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...utils import import_optional_dependency
from ..configs import ParametricDescriptorConfig


@dataclass(slots=True)
class _ParametricFitBatch:
    """
    Batch-scoped parametric fit payload shared across PSD consumers.

    Attributes
    ----------
    freqs : np.ndarray
        Frequency grid used for the fitted spectra.
    metrics : dict of str to np.ndarray
        Scalar metric arrays aligned to ``(n_obs, n_channels)`` for each
        requested parametric metric.
    errors : list of tuple
        Collected fit failures as ``(obs_index, channel_index,
        exception_type, message)``.
    periodic_psds : np.ndarray | None
        Periodic-only PSDs aligned to ``(n_obs, n_channels, n_freqs)`` when
        corrected spectral outputs request them.
    meta : dict
        Lightweight fit metadata propagated into downstream descriptor blocks.
    """

    freqs: np.ndarray
    metrics: dict[str, np.ndarray]
    errors: list[tuple[int, int, str, str]] = field(default_factory=list)
    periodic_psds: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def fit_single_spectrum(
    freqs: np.ndarray,
    spectrum: np.ndarray,
    config: ParametricDescriptorConfig,
    need_periodic_psd: bool = False,
) -> tuple[dict[str, float], np.ndarray | None]:
    """
    Fit one specparam model to one PSD spectrum.

    Parameters
    ----------
    freqs : np.ndarray
        Frequency grid for the input spectrum.
    spectrum : np.ndarray
        One PSD spectrum aligned with ``freqs``.
    config : ParametricDescriptorConfig
        Parsed parametric fit configuration.
    need_periodic_psd : bool, default=False
        Whether to reconstruct the periodic-only PSD from the fitted model.

    Returns
    -------
    tuple[dict[str, float], np.ndarray | None]
        Scalar fit metrics and, when requested, the periodic-only PSD on the
        same frequency grid.

    Raises
    ------
    ValueError
        If the spectrum is constant or entirely non-finite.
    RuntimeError
        If specparam fails to produce a usable model or if reconstructed model
        components become non-finite.
    """
    finite = spectrum[np.isfinite(spectrum)]
    if finite.size == 0 or np.ptp(finite) < np.finfo(float).eps:
        raise ValueError("Parametric fitting requires a non-constant spectrum.")

    SpectralModel = import_optional_dependency(
        lambda: (
            __import__(
                "specparam.models",
                fromlist=["SpectralModel"],
            ).SpectralModel
        ),
        feature="parametric descriptor extraction",
        dependency="specparam",
        install_hint="pip install coco-pipe[descriptors]",
    )
    model = SpectralModel(
        aperiodic_mode=config.aperiodic_mode,
        peak_width_limits=config.peak_width_limits,
        max_n_peaks=config.max_n_peaks,
        verbose=False,
    )
    model.fit(freqs, spectrum, list(config.freq_range))

    if not model.results.has_model:
        raise RuntimeError("Specparam fitting was unsuccessful.")

    aperiodic = np.asarray(model.results.get_params("aperiodic"))
    periodic = np.asarray(model.results.get_params("periodic"))
    error = float(np.asarray(model.results.get_metrics("error")).squeeze())
    r_squared = float(
        np.asarray(model.results.get_metrics("gof", "rsquared")).squeeze()
    )

    if periodic.size == 0 or np.all(np.isnan(periodic)):
        peak_count = 0.0
        dominant_freq = np.nan
        dominant_power = np.nan
    else:
        periodic = np.atleast_2d(periodic)
        peak_count = float(periodic.shape[0])
        dominant_idx = int(np.nanargmax(periodic[:, 1]))
        dominant_freq = float(periodic[dominant_idx, 0])
        dominant_power = float(periodic[dominant_idx, 1])

    offset = float(aperiodic[0]) if aperiodic.size >= 1 else np.nan
    knee = float(aperiodic[1]) if aperiodic.size == 3 else np.nan
    exponent = float(aperiodic[-1]) if aperiodic.size >= 2 else np.nan

    periodic_psd = None
    if need_periodic_psd:
        full_log = np.asarray(model.results.model.get_component("full"), dtype=float)
        aperiodic_log = np.asarray(
            model.results.model.get_component("aperiodic"),
            dtype=float,
        )
        if not np.all(np.isfinite(full_log)) or not np.all(np.isfinite(aperiodic_log)):
            raise RuntimeError(
                "Specparam model components became non-finite for corrected bands."
            )
        periodic_psd = np.clip(
            np.power(10.0, full_log) - np.power(10.0, aperiodic_log),
            0.0,
            None,
        )

    return {
        "offset": offset,
        "knee": knee,
        "exponent": exponent,
        "fit_error": error,
        "r_squared": r_squared,
        "peak_count": peak_count,
        "peak_freq_dom": dominant_freq,
        "peak_power_dom": dominant_power,
    }, periodic_psd


def fit_parametric_batch(
    psds: np.ndarray,
    freqs: np.ndarray,
    config: ParametricDescriptorConfig,
    runtime,
    need_periodic_psd: bool = False,
    include_metrics: bool = True,
) -> _ParametricFitBatch:
    """
    Fit parametric models over one PSD batch.

    Parameters
    ----------
    psds : np.ndarray
        PSD batch with shape ``(n_obs, n_channels, n_freqs)``.
    freqs : np.ndarray
        Frequency grid aligned with the last axis of ``psds``.
    config : ParametricDescriptorConfig
        Parsed parametric fit configuration.
    runtime : DescriptorRuntimeConfig
        Runtime execution controls. Only the inner fitting parallelism path
        uses this object here.
    need_periodic_psd : bool, default=False
        Whether to reconstruct periodic-only PSDs for each fitted spectrum.
    include_metrics : bool, default=True
        Whether to materialize scalar metric arrays in the returned payload.

    Returns
    -------
    _ParametricFitBatch
        Batch-scoped fit payload aligned to the input observation and channel
        axes after restricting the PSD to ``config.freq_range``.
    """
    freq_mask = (freqs >= config.freq_range[0]) & (freqs <= config.freq_range[1])
    local_freqs = freqs[freq_mask]
    local_psds = psds[..., freq_mask]

    metric_names: list[str] = []
    if include_metrics:
        if "aperiodic" in config.outputs:
            if config.aperiodic_mode == "knee":
                metric_names.extend(["offset", "knee", "exponent"])
            else:
                metric_names.extend(["offset", "exponent"])
        if "fit_quality" in config.outputs:
            metric_names.extend(["fit_error", "r_squared"])
        if "peak_summary" in config.outputs:
            metric_names.extend(["peak_count", "peak_freq_dom", "peak_power_dom"])
    metric_arrays = {
        metric_name: np.full(
            (local_psds.shape[0], local_psds.shape[1]),
            np.nan,
            dtype=float,
        )
        for metric_name in metric_names
    }
    periodic_psds = (
        np.full(local_psds.shape, np.nan, dtype=float) if need_periodic_psd else None
    )

    def fit_one(
        obs_rel: int,
        unit_idx: int,
    ) -> tuple[
        int,
        int,
        dict[str, float] | None,
        np.ndarray | None,
        dict[str, str] | None,
    ]:
        try:
            metrics, periodic = fit_single_spectrum(
                local_freqs,
                local_psds[obs_rel, unit_idx],
                config,
                need_periodic_psd=need_periodic_psd,
            )
            return obs_rel, unit_idx, metrics, periodic, None
        except Exception as exc:  # pragma: no cover - exercised via callers
            return (
                obs_rel,
                unit_idx,
                None,
                None,
                {
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                },
            )

    if runtime.execution_backend != "sequential" and runtime.n_jobs != 1:
        import joblib

        tasks = [
            (obs_rel, unit_idx)
            for obs_rel in range(local_psds.shape[0])
            for unit_idx in range(local_psds.shape[1])
        ]
        fit_results = joblib.Parallel(
            n_jobs=runtime.n_jobs,
            prefer="threads",
        )(joblib.delayed(fit_one)(obs_rel, unit_idx) for obs_rel, unit_idx in tasks)
    else:
        fit_results = [
            fit_one(obs_rel, unit_idx)
            for obs_rel in range(local_psds.shape[0])
            for unit_idx in range(local_psds.shape[1])
        ]

    errors: list[tuple[int, int, str, str]] = []
    for obs_rel, unit_idx, metrics, periodic, error in fit_results:
        if metrics is not None:
            for metric_name in metric_names:
                metric_arrays[metric_name][obs_rel, unit_idx] = metrics[metric_name]
            if periodic_psds is not None and periodic is not None:
                periodic_psds[obs_rel, unit_idx] = periodic
            continue
        errors.append(
            (
                obs_rel,
                unit_idx,
                error["exception_type"],
                error["message"],
            )
        )

    return _ParametricFitBatch(
        freqs=np.asarray(local_freqs, dtype=float),
        metrics=metric_arrays,
        errors=errors,
        periodic_psds=periodic_psds,
        meta={
            "backend": config.backend,
            "freq_range": list(config.freq_range),
            "aperiodic_mode": config.aperiodic_mode,
        },
    )

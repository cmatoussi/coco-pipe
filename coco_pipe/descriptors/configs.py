"""
Descriptor Configuration
========================

Strict Pydantic configuration models for the descriptors module.

This module defines the static, typed configuration surface for descriptor
extraction:

- explicit runtime input requirements
- family-specific configs for bands, parametric fitting, and complexity
- output formatting controls
- runtime execution controls

These models validate local field structure and family-local constraints. The
remaining cross-family compatibility rule for corrected spectral outputs is
enforced by :class:`coco_pipe.descriptors.core.DescriptorPipeline` after config
parsing, because it depends on how multiple family configs interact.

Author: Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "DescriptorInputConfig",
    "BandDescriptorConfig",
    "ParametricDescriptorConfig",
    "ComplexityDescriptorConfig",
    "DescriptorFamiliesConfig",
    "DescriptorOutputConfig",
    "DescriptorRuntimeConfig",
    "DescriptorConfig",
]


CANONICAL_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

_BAND_OUTPUTS = (
    "absolute_power",
    "relative_power",
    "ratios",
    "corrected_absolute_power",
    "corrected_relative_power",
    "corrected_ratios",
)
_PARAM_OUTPUTS = ("aperiodic", "fit_quality", "peak_summary")
_COMPLEXITY_MEASURES = (
    "sample_entropy",
    "perm_entropy",
    "spectral_entropy",
    "hjorth_mobility",
    "hjorth_complexity",
    "lziv_complexity",
)


class _StrictConfigModel(BaseModel):
    """Shared strict Pydantic behavior."""

    model_config = ConfigDict(extra="forbid")


class DescriptorInputConfig(_StrictConfigModel):
    """
    Explicit runtime input requirements for descriptor extraction.

    Parameters
    ----------
    require_sfreq : bool, default=True
        Whether extraction requires an explicit sampling frequency input.
    require_channel_names : bool, default=False
        Whether extraction requires explicit channel names at runtime.

    Notes
    -----
    The descriptors module accepts only explicit NumPy-like arrays with shape
    ``(n_obs, n_channels, n_times)`` in observation-channel-time order. That
    structural contract is fixed by the module and enforced at runtime; this
    config only controls which additional runtime inputs must also be passed.
    """

    require_sfreq: bool = True
    require_channel_names: bool = False


class BandDescriptorConfig(_StrictConfigModel):
    """
    Configuration for PSD-based band summary descriptors.

    Parameters
    ----------
    enabled : bool, default=False
        Whether the band family is enabled.
    psd_method : {"welch", "multitaper"}, default="welch"
        PSD estimator used before computing band summaries.
    fmin, fmax : float
        Global frequency window within which PSDs and bands are evaluated.
    bands : dict of str to tuple of float, default=canonical EEG bands
        Mapping from band name to ``(low, high)`` boundaries.
    outputs : list of {"absolute_power", "relative_power", "ratios", \
"corrected_absolute_power", "corrected_relative_power", "corrected_ratios"}
        Band descriptors to emit.
    ratio_pairs : list of tuple of str, default=[]
        Explicit numerator and denominator band names for ratio outputs.
    log_power : bool, default=False
        Whether to emit log-transformed absolute band power in addition to
        absolute power when that output is enabled.

    Notes
    -----
    Corrected band outputs are configured here, but their cross-family
    compatibility with the parametric fit range is checked later by the
    descriptor pipeline because that rule depends on both the band and
    parametric family configs together.
    """

    enabled: bool = False
    psd_method: Literal["welch", "multitaper"] = "welch"
    fmin: float = Field(1.0, ge=0.0)
    fmax: float = Field(45.0, gt=0.0)
    bands: dict[str, tuple[float, float]] = Field(
        default_factory=lambda: dict(CANONICAL_BANDS)
    )
    outputs: list[
        Literal[
            "absolute_power",
            "relative_power",
            "ratios",
            "corrected_absolute_power",
            "corrected_relative_power",
            "corrected_ratios",
        ]
    ] = Field(default_factory=lambda: ["absolute_power"])
    ratio_pairs: list[tuple[str, str]] = Field(default_factory=list)
    log_power: bool = False

    @field_validator("bands", mode="before")
    @classmethod
    def _coerce_bands(cls, value: Any) -> dict[str, tuple[float, float]]:
        if value is None:
            return dict(CANONICAL_BANDS)
        return {str(key): tuple(bounds) for key, bounds in dict(value).items()}

    @field_validator("outputs", mode="before")
    @classmethod
    def _validate_outputs(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("Band outputs must not contain duplicates.")
        invalid = sorted(set(value) - set(_BAND_OUTPUTS))
        if invalid:
            raise ValueError(f"Unknown band outputs: {invalid}")
        return value

    @field_validator("ratio_pairs", mode="before")
    @classmethod
    def _coerce_ratio_pairs(cls, value: Any) -> list[tuple[str, str]]:
        if value is None:
            return []
        return [tuple(pair) for pair in value]

    @model_validator(mode="after")
    def _validate_model(self) -> "BandDescriptorConfig":
        if self.fmin >= self.fmax:
            raise ValueError("Band descriptor config requires fmin < fmax.")
        for name, (low, high) in self.bands.items():
            if low >= high:
                raise ValueError(f"Band '{name}' requires low < high.")
            if low < self.fmin or high > self.fmax:
                raise ValueError(
                    "Band "
                    f"'{name}' must stay within the configured "
                    f"[{self.fmin}, {self.fmax}] range."
                )
        if (
            "ratios" in self.outputs or "corrected_ratios" in self.outputs
        ) and not self.ratio_pairs:
            raise ValueError("Band ratios require explicit ratio_pairs.")
        return self


class ParametricDescriptorConfig(_StrictConfigModel):
    """
    Configuration for specparam-based spectral summary descriptors.

    Parameters
    ----------
    enabled : bool, default=False
        Whether the parametric family is enabled.
    backend : {"specparam"}, default="specparam"
        Parametric modeling backend.
    psd_method : {"welch", "multitaper"}, default="welch"
        PSD estimator used before fitting the parametric model.
    freq_range : tuple of float, default=(1.0, 45.0)
        Frequency range passed to the parametric model.
    peak_width_limits : tuple of float, default=(1.0, 12.0)
        Peak width bounds forwarded to the model backend.
    max_n_peaks : int, default=6
        Maximum number of periodic peaks to fit.
    aperiodic_mode : {"fixed", "knee"}, default="fixed"
        Aperiodic model form used by specparam.
    outputs : list of {"aperiodic", "fit_quality", "peak_summary"}
        Parametric descriptor groups to emit.

    Notes
    -----
    This config describes how the shared parametric fit is produced. The same
    fit can be reused by the parametric family itself and by corrected spectral
    outputs when the planner detects compatible requests.
    """

    enabled: bool = False
    backend: Literal["specparam"] = "specparam"
    psd_method: Literal["welch", "multitaper"] = "welch"
    freq_range: tuple[float, float] = (1.0, 45.0)
    peak_width_limits: tuple[float, float] = (1.0, 12.0)
    max_n_peaks: int = Field(6, ge=0)
    aperiodic_mode: Literal["fixed", "knee"] = "fixed"
    outputs: list[Literal["aperiodic", "fit_quality", "peak_summary"]] = Field(
        default_factory=lambda: ["aperiodic", "fit_quality", "peak_summary"]
    )

    @field_validator("outputs", mode="before")
    @classmethod
    def _validate_outputs(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("Parametric outputs must not contain duplicates.")
        invalid = sorted(set(value) - set(_PARAM_OUTPUTS))
        if invalid:
            raise ValueError(f"Unknown parametric outputs: {invalid}")
        return value

    @model_validator(mode="after")
    def _validate_model(self) -> "ParametricDescriptorConfig":
        if self.freq_range[0] >= self.freq_range[1]:
            raise ValueError("Parametric freq_range requires low < high.")
        if self.peak_width_limits[0] >= self.peak_width_limits[1]:
            raise ValueError("peak_width_limits requires low < high.")
        return self


class ComplexityDescriptorConfig(_StrictConfigModel):
    """
    Configuration for signal-complexity descriptors.

    Parameters
    ----------
    enabled : bool, default=False
        Whether the complexity family is enabled.
    backend : {"antropy", "neurokit2", "auto"}, default="antropy"
        Complexity backend used for supported measures.
    measures : list of str
        Complexity measures to compute.
    measure_kwargs : dict of str to dict, default={}
        Per-measure keyword arguments forwarded to the backend implementation.

    Notes
    -----
    Complexity measures are signal-domain descriptors. Unlike the PSD-based
    families, they do not participate in shared PSD planning.
    """

    enabled: bool = False
    backend: Literal["antropy", "neurokit2", "auto"] = "antropy"
    measures: list[str] = Field(default_factory=lambda: list(_COMPLEXITY_MEASURES))
    measure_kwargs: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @field_validator("measures", mode="before")
    @classmethod
    def _validate_measures(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("Complexity measures must not contain duplicates.")
        invalid = sorted(set(value) - set(_COMPLEXITY_MEASURES))
        if invalid:
            raise ValueError(f"Unknown complexity measures: {invalid}")
        return value


class DescriptorFamiliesConfig(_StrictConfigModel):
    """
    Group descriptor-family configuration under one top-level field.

    Attributes
    ----------
    bands : BandDescriptorConfig
        Configuration for PSD-based band summaries.
    parametric : ParametricDescriptorConfig
        Configuration for specparam-based summaries.
    complexity : ComplexityDescriptorConfig
        Configuration for complexity measures.
    """

    bands: BandDescriptorConfig = Field(default_factory=BandDescriptorConfig)
    parametric: ParametricDescriptorConfig = Field(
        default_factory=ParametricDescriptorConfig
    )
    complexity: ComplexityDescriptorConfig = Field(
        default_factory=ComplexityDescriptorConfig
    )


class DescriptorOutputConfig(_StrictConfigModel):
    """
    Controls output precision and descriptor-level channel pooling.

    Parameters
    ----------
    precision : {"float32", "float64"}, default="float32"
        Output dtype used for the final descriptor matrix.
    channel_pooling : {"none", "all"} or dict of str to list of str, default="none"
        Descriptor-level channel pooling policy applied after per-channel
        descriptors are computed. ``"none"`` keeps one descriptor per sensor,
        ``"all"`` averages descriptor values across all sensors, and a mapping
        averages descriptor values within each named group while leaving
        ungrouped sensors unchanged.

    Notes
    -----
    Output config is intentionally small. The descriptors module now returns a
    minimal result object, so output controls are limited to matrix precision
    and descriptor-level channel pooling.
    """

    precision: Literal["float32", "float64"] = "float32"
    channel_pooling: Literal["none", "all"] | dict[str, list[str]] = "none"

    @field_validator("channel_pooling", mode="before")
    @classmethod
    def _coerce_channel_pooling(
        cls, value: Any
    ) -> Literal["none", "all"] | dict[str, list[str]]:
        if value in (None, {}):
            return "none"
        if isinstance(value, str):
            return value
        return {
            str(group_name): [str(member) for member in members]
            for group_name, members in dict(value).items()
        }

    @field_validator("channel_pooling")
    @classmethod
    def _validate_channel_pooling(
        cls, value: Literal["none", "all"] | dict[str, list[str]]
    ) -> Literal["none", "all"] | dict[str, list[str]]:
        if isinstance(value, str):
            if value not in {"none", "all"}:
                raise ValueError("channel_pooling must be 'none', 'all', or a mapping.")
            return value
        for group_name, members in value.items():
            if not group_name:
                raise ValueError(
                    "channel_pooling mapping keys must be non-empty strings."
                )
            if not members:
                raise ValueError(
                    f"channel_pooling['{group_name}'] must define at least one channel."
                )
            if len(set(members)) != len(members):
                raise ValueError(
                    f"channel_pooling['{group_name}'] must not contain duplicates."
                )
        return value


class DescriptorRuntimeConfig(_StrictConfigModel):
    """
    Runtime execution controls for descriptor extraction.

    Parameters
    ----------
    execution_backend : {"joblib", "sequential"}, default="joblib"
        Execution backend used by the pipeline.
    n_jobs : int, default=1
        Number of worker slots requested for supported parallel paths.
        ``-1`` means "use as much useful parallelism as the current stage can
        use", while positive integers request an explicit worker count.
    obs_chunk : int, default=128
        Number of observations processed per batch.
    on_error : {"raise", "warn", "collect"}, default="collect"
        Failure policy applied during extraction.

    Notes
    -----
    Runtime config controls execution only. It does not add provenance,
    reporting, or persistence metadata to the returned descriptor result.
    """

    execution_backend: Literal["joblib", "sequential"] = "joblib"
    n_jobs: int = 1
    obs_chunk: int = Field(128, gt=0)
    on_error: Literal["raise", "warn", "collect"] = Field(
        "collect",
        description=(
            "Policies: "
            "'raise' re-raises the first exception immediately; "
            "'warn' collects all failures and emits one aggregate warning; "
            "'collect' stores failures silently for inspection in result['failures']."
        ),
    )

    @field_validator("n_jobs")
    @classmethod
    def _validate_n_jobs(cls, value: int) -> int:
        if value == 0 or value < -1:
            raise ValueError("n_jobs must be -1 or a positive integer.")
        return value


class DescriptorConfig(_StrictConfigModel):
    """
    Top-level descriptors configuration object.

    Attributes
    ----------
    input : DescriptorInputConfig
        Runtime input requirements for explicit array extraction.
    families : DescriptorFamiliesConfig
        Enabled descriptor families and their typed configs.
    output : DescriptorOutputConfig
        Output precision and formatting settings.
    runtime : DescriptorRuntimeConfig
        Runtime execution and error-handling settings.

    Notes
    -----
    This object is the stable config boundary for
    :class:`coco_pipe.descriptors.core.DescriptorPipeline`. Parsing this config
    validates local structure here, then the pipeline applies the remaining
    cross-family compatibility checks when it builds the execution plan.
    """

    input: DescriptorInputConfig = Field(default_factory=DescriptorInputConfig)
    families: DescriptorFamiliesConfig = Field(default_factory=DescriptorFamiliesConfig)
    output: DescriptorOutputConfig = Field(default_factory=DescriptorOutputConfig)
    runtime: DescriptorRuntimeConfig = Field(default_factory=DescriptorRuntimeConfig)

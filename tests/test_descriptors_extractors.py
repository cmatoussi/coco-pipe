"""
Comprehensive Test Suite for Descriptor Extractors
==================================================

Unified tests for Spectral, Parametric, and Complexity descriptor extractors.
"""

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

from coco_pipe.descriptors.configs import (
    BandDescriptorConfig,
    ComplexityDescriptorConfig,
    ParametricDescriptorConfig,
)
from coco_pipe.descriptors.extractors._parametric_fit import _ParametricFitBatch
from coco_pipe.descriptors.extractors._psd import compute_psd
from coco_pipe.descriptors.extractors.base import _DescriptorBlock
from coco_pipe.descriptors.extractors.complexity import ComplexityDescriptorExtractor
from coco_pipe.descriptors.extractors.parametric import ParametricDescriptorExtractor
from coco_pipe.descriptors.extractors.spectral import BandDescriptorExtractor
from coco_pipe.descriptors.extractors.utils import (
    average_channel_matrix,
    make_failure_record,
    pool_channel_descriptor_matrix,
)

# --- Fixtures ---


@pytest.fixture
def signal_data():
    """Standard signal data: (n_obs, n_channels, n_times)."""
    rng = np.random.default_rng(42)
    sfreq = 250.0
    # Increase to 2 seconds for better Welch/entropy estimation
    t = np.arange(0, 2, 1 / sfreq)
    n_obs, n_chans = 5, 3

    # Create 1/f-like noise
    freqs = np.fft.rfftfreq(len(t), 1 / sfreq)
    weights = 1 / (freqs + 1.0)

    X = np.zeros((n_obs, n_chans, len(t)))
    for o in range(n_obs):
        for c in range(n_chans):
            white = rng.standard_normal(len(t))
            # Quick and dirty 1/f approximation
            X[o, c, :] = np.fft.irfft(np.fft.rfft(white) * weights, n=len(t))

    # Add strong oscillations to ensure fitting works
    X[:, 0, :] += 2.0 * np.sin(2 * np.pi * 10 * t)
    X[:, 1, :] += 1.5 * np.sin(2 * np.pi * 20 * t)

    return X, sfreq, ["Fz", "Cz", "Pz"]


@pytest.fixture
def psd_data(signal_data):
    """Standard PSD data: (n_obs, n_channels, n_freqs)."""
    X, sfreq, _ = signal_data
    psds, freqs = compute_psd(
        X, sfreq=sfreq, method="welch", fmin=1.0, fmax=45.0, n_jobs=1
    )
    return psds, freqs


@pytest.fixture
def mock_fit_batch(psd_data):
    """A mock ParametricFitBatch for testing corrected bands."""
    psds, freqs = psd_data
    n_obs, n_chans, n_freqs = psds.shape

    periodic_psds = np.zeros_like(psds)
    # Add a "peak" at 10Hz (approx index)
    f_idx = np.argmin(np.abs(freqs - 10.0))
    periodic_psds[:, :, f_idx] = 1.0

    metrics = {
        "offset": np.zeros((n_obs, n_chans)),
        "exponent": np.ones((n_obs, n_chans)) * 1.5,
    }

    return _ParametricFitBatch(
        freqs=freqs,
        metrics=metrics,
        errors=[],
        periodic_psds=periodic_psds,
    )


# --- 1. Base Interfaces and Utilities ---


def test_descriptor_block_structure():
    """Verify _DescriptorBlock simple data container."""
    X = np.zeros((5, 10))
    names = [f"desc_{i}" for i in range(10)]

    block = _DescriptorBlock(family="test", X=X, descriptor_names=names)
    assert block.X.shape == (5, 10)
    assert block.descriptor_names == names


def test_pool_channel_descriptor_matrix_logic():
    """Test all variants of channel pooling names and values."""
    # (n_obs, n_channels)
    values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    ch_names = ["Fz", "Cz"]

    # None
    pooled, scopes = pool_channel_descriptor_matrix(values, ch_names, "none")
    assert np.array_equal(pooled, values)
    assert scopes == ["ch-Fz", "ch-Cz"]

    # All (mean across channels)
    pooled, scopes = pool_channel_descriptor_matrix(values, ch_names, "all")
    assert pooled.shape == (3, 1)
    assert np.allclose(pooled[:, 0], [1.5, 3.5, 5.5])
    assert scopes == ["ch-all"]

    # Dict grouping
    values3 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    ch_names3 = ["Fz", "Cz", "Pz"]
    pooling = {"Frontal": ["Fz", "Cz"]}
    pooled, scopes = pool_channel_descriptor_matrix(values3, ch_names3, pooling)
    # Frontal (mean of 0,1) + Pz (remains)
    assert pooled.shape == (2, 2)
    assert np.allclose(pooled[:, 0], [1.5, 4.5])  # mean([1,2], [4,5])
    assert np.allclose(pooled[:, 1], [3.0, 6.0])  # Pz
    assert scopes == ["chgrp-Frontal", "ch-Pz"]


def test_average_channel_matrix_robustness():
    """Verify averaging handles NaNs correctly."""
    X = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, np.nan], [np.nan, np.nan]])
    res = average_channel_matrix(X)
    assert np.allclose(res[:3], [1.5, 4.0, 5.0])
    assert np.isnan(res[3])


def test_make_failure_record_schema():
    """Check fixed schema for failure records."""
    rec = make_failure_record(
        family="spectral",
        obs_index=5,
        exception_type="ValueError",
        message="test error",
        channel_index=2,
        channel_name="Cz",
    )
    assert rec["obs_index"] == 5
    assert rec["family"] == "spectral"
    assert rec["exception_type"] == "ValueError"
    assert rec["message"] == "test error"
    assert rec["channel_index"] == 2
    assert rec["channel_name"] == "Cz"


# --- 2. Spectral (Band) Extractor ---


class TestBandExtractor:
    def test_basic_extraction(self, psd_data, signal_data):
        psds, freqs = psd_data
        _, _, ch_names = signal_data
        config = BandDescriptorConfig(
            enabled=True,
            outputs=["absolute_power", "relative_power"],
            bands={"alpha": (8, 12), "beta": (15, 30)},
        )
        extractor = BandDescriptorExtractor(config)

        block = extractor.extract_psd(
            psds,
            freqs,
            channel_names=ch_names,
            channel_pooling="none",
            ids=None,
            runtime=MagicMock(),
        )
        assert block.family == "bands"
        # 2 bands * 2 outputs * 3 channels = 12 columns
        assert block.X.shape == (psds.shape[0], 12)
        assert "band_abs_alpha_ch-Fz" in block.descriptor_names
        assert "band_rel_beta_ch-Pz" in block.descriptor_names

    def test_corrected_outputs(self, psd_data, signal_data, mock_fit_batch):
        psds, freqs = psd_data
        _, _, ch_names = signal_data
        config = BandDescriptorConfig(
            enabled=True,
            outputs=["corrected_absolute_power", "corrected_ratios"],
            bands={"alpha": (8, 12), "beta": (13, 30)},
            ratio_pairs=[("alpha", "beta")],
        )
        extractor = BandDescriptorExtractor(config)

        block = extractor.extract_psd(
            psds,
            freqs,
            channel_names=ch_names,
            channel_pooling="all",
            ids=None,
            fit_batch=mock_fit_batch,
            runtime=MagicMock(),
        )
        # alpha, beta corrected + 1 ratio = 3 columns
        assert block.X.shape == (psds.shape[0], 3)
        assert "band_corr_abs_alpha_ch-all" in block.descriptor_names
        assert "band_corr_ratio_alpha_beta_ch-all" in block.descriptor_names

    def test_missing_fit_batch_raises(self, psd_data, signal_data):
        psds, freqs = psd_data
        _, _, ch_names = signal_data
        config = BandDescriptorConfig(
            enabled=True,
            outputs=["corrected_absolute_power"],
        )
        extractor = BandDescriptorExtractor(config)

        with pytest.raises(ValueError, match="require a supplied parametric fit_batch"):
            extractor.extract_psd(
                psds,
                freqs,
                channel_names=ch_names,
                channel_pooling="none",
                ids=None,
                runtime=MagicMock(),
            )

    def test_band_resolution_error(self, psd_data, signal_data):
        psds, freqs = psd_data
        _, _, ch_names = signal_data
        config = BandDescriptorConfig(
            enabled=True,
            fmax=250.0,
            bands={"ultra": (100, 200)},
            outputs=["absolute_power"],
        )
        extractor = BandDescriptorExtractor(config)

        runtime = MagicMock()
        runtime.on_error = "collect"

        block = extractor.extract_psd(
            psds,
            freqs,
            channel_names=ch_names,
            channel_pooling="all",
            ids=None,
            runtime=runtime,
        )
        assert len(block.failures) > 0
        assert block.failures[0]["exception_type"] == "BandResolutionError"

    def test_spectral_capabilities_and_requests(self):
        config = BandDescriptorConfig(enabled=True)
        extractor = BandDescriptorExtractor(config)
        assert extractor.capabilities["requires_sfreq"] is True

        # corrected without fit_config raises
        config_corr = BandDescriptorConfig(
            enabled=True, outputs=["corrected_absolute_power"]
        )
        extractor_corr = BandDescriptorExtractor(config_corr, fit_config=None)
        with pytest.raises(ValueError, match="Corrected band outputs require"):
            extractor_corr.psd_request()

    def test_spectral_extract_psd_edge_cases(self, signal_data):
        from unittest.mock import MagicMock

        X, sfreq, ch_names = signal_data

        # log_power coverage
        config = BandDescriptorConfig(
            enabled=True, outputs=["absolute_power"], log_power=True
        )
        extractor = BandDescriptorExtractor(config)
        psds = np.ones((1, 3, 10))
        freqs = np.linspace(1, 45, 10)
        block = extractor.extract_psd(psds, freqs, ch_names, "all", None, MagicMock())
        assert "band_log_abs_delta_ch-all" in block.descriptor_names

        config_rel = BandDescriptorConfig(
            enabled=True,
            outputs=["relative_power"],
            fmin=100.0,
            fmax=200.0,
            bands={"high": (120, 150)},
        )
        extractor_rel = BandDescriptorExtractor(config_rel)
        block_rel = extractor_rel.extract_psd(
            psds, freqs, ch_names, "all", None, MagicMock()
        )
        assert np.isnan(block_rel.X).all()

        # fit_batch errors and corrected relative power empty freq
        from coco_pipe.descriptors.extractors._parametric_fit import _ParametricFitBatch

        fit_batch = _ParametricFitBatch(
            freqs=np.array([1, 10, 20]),  # Must be non-empty and non-None
            metrics={},
            periodic_psds=np.zeros((1, 3, 3)),
            errors=[(0, 0, "FakeError", "Fit failed")],
            meta={},
        )
        config_corr = BandDescriptorConfig(
            enabled=True, outputs=["corrected_relative_power"]
        )
        extractor_corr = BandDescriptorExtractor(config_corr)
        block_corr = extractor_corr.extract_psd(
            psds, freqs, ch_names, "all", None, MagicMock(), fit_batch=fit_batch
        )
        assert len(block_corr.failures) > 0
        assert "FakeError" in block_corr.failures[0]["exception_type"]

    def test_spectral_standalone_extract(self, signal_data):
        from unittest.mock import MagicMock

        X, sfreq, ch_names = signal_data
        config = BandDescriptorConfig(enabled=True, outputs=["absolute_power"])
        extractor = BandDescriptorExtractor(config)
        block = extractor.extract(X, sfreq, ch_names, "all", None, MagicMock())
        # 5 bands (delta, theta, alpha, beta, gamma) pooled to 'all' -> 5 columns
        assert block.X.shape == (5, 5)

    def test_spectral_empty_output_block(self):
        # Empty output when no families enabled or no outputs requested
        config = BandDescriptorConfig(enabled=True, outputs=[])
        extractor = BandDescriptorExtractor(config)
        psds = np.ones((2, 2, 10))
        freqs = np.linspace(1, 45, 10)
        block = extractor.extract_psd(
            psds, freqs, ["ch1", "ch2"], "all", None, MagicMock()
        )
        assert block.X.shape == (2, 0)

    def test_spectral_standalone_extract_raises_for_corrected(self, signal_data):
        from unittest.mock import MagicMock

        X, sfreq, ch_names = signal_data
        config = BandDescriptorConfig(
            enabled=True, outputs=["corrected_absolute_power"]
        )
        extractor = BandDescriptorExtractor(config)
        with pytest.raises(
            ValueError, match="Corrected band outputs are only available"
        ):
            extractor.extract(X, sfreq, ch_names, "all", None, MagicMock())


# --- 3. Parametric Extractor ---


class TestParametricExtractor:
    def test_standalone_extract(self, signal_data):
        """Verify real specparam-based extraction."""
        X, sfreq, ch_names = signal_data
        config = ParametricDescriptorConfig(
            enabled=True,
            outputs=["aperiodic"],
            psd_method="welch",
            freq_range=(1.0, 45.0),
        )
        extractor = ParametricDescriptorExtractor(config)

        block = extractor.extract(
            X,
            sfreq=sfreq,
            channel_names=ch_names,
            channel_pooling="all",
            ids=None,
            runtime=MagicMock(),
            obs_offset=0,
        )
        assert "param_exponent_ch-all" in block.descriptor_names
        # 2 features (offset, exponent) for aperiodic 'fixed' mode
        assert block.X.shape == (X.shape[0], 2)
        # Check that exponent is reasonable (> 0) and offset is finite
        assert np.all(block.X[:, 1] > 0)
        assert np.all(np.isfinite(block.X[:, 0]))

    def test_extract_psd_requires_fit_batch(self, psd_data, signal_data):
        psds, freqs = psd_data
        _, _, ch_names = signal_data
        extractor = ParametricDescriptorExtractor(ParametricDescriptorConfig())

        with pytest.raises(ValueError, match="requires a supplied fit_batch"):
            extractor.extract_psd(
                psds,
                freqs,
                channel_names=ch_names,
                channel_pooling="all",
                ids=None,
                runtime=MagicMock(),
                obs_offset=0,
            )


# --- 4. Complexity Extractor ---


class TestComplexityExtractor:
    def test_backend_dispatch_antropy(self, signal_data):
        X, sfreq, ch_names = signal_data
        config = ComplexityDescriptorConfig(
            enabled=True, backend="antropy", measures=["spectral_entropy"]
        )
        extractor = ComplexityDescriptorExtractor(config)

        block = extractor.extract(
            X,
            sfreq=sfreq,
            channel_names=ch_names,
            channel_pooling="all",
            ids=None,
            runtime=MagicMock(),
            obs_offset=0,
        )
        assert "complexity_spectral_entropy_ch-all" in block.descriptor_names
        assert not np.isnan(block.X).any()

    def test_backend_dispatch_neurokit2(self, signal_data):
        X, sfreq, ch_names = signal_data
        config = ComplexityDescriptorConfig(
            enabled=True, backend="neurokit2", measures=["perm_entropy"]
        )
        extractor = ComplexityDescriptorExtractor(config)

        block = extractor.extract(
            X,
            sfreq=sfreq,
            channel_names=ch_names,
            channel_pooling="all",
            ids=None,
            runtime=MagicMock(),
            obs_offset=0,
        )
        assert "complexity_perm_entropy_ch-all" in block.descriptor_names
        # Check that it's finite
        assert not np.isnan(block.X).any()

    def test_mixed_execution_strategy(self, signal_data):
        """Verify execution paths for combined complexity measures."""
        X, sfreq, ch_names = signal_data
        config = ComplexityDescriptorConfig(
            enabled=True,
            backend="antropy",
            measures=["spectral_entropy", "sample_entropy"],
        )
        extractor = ComplexityDescriptorExtractor(config)

        block = extractor.extract(
            X,
            sfreq=sfreq,
            channel_names=ch_names,
            channel_pooling="none",
            ids=None,
            runtime=MagicMock(),
            obs_offset=0,
        )

        # 2 measures * 3 channels = 6 columns
        assert block.X.shape == (X.shape[0], 6)
        assert "complexity_spectral_entropy_ch-Fz" in block.descriptor_names
        assert "complexity_sample_entropy_ch-Pz" in block.descriptor_names


# --- 5. Lazy Loading and Dependency Guards ---


def test_lazy_loading_failure_antropy(monkeypatch):
    """Verify informative error when antropy is missing."""
    monkeypatch.setitem(sys.modules, "antropy", None)
    config = ComplexityDescriptorConfig(enabled=True, backend="antropy")
    extractor = ComplexityDescriptorExtractor(config)

    with pytest.raises(ImportError, match="antropy"):
        extractor._load_antropy()


def test_lazy_loading_failure_neurokit2(monkeypatch):
    monkeypatch.setitem(sys.modules, "neurokit2", None)
    config = ComplexityDescriptorConfig(enabled=True, backend="neurokit2")
    extractor = ComplexityDescriptorExtractor(config)

    with pytest.raises(ImportError, match="neurokit"):
        extractor._load_neurokit()

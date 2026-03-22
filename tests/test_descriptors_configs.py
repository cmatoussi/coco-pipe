import pytest
from pydantic import ValidationError

from coco_pipe.descriptors import DescriptorPipeline
from coco_pipe.descriptors.configs import DescriptorConfig


def test_descriptor_config_is_strict():
    with pytest.raises(ValidationError):
        DescriptorConfig(unknown_field=1)


def test_band_ratios_require_pairs():
    with pytest.raises(ValidationError, match="ratio_pairs"):
        DescriptorConfig(
            families={
                "bands": {
                    "enabled": True,
                    "outputs": ["absolute_power", "ratios"],
                }
            }
        )


def test_corrected_band_ratios_require_pairs():
    with pytest.raises(ValidationError, match="ratio_pairs"):
        DescriptorConfig(
            families={
                "bands": {
                    "enabled": True,
                    "outputs": ["corrected_absolute_power", "corrected_ratios"],
                }
            }
        )


def test_corrected_bands_require_parametric_fit_range_to_cover_band_window():
    with pytest.raises(ValueError, match="Corrected band outputs require"):
        DescriptorPipeline(
            {
                "families": {
                    "bands": {
                        "enabled": True,
                        "fmin": 1.0,
                        "fmax": 45.0,
                        "outputs": ["corrected_absolute_power"],
                    },
                    "parametric": {
                        "freq_range": [4.0, 30.0],
                    },
                }
            }
        )


def test_channel_pooling_accepts_none_all_or_mapping():
    assert (
        DescriptorConfig(output={"channel_pooling": "none"}).output.channel_pooling
        == "none"
    )
    assert (
        DescriptorConfig(output={"channel_pooling": "all"}).output.channel_pooling
        == "all"
    )
    assert DescriptorConfig(
        output={"channel_pooling": {"Frontal": ["Fz", "Cz"]}}
    ).output.channel_pooling == {"Frontal": ["Fz", "Cz"]}


def test_runtime_and_output_flags_parse_strictly():
    config = DescriptorConfig.model_validate(
        {
            "input": {
                "require_sfreq": False,
                "require_channel_names": True,
            },
            "output": {
                "precision": "float64",
                "channel_pooling": "all",
            },
            "runtime": {
                "execution_backend": "joblib",
                "n_jobs": -1,
                "obs_chunk": 16,
                "on_error": "collect",
            },
        }
    )

    assert config.input.require_sfreq is False
    assert config.input.require_channel_names is True
    assert config.output.precision == "float64"
    assert config.output.channel_pooling == "all"
    assert config.runtime.execution_backend == "joblib"
    assert config.runtime.n_jobs == -1
    assert config.runtime.obs_chunk == 16


def test_runtime_n_jobs_must_be_minus_one_or_positive():
    with pytest.raises(ValidationError, match="n_jobs"):
        DescriptorConfig.model_validate({"runtime": {"n_jobs": 0}})

    with pytest.raises(ValidationError, match="n_jobs"):
        DescriptorConfig.model_validate({"runtime": {"n_jobs": -2}})


def test_removed_ceremonial_fields_are_rejected():
    with pytest.raises(ValidationError):
        DescriptorConfig.model_validate(
            {
                "input": {"expected_ndim": 3},
            }
        )

    with pytest.raises(ValidationError):
        DescriptorConfig.model_validate(
            {
                "runtime": {"batch_size": 32},
            }
        )

    with pytest.raises(ValidationError):
        DescriptorConfig.model_validate(
            {
                "output": {"return_format": "dict"},
            }
        )

    with pytest.raises(ValidationError):
        DescriptorConfig.model_validate(
            {
                "output": {"include_failure_summary": False},
            }
        )

    with pytest.raises(ValidationError):
        DescriptorConfig.model_validate(
            {
                "output": {"include_runtime_meta": False},
            }
        )

    with pytest.raises(ValidationError):
        DescriptorConfig.model_validate(
            {
                "output": {"channel_resolved": False},
            }
        )

    with pytest.raises(ValidationError):
        DescriptorConfig.model_validate(
            {
                "output": {"channel_groups": {"Frontal": ["Fz", "Cz"]}},
            }
        )

    with pytest.raises(ValidationError):
        DescriptorConfig.model_validate(
            {
                "families": {"bands": {"reduce_channels": "mean"}},
            }
        )

    with pytest.raises(ValidationError):
        DescriptorConfig.model_validate(
            {
                "families": {"bands": {"per_channel": True}},
            }
        )

    with pytest.raises(ValidationError):
        DescriptorConfig.model_validate(
            {
                "families": {"parametric": {"store_failures": True}},
            }
        )

    with pytest.raises(ValidationError):
        DescriptorConfig.model_validate(
            {
                "runtime": {"chunking": {"obs_chunk": 16}},
            }
        )

    with pytest.raises(ValidationError):
        DescriptorConfig.model_validate(
            {
                "runtime": {"channel_chunk": 4},
            }
        )

    with pytest.raises(ValidationError):
        DescriptorConfig.model_validate(
            {
                "runtime": {"time_chunk": 128},
            }
        )

    with pytest.raises(ValidationError):
        DescriptorConfig.model_validate(
            {
                "runtime": {"random_state": 7},
            }
        )


def test_band_validation_edge_cases():
    # Duplicate outputs
    with pytest.raises(ValidationError, match="duplicates"):
        DescriptorConfig(
            families={"bands": {"outputs": ["absolute_power", "absolute_power"]}}
        )

    # Invalid fmin/fmax
    with pytest.raises(ValidationError, match="fmin < fmax"):
        DescriptorConfig(families={"bands": {"fmin": 10, "fmax": 5}})

    # Band low >= high
    with pytest.raises(ValidationError, match="low < high"):
        DescriptorConfig(families={"bands": {"bands": {"bad": (10, 5)}}})

    # Band out of range
    with pytest.raises(ValidationError, match="stay within"):
        DescriptorConfig(
            families={"bands": {"fmin": 1, "fmax": 10, "bands": {"out": (5, 15)}}}
        )

    # Unknown outputs
    with pytest.raises(ValidationError, match="Unknown band outputs"):
        DescriptorConfig(families={"bands": {"outputs": ["non_existent"]}})


def test_parametric_validation_edge_cases():
    # Duplicate outputs
    with pytest.raises(ValidationError, match="duplicates"):
        DescriptorConfig(
            families={"parametric": {"outputs": ["aperiodic", "aperiodic"]}}
        )

    # Invalid freq_range
    with pytest.raises(ValidationError, match="low < high"):
        DescriptorConfig(families={"parametric": {"freq_range": (10, 5)}})

    # Invalid peak_width_limits
    with pytest.raises(ValidationError, match="low < high"):
        DescriptorConfig(families={"parametric": {"peak_width_limits": (5, 2)}})

    # Unknown outputs
    with pytest.raises(ValidationError, match="Unknown parametric outputs"):
        DescriptorConfig(families={"parametric": {"outputs": ["non_existent"]}})


def test_complexity_validation_edge_cases():
    # Duplicate measures
    with pytest.raises(ValidationError, match="duplicates"):
        DescriptorConfig(
            families={"complexity": {"measures": ["sample_entropy", "sample_entropy"]}}
        )

    # Unknown measures
    with pytest.raises(ValidationError, match="Unknown complexity measures"):
        DescriptorConfig(families={"complexity": {"measures": ["non_existent"]}})


def test_channel_pooling_validation_edge_cases():
    # Invalid string
    with pytest.raises(
        ValidationError,
        match="Input should be 'none' or 'all'|Input should be a valid dictionary",
    ):
        DescriptorConfig(output={"channel_pooling": "some_string"})

    # Empty group name
    with pytest.raises(ValidationError, match="non-empty strings"):
        DescriptorConfig(output={"channel_pooling": {"": ["ch1"]}})

    # Empty members
    with pytest.raises(ValidationError, match="at least one channel"):
        DescriptorConfig(output={"channel_pooling": {"G1": []}})

    # Duplicate members
    with pytest.raises(ValidationError, match="not contain duplicates"):
        DescriptorConfig(output={"channel_pooling": {"G1": ["ch1", "ch1"]}})

    # Coerce None/{} to none
    assert (
        DescriptorConfig(output={"channel_pooling": None}).output.channel_pooling
        == "none"
    )
    assert (
        DescriptorConfig(output={"channel_pooling": {}}).output.channel_pooling
        == "none"
    )


def test_coercion_logic_smoke():
    # Coerce bands
    config = DescriptorConfig(families={"bands": {"bands": {"delta": [1, 4]}}})
    assert config.families.bands.bands["delta"] == (1.0, 4.0)

    # Coerce ratio_pairs
    config = DescriptorConfig(
        families={"bands": {"outputs": ["ratios"], "ratio_pairs": [["theta", "beta"]]}}
    )
    assert config.families.bands.ratio_pairs == [("theta", "beta")]

    # Coerce None bands
    from coco_pipe.descriptors.configs import BandDescriptorConfig

    assert BandDescriptorConfig(bands=None).bands["alpha"] == (8.0, 13.0)

    # Coerce None ratio_pairs
    assert BandDescriptorConfig(ratio_pairs=None).ratio_pairs == []

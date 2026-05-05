#!/usr/bin/env python3
"""
Run the descriptors pipeline from a YAML configuration.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import yaml

from coco_pipe.descriptors import DescriptorConfig, DescriptorPipeline
from coco_pipe.io import load_data

_ALLOWED_DATA_TYPES = {"auto", "tabular", "bids", "embedding"}


def _normalize_data_config(data_cfg):
    normalized = dict(data_cfg)
    if "tasks" in normalized:
        if "task" in normalized:
            raise ValueError("Use only one of `task` or `tasks` in the data config.")
        tasks = normalized.pop("tasks")
        if isinstance(tasks, str):
            normalized["task"] = tasks
        else:
            tasks = list(tasks)
            if len(tasks) != 1:
                raise ValueError(
                    "run_descriptors.py supports exactly one task in `data.tasks`."
                )
            normalized["task"] = tasks[0]
    return normalized


def _validate_data_config(data_cfg):
    if not isinstance(data_cfg, Mapping):
        raise ValueError("The YAML `data` section must be a mapping.")

    normalized = _normalize_data_config(data_cfg)
    path = normalized.get("path")
    if not path:
        raise ValueError("The YAML `data` section must define a non-empty `path`.")

    data_type = normalized.get("type", "auto")
    if data_type not in _ALLOWED_DATA_TYPES:
        raise ValueError(
            "`data.type` must be one of "
            f"{sorted(_ALLOWED_DATA_TYPES)}; got {data_type!r}."
        )

    sfreq_override = normalized.get("sfreq_override")
    if sfreq_override is not None and float(sfreq_override) <= 0:
        raise ValueError("`data.sfreq_override` must be positive when provided.")

    return normalized


def _extract_explicit_inputs(container):
    sfreq = getattr(container, "meta", {}).get("sfreq")
    channel_names = None
    coords = getattr(container, "coords", {}) or {}
    if "channel" in coords:
        channel_names = list(np.asarray(coords["channel"]).tolist())

    return {
        "X": np.asarray(container.X),
        "ids": getattr(container, "ids", None),
        "sfreq": sfreq,
        "channel_names": channel_names,
    }


def _save_result(save_path: Path, result):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        save_path,
        X=result["X"],
        descriptor_names=np.asarray(result["descriptor_names"], dtype=object),
        failures=np.asarray(result["failures"], dtype=object),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run descriptor extraction.")
    parser.add_argument("--config", required=True, help="Path to YAML configuration.")
    args = parser.parse_args()

    config_path = Path(args.config)
    payload = yaml.safe_load(config_path.read_text())
    if not isinstance(payload, Mapping):
        raise ValueError("The YAML config must define a top-level mapping.")
    if "data" not in payload or "descriptors" not in payload:
        raise ValueError("The YAML config must contain `data` and `descriptors`.")

    data_cfg = _validate_data_config(payload["data"])
    data_path = data_cfg.pop("path")
    mode = data_cfg.pop("type", "auto")
    sfreq_override = data_cfg.pop("sfreq_override", None)
    save_path = payload.get("save_path")

    container = load_data(data_path, mode=mode, **data_cfg)
    explicit_inputs = _extract_explicit_inputs(container)
    if explicit_inputs["sfreq"] is None and sfreq_override is not None:
        explicit_inputs["sfreq"] = float(sfreq_override)
    if explicit_inputs["sfreq"] is None:
        raise ValueError(
            "Could not find `sfreq` in the loaded data container's metadata. "
            "Ensure the data is loaded with sampling frequency metadata, or add "
            "`sfreq_override` to the YAML configuration."
        )

    descriptor_cfg = DescriptorConfig.model_validate(payload["descriptors"])
    pipe = DescriptorPipeline(descriptor_cfg)
    result = pipe.extract(**explicit_inputs)

    channel_groups = payload.get("channel_groups")
    if channel_groups == "all":
        channel_groups = {"all": explicit_inputs["channel_names"]}

    if channel_groups:
        result = pipe.pool_channels(result, channel_groups)

    if save_path:
        _save_result(Path(save_path), result)

    print(
        {
            "shape": result["X"].shape,
            "n_descriptors": len(result["descriptor_names"]),
            "n_failures": len(result["failures"]),
        }
    )


if __name__ == "__main__":
    main()

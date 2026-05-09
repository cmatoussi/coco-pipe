"""
Cache-key helpers for decoding feature extraction.

The decoding module does not own a persistent embedding cache yet. This helper
defines the key contract future cache users should follow so fitted train-fold
transforms cannot be reused for incompatible test-fold samples.
"""

import hashlib
import json
from typing import Any, Sequence


def make_feature_cache_key(
    train_sample_ids: Sequence[Any],
    test_sample_ids: Sequence[Any],
    preprocessing_fingerprint: str,
    backbone_fingerprint: str,
) -> str:
    """
    Build a stable cache key for split-specific feature extraction artifacts.

    Parameters
    ----------
    train_sample_ids, test_sample_ids
        Sample IDs defining the split identity.
    preprocessing_fingerprint
        Fingerprint of fitted preprocessing/transform configuration.
    backbone_fingerprint
        Fingerprint of the feature extractor/backbone.
    """
    payload = {
        "train_sample_ids": [str(value) for value in train_sample_ids],
        "test_sample_ids": [str(value) for value in test_sample_ids],
        "preprocessing_fingerprint": preprocessing_fingerprint,
        "backbone_fingerprint": backbone_fingerprint,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()

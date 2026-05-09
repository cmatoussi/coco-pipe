"""Split-safe embedding cache helpers."""

from __future__ import annotations

from typing import Any, Sequence

from .cache import make_feature_cache_key


def make_embedding_cache_key(
    train_sample_ids: Sequence[Any],
    test_sample_ids: Sequence[Any],
    preprocessing_fingerprint: str,
    backbone_fingerprint: str,
) -> str:
    """Return the canonical split-safe embedding cache key."""
    return make_feature_cache_key(
        train_sample_ids=train_sample_ids,
        test_sample_ids=test_sample_ids,
        preprocessing_fingerprint=preprocessing_fingerprint,
        backbone_fingerprint=backbone_fingerprint,
    )

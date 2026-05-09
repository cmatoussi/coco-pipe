"""
Decoding Splitters
==================

Cross-validation splitters for the decoding module.
"""

from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    BaseCrossValidator,
    GroupKFold,
    KFold,
    LeaveOneGroupOut,
    LeavePGroupsOut,
    StratifiedGroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
)

from .configs import CVConfig


class _CVWithGroups(BaseCrossValidator):
    """
    Bind fixed groups to a cross-validator.

    This wrapper only ensures that the same group array is supplied whenever
    ``split`` or ``get_n_splits`` is called. It does not make a non-group
    splitter group-safe; group boundaries are enforced only by group-aware
    sklearn splitters such as ``GroupKFold``.
    """

    def __init__(self, cv, groups):
        self.cv = cv
        self.groups = groups

    def split(self, X, y=None, groups=None):
        return self.cv.split(X, y, self.groups)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.cv.get_n_splits(X, y, self.groups)

    def _get_tags(self):
        tags = getattr(self.cv, "_get_tags", lambda: {})()
        return {**tags, "non_deterministic": tags.get("non_deterministic", False)}

    def get_params(self, deep=True):
        return {"cv": self.cv, "groups": self.groups}

    def __repr__(self):
        return f"_CVWithGroups(cv={self.cv!r})"


class SimpleSplit(BaseCrossValidator):
    """One train/test split using ``train_test_split``."""

    def __init__(
        self,
        test_size: float = 0.2,
        shuffle: bool = True,
        random_state: Optional[int] = None,
        stratify: Optional[Union[bool, pd.Series, np.ndarray]] = None,
    ):
        if not (0 < test_size < 1):
            raise ValueError("test_size must be between 0 and 1.")
        self.test_size = test_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.stratify = stratify

    def split(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        groups: Optional[Sequence] = None,
    ):
        idx = np.arange(len(X))
        if self.stratify is True:
            strat = y
        elif self.stratify is False:
            strat = None
        else:
            strat = self.stratify

        train_idx, test_idx = train_test_split(
            idx,
            test_size=self.test_size,
            shuffle=self.shuffle,
            random_state=self.random_state if self.shuffle else None,
            stratify=strat,
        )
        yield train_idx, test_idx

    def get_n_splits(
        self,
        X: Any = None,
        y: Any = None,
        groups: Any = None,
    ) -> int:
        return 1

    def _get_tags(self):
        return {"non_deterministic": self.shuffle}


def get_cv_splitter(
    config: CVConfig,
    groups: Optional[Sequence] = None,
    y: Optional[Sequence] = None,
    require_groups: bool = True,
) -> BaseCrossValidator:
    """Create a scikit-learn cross-validator from ``CVConfig``."""
    strat = config.strategy.lower()
    group_strategies = {
        "group_kfold",
        "stratified_group_kfold",
        "leave_p_out",
        "leave_one_group_out",
    }

    if strat in group_strategies and require_groups and groups is None:
        raise ValueError(f"CV strategy '{config.strategy}' requires groups.")

    common_kwargs = {}
    if strat not in ["leave_one_group_out", "leave_p_out", "split", "timeseries"]:
        common_kwargs["n_splits"] = config.n_splits

    if strat in ["stratified", "kfold", "stratified_group_kfold", "split"]:
        common_kwargs["shuffle"] = config.shuffle
        common_kwargs["random_state"] = config.random_state if config.shuffle else None

    if strat == "stratified":
        splitter = StratifiedKFold(**common_kwargs)
    elif strat == "kfold":
        splitter = KFold(**common_kwargs)
    elif strat == "group_kfold":
        splitter = GroupKFold(n_splits=config.n_splits)
    elif strat == "stratified_group_kfold":
        splitter = StratifiedGroupKFold(**common_kwargs)
    elif strat == "leave_p_out":
        splitter = LeavePGroupsOut(n_groups=config.n_splits)
    elif strat == "leave_one_group_out":
        splitter = LeaveOneGroupOut()
    elif strat == "timeseries":
        splitter = TimeSeriesSplit(n_splits=config.n_splits)
    elif strat == "split":
        splitter = SimpleSplit(
            test_size=config.test_size,
            shuffle=config.shuffle,
            random_state=config.random_state,
            stratify=y if config.stratify and y is not None else config.stratify,
        )
    else:
        raise ValueError(f"Unknown CV strategy: {config.strategy}")

    if groups is not None:
        splitter = _CVWithGroups(splitter, groups)

    return splitter

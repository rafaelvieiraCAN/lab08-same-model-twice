"""Tests for data preparation."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from prepare import (
    BikeDataset,
    make_splits,
)


def test_dataset_length():
    dataset = BikeDataset(
        np.zeros((5, 2)),
        np.zeros(5),
    )

    assert len(dataset) == 5


def test_dataset_returns_float32():
    dataset = BikeDataset(
        np.zeros((3, 2)),
        np.zeros(3),
    )

    x, y = dataset[0]

    assert x.dtype == torch.float32
    assert y.dtype == torch.float32


def test_split_sizes():
    splits = make_splits()

    assert len(splits.train) == 3600
    assert len(splits.val) == 1200
    assert len(splits.test) == 1200


def test_total_rows():
    splits = make_splits()

    total = len(splits.train) + len(splits.val) + len(splits.test)

    assert total == 6000


def test_feature_count():
    splits = make_splits()

    assert splits.n_features == 8

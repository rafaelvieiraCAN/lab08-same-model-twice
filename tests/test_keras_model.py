"""Tests for Keras BikeNet."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from keras_model import (
    build_keras_model,
    count_keras_parameters,
)


def test_keras_output_shape():
    model = build_keras_model(
        n_features=8,
        hidden=32,
    )

    assert model.output_shape == (
        None,
        1,
    )


def test_parameter_count():
    model = build_keras_model(
        n_features=8,
        hidden=32,
    )

    assert count_keras_parameters(model) == 1377

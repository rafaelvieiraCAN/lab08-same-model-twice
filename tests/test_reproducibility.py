"""Reproducibility test."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from tensorflow import keras

from keras_model import build_keras_model


def test_same_seed_same_weights():
    keras.utils.set_random_seed(42)

    model_a = build_keras_model(
        n_features=8,
        hidden=32,
    )

    x = np.random.rand(
        10,
        8,
    ).astype("float32")

    model_a(x)

    weights_a = model_a.get_weights()

    keras.utils.set_random_seed(42)

    model_b = build_keras_model(
        n_features=8,
        hidden=32,
    )

    model_b(x)

    weights_b = model_b.get_weights()

    for wa, wb in zip(
        weights_a,
        weights_b,
    ):
        np.testing.assert_allclose(
            wa,
            wb,
        )

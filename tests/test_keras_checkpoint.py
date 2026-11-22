"""Tests for Keras checkpoint and reload."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from tensorflow import keras

from keras_model import build_keras_model


def test_save_and_load_model():
    model = build_keras_model(
        n_features=8,
        hidden=32,
    )

    x = np.random.rand(
        10,
        8,
    ).astype("float32")

    model(x)

    Path("reports").mkdir(
        exist_ok=True,
    )

    model.save("reports/test_model.keras")

    loaded = keras.models.load_model("reports/test_model.keras")

    pred1 = model.predict(
        x,
        verbose=0,
    )

    pred2 = loaded.predict(
        x,
        verbose=0,
    )

    np.testing.assert_allclose(
        pred1,
        pred2,
        rtol=1e-6,
        atol=1e-6,
    )

"""Equivalence test between PyTorch and Keras."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from keras_model import (
    build_keras_model,
    copy_weights_from_pytorch,
    predict_keras,
)
from model import BikeNet


def test_same_predictions_after_copy():
    torch.manual_seed(42)
    np.random.seed(42)

    pytorch_model = BikeNet(
        n_features=8,
        hidden=32,
    )

    keras_model = build_keras_model(
        n_features=8,
        hidden=32,
    )

    dummy_input = np.random.rand(
        5,
        8,
    ).astype(np.float32)

    keras_model(dummy_input)

    copy_weights_from_pytorch(
        pytorch_model,
        keras_model,
    )

    torch_output = pytorch_model(torch.tensor(dummy_input)).detach().numpy()

    keras_output = predict_keras(
        keras_model,
        dummy_input,
    )

    np.testing.assert_allclose(
        torch_output,
        keras_output,
        rtol=1e-5,
        atol=1e-5,
    )

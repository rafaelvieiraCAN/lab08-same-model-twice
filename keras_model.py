"""Keras implementation of BikeNet and weight transfer utilities."""

import numpy as np
from tensorflow import keras


def build_keras_model(
    n_features: int,
    hidden: int = 32,
):
    """Build the Keras version of BikeNet."""

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(n_features,)),
            keras.layers.Dense(
                hidden,
                activation="relu",
            ),
            keras.layers.Dense(
                hidden,
                activation="relu",
            ),
            keras.layers.Dense(1),
        ]
    )

    return model


def count_keras_parameters(
    model,
) -> int:
    """Return number of trainable parameters."""

    return model.count_params()


def copy_weights_from_pytorch(
    pytorch_model,
    keras_model,
):
    """
    Copy weights from PyTorch BikeNet to Keras.

    PyTorch Linear:
        weight shape = (out_features, in_features)

    Keras Dense:
        kernel shape = (in_features, out_features)

    Therefore weights must be transposed.
    """

    linear_layers = []

    for layer in pytorch_model.layers:
        if hasattr(layer, "weight"):
            linear_layers.append(layer)

    keras_dense_layers = [
        layer
        for layer in keras_model.layers
        if isinstance(
            layer,
            keras.layers.Dense,
        )
    ]

    for torch_layer, keras_layer in zip(
        linear_layers,
        keras_dense_layers,
    ):
        torch_weight = torch_layer.weight.detach().numpy().T

        torch_bias = torch_layer.bias.detach().numpy()

        keras_layer.set_weights(
            [
                torch_weight,
                torch_bias,
            ]
        )

    return keras_model


def predict_keras(
    model,
    x: np.ndarray,
):
    """Generate predictions as a flat vector."""

    predictions = model.predict(
        x,
        verbose=0,
    )

    return predictions.squeeze()

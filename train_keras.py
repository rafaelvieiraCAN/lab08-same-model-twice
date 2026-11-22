"""Keras training loop for BikeNet."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras

from baselines import (
    linear_regression_baseline,
    mean_baseline,
)
from keras_model import build_keras_model
from metrics import mae
from prepare import make_splits

EPOCHS = 20
LEARNING_RATE = 0.01
SEED = 42
BATCH_SIZE = 64


def build_numpy_arrays(splits):
    """Convert datasets to NumPy arrays."""

    train_x = splits.train.x.numpy()
    train_y = splits.train.y.numpy()

    val_x = splits.val.x.numpy()
    val_y = splits.val.y.numpy()

    test_x = splits.test.x.numpy()
    test_y = splits.test.y.numpy()

    return (
        train_x,
        train_y,
        val_x,
        val_y,
        test_x,
        test_y,
    )


def save_learning_curve(history):
    """Save learning curve."""

    Path("reports").mkdir(
        exist_ok=True,
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["loss"],
        label="Train Loss",
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Keras Learning Curve")

    plt.legend()
    plt.tight_layout()

    plt.savefig("reports/learning_curve_keras.png")

    plt.close()


def save_history(history):
    """Save epoch history."""

    frame = pd.DataFrame(
        history.history,
    )

    frame.insert(
        0,
        "epoch",
        np.arange(
            1,
            len(frame) + 1,
        ),
    )

    frame.to_csv(
        "reports/history_keras.csv",
        index=False,
    )

    return frame


def evaluate_keras(
    model,
    train_x,
    train_y,
    val_x,
    val_y,
    test_x,
    test_y,
):
    """Build Keras results table."""

    val_pred = model.predict(
        val_x,
        verbose=0,
    ).flatten()

    test_pred = model.predict(
        test_x,
        verbose=0,
    ).flatten()

    rows = [
        {
            "model": "Mean Baseline",
            "validation_mae": mean_baseline(
                train_y,
                val_y,
            ),
            "test_mae": mean_baseline(
                train_y,
                test_y,
            ),
        },
        {
            "model": "Linear Regression",
            "validation_mae": linear_regression_baseline(
                train_x,
                train_y,
                val_x,
                val_y,
            ),
            "test_mae": linear_regression_baseline(
                train_x,
                train_y,
                test_x,
                test_y,
            ),
        },
        {
            "model": "Keras Neural Network",
            "validation_mae": mae(
                val_y,
                val_pred,
            ),
            "test_mae": mae(
                test_y,
                test_pred,
            ),
        },
    ]

    frame = pd.DataFrame(rows)

    frame.to_csv(
        "reports/results_keras.csv",
        index=False,
    )

    return frame


def main(seed=SEED):
    keras.utils.set_random_seed(seed)

    splits = make_splits(seed)

    (
        train_x,
        train_y,
        val_x,
        val_y,
        test_x,
        test_y,
    ) = build_numpy_arrays(splits)

    model = build_keras_model(splits.n_features)

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=LEARNING_RATE,
        ),
        loss="mse",
        metrics=["mae"],
    )

    Path("reports").mkdir(
        exist_ok=True,
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            restore_best_weights=True,
            patience=EPOCHS,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath="reports/best_model.keras",
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        train_x,
        train_y,
        validation_data=(
            val_x,
            val_y,
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=callbacks,
    )

    save_learning_curve(
        history,
    )

    history_frame = save_history(
        history,
    )

    results = evaluate_keras(
        model,
        train_x,
        train_y,
        val_x,
        val_y,
        test_x,
        test_y,
    )

    print("\nResults")
    print(results)

    print("\nKeras training completed.")

    return {
        "history": history_frame,
        "results": results,
    }


if __name__ == "__main__":
    main()

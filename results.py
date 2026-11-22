"""Model evaluation and reporting."""

from pathlib import Path

import pandas as pd

from baselines import (
    linear_regression_baseline,
    mean_baseline,
)
from metrics import mae


def neural_network_mae(
    model,
    dataset,
):
    """Evaluate a trained neural network."""

    predictions = model(dataset.x).detach().numpy()

    targets = dataset.y.numpy()

    return mae(
        targets,
        predictions,
    )


def build_results_table(
    model,
    splits,
):
    """Create comparison table."""

    train_x = splits.train.x.numpy()
    train_y = splits.train.y.numpy()

    val_x = splits.val.x.numpy()
    val_y = splits.val.y.numpy()

    test_x = splits.test.x.numpy()
    test_y = splits.test.y.numpy()

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
            "model": "Neural Network",
            "validation_mae": neural_network_mae(
                model,
                splits.val,
            ),
            "test_mae": neural_network_mae(
                model,
                splits.test,
            ),
        },
    ]

    return pd.DataFrame(rows)


def save_results(
    model,
    splits,
):
    """Save CSV report."""

    Path("reports").mkdir(
        exist_ok=True,
    )

    results = build_results_table(
        model,
        splits,
    )

    results.to_csv(
        "reports/results.csv",
        index=False,
    )

    return results

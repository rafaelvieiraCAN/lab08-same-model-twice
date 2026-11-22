"""Baseline models for comparison."""

import numpy as np
from sklearn.linear_model import LinearRegression

from metrics import mae


def mean_baseline(
    train_labels,
    target_labels,
):
    """Predict the training mean."""

    prediction = np.full(
        len(target_labels),
        np.mean(train_labels),
    )

    return mae(
        target_labels,
        prediction,
    )


def linear_regression_baseline(
    x_train,
    y_train,
    x_target,
    y_target,
):
    """Train linear regression and evaluate MAE."""

    model = LinearRegression()

    model.fit(
        x_train,
        y_train,
    )

    prediction = model.predict(
        x_target,
    )

    return mae(
        y_target,
        prediction,
    )

"""Metrics used by the project."""

import numpy as np


def mae(
    y_true,
    y_pred,
):
    """Mean Absolute Error."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.mean(np.abs(y_true - y_pred))

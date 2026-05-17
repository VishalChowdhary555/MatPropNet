"""
metrics.py

Evaluation metrics for regression models.
"""

import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def regression_metrics(y_true, y_pred):
    """
    Compute common regression metrics.
    """

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


def print_metrics(metrics, title="Regression Results"):
    """
    Print regression metrics neatly.
    """

    print(f"\n===== {title} =====")
    print(f"MAE  : {metrics['MAE']:.4f}")
    print(f"RMSE : {metrics['RMSE']:.4f}")
    print(f"R²   : {metrics['R2']:.4f}")

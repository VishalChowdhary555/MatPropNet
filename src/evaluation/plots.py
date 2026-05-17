"""
plots.py

Visualization utilities for model evaluation.
"""

import matplotlib.pyplot as plt


def parity_plot(y_true, y_pred, title, save_path=None):
    """
    Plot true vs predicted values.
    """

    plt.figure(figsize=(7, 7))

    plt.scatter(y_true, y_pred, alpha=0.5)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())

    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    plt.xlabel("True Value")
    plt.ylabel("Predicted Value")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()


def residual_distribution(y_true, y_pred, title, save_path=None):
    """
    Plot residual error distribution.
    """

    residuals = y_true - y_pred

    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=60)

    plt.xlabel("Residual Error")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()


def training_curve(history, metric_name, title, save_path=None):
    """
    Plot a training/evaluation metric over epochs.
    """

    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history[metric_name])

    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()

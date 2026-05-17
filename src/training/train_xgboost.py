"""
train_xgboost.py

Training pipeline for XGBoost baseline model.
"""

import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.models.xgboost_baseline import build_xgboost_model
from src.evaluation.metrics import regression_metrics, print_metrics


def train_xgboost(
    feature_df,
    target_series,
    test_size: float = 0.2,
    random_state: int = 42,
    save_path: str = None
):
    """
    Train and evaluate an XGBoost model.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        feature_df,
        target_series,
        test_size=test_size,
        random_state=random_state
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = build_xgboost_model()

    print("Training XGBoost model...")
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    metrics = regression_metrics(y_test, y_pred)
    print_metrics(metrics, title="XGBoost Results")

    if save_path:
        joblib.dump(
            {
                "model": model,
                "scaler": scaler,
                "metrics": metrics
            },
            save_path
        )

        print(f"Saved XGBoost artifacts to {save_path}")

    return model, scaler, metrics, y_test, y_pred

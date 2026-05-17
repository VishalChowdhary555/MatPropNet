"""
xgboost_baseline.py

XGBoost baseline model for materials property prediction.
"""

from xgboost import XGBRegressor


def build_xgboost_model(
    n_estimators: int = 300,
    max_depth: int = 8,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42
):
    """
    Build an XGBoost regression model.
    """

    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        n_jobs=-1
    )

    return model

import pandas as pd
from typing import Dict, Any
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import xgboost as xgb

from config.settings import REVENUE_MODEL_PATH, REVENUE_BEST_PARAMS_PATH
from config.feature_lists import REVENUE_FEATURES
from config.model_params import REVENUE_PARAM_GRID
from revenue.model import RevenueModel
from revenue.features import create_revenue_features
from utils.saving import save_models, save_params_json
from utils.metrics import calculate_metrics


def tune_revenue_model(X_train: pd.DataFrame, y_train: pd.Series, n_splits: int = 5) -> dict:
    """
    Runs GridSearchCV with TimeSeriesSplit to find the best XGBoost hyperparameters.
    Uses all available CPU cores (n_jobs=-1).

    Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target (monthly revenue).
        n_splits (int): Number of temporal folds for TimeSeriesSplit.

    Returns:
        dict: Best hyperparameter dict (always includes random_state=42).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    base = xgb.XGBRegressor(random_state=42)

    gs = GridSearchCV(
        estimator=base,
        param_grid=REVENUE_PARAM_GRID,
        cv=tscv,
        scoring="neg_mean_absolute_percentage_error",
        n_jobs=-1,
        verbose=1,
    )
    gs.fit(X_train, y_train)

    best_params = gs.best_params_
    # GridSearchCV strips constructor-only kwargs; re-add random_state for reproducibility
    best_params.setdefault("random_state", 42)
    return best_params


def train_revenue_model(orders_df: pd.DataFrame, order_lines_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Full training pipeline for the revenue forecasting model:
      1. Feature engineering
      2. Temporal train/test split (last 3 months held out)
      3. GridSearchCV hyperparameter tuning on the training portion
      4. Persist best params → models/revenue_best_params.json
      5. Evaluate best model on the held-out test set
      6. Retrain on the full dataset with best params → save revenue.pkl

    Parameters:
        orders_df (pd.DataFrame): Cleaned orders DataFrame (must contain 'order_date' and 'total').
        order_lines_df (pd.DataFrame): Cleaned order lines DataFrame.

    Returns:
        Dict[str, Any]: Dict with keys: status, best_params, test_metrics, saved_path.
    """
    # 1. Feature Engineering
    df_features = create_revenue_features(orders_df, order_lines_df)

    # Ensure we have enough data points for a meaningful split + CV
    if len(df_features) < 12:
        raise ValueError(
            f"Insufficient data to train revenue model: got {len(df_features)} monthly periods, "
            "need at least 12."
        )

    # 2. Full X / y, then temporal split
    X = df_features[REVENUE_FEATURES]
    y = df_features["revenue"]

    # Hold out last 3 months as a final evaluation set (tuning only sees X_train / y_train)
    test_months = 3
    X_train, X_test = X.iloc[:-test_months], X.iloc[-test_months:]
    y_train, y_test = y.iloc[:-test_months], y.iloc[-test_months:]

    # 3. Hyperparameter tuning (TimeSeriesSplit CV on training portion only)
    print("Starting hyperparameter tuning — this may take several minutes...")
    best_params = tune_revenue_model(X_train, y_train)

    # 4. Persist best params so RevenueModel() picks them up automatically
    save_params_json(best_params, REVENUE_BEST_PARAMS_PATH)
    print(f"Best params saved to {REVENUE_BEST_PARAMS_PATH}")

    # 5. Evaluate with best params on the held-out test set
    eval_model = RevenueModel(params=best_params)
    eval_model.fit(X_train, y_train)
    y_pred = eval_model.predict(X_test)
    test_metrics = calculate_metrics(y_test, y_pred, model_type="regression")

    # 6. Retrain on the FULL dataset with best params, then save the production model
    final_model = RevenueModel(params=best_params)
    final_model.fit(X, y)
    save_models(final_model, REVENUE_MODEL_PATH)

    return {
        "status": "success",
        "best_params": best_params,
        "test_metrics": test_metrics,
        "saved_path": str(REVENUE_MODEL_PATH),
    }
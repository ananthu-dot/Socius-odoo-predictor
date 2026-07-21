import pandas as pd
from typing import Dict, Any
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import xgboost as xgb

from config.settings import PRODUCT_MODEL_PATH, PRODUCT_BEST_PARAMS_PATH, PRODUCT_FEATURE_LIST_PATH
from config.model_params import PRODUCT_PARAM_GRID
from products.model import ProductModel
from products.features import create_product_features
from utils.saving import save_models, save_params_json
from utils.metrics import calculate_metrics

def tune_product_model(X_train: pd.DataFrame, y_train: pd.Series, n_splits: int = 5) -> dict:
    """
    Runs GridSearchCV with TimeSeriesSplit to find the best XGBoost hyperparameters for product demand.

    Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target (units_sold).
        n_splits (int): Number of temporal folds for TimeSeriesSplit.

    Returns:
        dict: Best hyperparameter dict (always includes random_state=42).
    """
    splits = min(n_splits, max(2, len(X_train) - 1))
    tscv = TimeSeriesSplit(n_splits=splits)
    base = xgb.XGBRegressor(random_state=42)

    gs = GridSearchCV(
        estimator=base,
        param_grid=PRODUCT_PARAM_GRID,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=1,
    )
    gs.fit(X_train, y_train)

    best_params = gs.best_params_
    best_params.setdefault("random_state", 42)
    return best_params

def train_product_model(order_lines_df: pd.DataFrame, orders_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Trains a product demand forecasting model from order history.
    Saves best params, feature list, and trained model artifact.

    Parameters:
        order_lines_df (pd.DataFrame): Cleaned order lines.
        orders_df (pd.DataFrame): Cleaned orders.

    Returns:
        Dict[str, Any]: Execution status, best params, saved path, and metrics.
    """
    # 1. Feature Engineering
    product_features_df = create_product_features(order_lines_df, orders_df)
    
    if len(product_features_df) < 10:
        raise ValueError("Insufficient data to train product demand model.")

    # Determine feature columns
    ignore_cols = ["units_sold", "month", "product_name", "revenue", "order_count"]
    feature_cols = [c for c in product_features_df.columns if c not in ignore_cols]

    # Save feature column names for prediction alignment
    save_params_json({"feature_cols": feature_cols}, PRODUCT_FEATURE_LIST_PATH)

    # 2. Split into features and target
    X = product_features_df[feature_cols]
    y = product_features_df["units_sold"]

    # Temporal train/test split (hold out last 15% of data points)
    test_size = max(1, int(len(product_features_df) * 0.15))
    X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
    y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

    # 3. Hyperparameter Tuning
    print("Starting product model hyperparameter tuning...")
    best_params = tune_product_model(X_train, y_train)

    # 4. Save best parameters
    save_params_json(best_params, PRODUCT_BEST_PARAMS_PATH)
    print(f"Best product model params saved to {PRODUCT_BEST_PARAMS_PATH}")

    # 5. Evaluate on test set
    eval_model = ProductModel(params=best_params)
    eval_model.fit(X_train, y_train)
    y_pred = eval_model.predict(X_test)
    test_metrics = calculate_metrics(y_test, y_pred, model_type="regression")

    # 6. Retrain on full dataset & Save model
    final_model = ProductModel(params=best_params)
    final_model.fit(X, y)
    save_models(final_model, PRODUCT_MODEL_PATH)

    return {
        "status": "success",
        "best_params": best_params,
        "saved_path": str(PRODUCT_MODEL_PATH),
        "test_metrics": test_metrics
    }

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from config.settings import REVENUE_MODEL_PATH
from config.feature_lists import REVENUE_FEATURES
from revenue.features import create_revenue_features, build_forecast_row, extract_business_metrics
from utils.saving import load_models

def load_revenue_model(orders_df: pd.DataFrame = None, order_lines_df: pd.DataFrame = None) -> Any:
    """
    Loads the serialized revenue model from disk.
    If the model artifact does not exist yet, trains and saves it on the fly.

    Returns:
        Any: Loaded RevenueModel instance.
    """
    try:
        return load_models(REVENUE_MODEL_PATH)
    except FileNotFoundError:
        if orders_df is not None and order_lines_df is not None:
            from revenue.train import train_revenue_model
            print("Revenue model file not found. Auto-training initial model...")
            train_revenue_model(orders_df, order_lines_df)
            return load_models(REVENUE_MODEL_PATH)
        raise FileNotFoundError(
            f"Model file not found at: {REVENUE_MODEL_PATH}. Please train the revenue model first."
        )

def predict_revenue(orders_df: pd.DataFrame, order_lines_df: pd.DataFrame) -> float:
    """
    Predicts the revenue for the immediate next month.

    Parameters:
        orders_df (pd.DataFrame): Cleaned orders DataFrame.
        order_lines_df (pd.DataFrame): Cleaned order lines DataFrame.

    Returns:
        float: Predicted revenue for the next month.
    """
    model = load_revenue_model(orders_df, order_lines_df)
    df_features = create_revenue_features(orders_df, order_lines_df)
    latest_features = df_features.tail(1)[REVENUE_FEATURES]
    prediction = model.predict(latest_features)
    return float(prediction[0])

def forecast_revenue(
    orders_df: pd.DataFrame,
    order_lines_df: pd.DataFrame,
    months: int = 12,
) -> Dict[str, float]:
    """
    Recursively forecasts revenue for the next `months` months.

    Parameters:
        orders_df (pd.DataFrame): Cleaned orders DataFrame.
        order_lines_df (pd.DataFrame): Cleaned order lines DataFrame.
        months (int): Number of months ahead to forecast.

    Returns:
        Dict[str, float]: {"YYYY-MM": predicted_revenue} for each forecast month.
    """
    model = load_revenue_model(orders_df, order_lines_df)

    # Build feature history from actuals
    df_features = create_revenue_features(orders_df, order_lines_df)

    # Revenue history: will grow as predictions are appended
    revenue_history = df_features["revenue"].copy()

    # Business history: actuals only
    business_history = extract_business_metrics(orders_df, order_lines_df)

    results = {}
    current_date = revenue_history.index[-1]

    for _ in range(months):
        current_date = current_date + pd.offsets.MonthEnd(1)

        # Build feature row using current revenue + business history
        row = build_forecast_row(current_date, revenue_history, business_history)

        # Predict
        X = pd.DataFrame([row])[REVENUE_FEATURES]
        prediction = float(model.predict(X)[0])
        prediction = max(prediction, 0.0)  # Guard against negative predictions

        results[current_date.strftime("%Y-%m")] = prediction

        # Append prediction to revenue history for use in next iteration's lags
        revenue_history = pd.concat(
            [revenue_history, pd.Series([prediction], index=[current_date])]
        )

    return results

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from config.settings import REVENUE_MODEL_PATH
from config.feature_lists import REVENUE_FEATURES
from revenue.features import create_revenue_features
from utils.saving import load_models

def load_revenue_model() -> Any:
    """
    Loads the serialized revenue model from disk.

    Returns:
        Any: Loaded RevenueModel instance.
    """
    return load_models(REVENUE_MODEL_PATH)

def predict_revenue(monthly_revenue_df: pd.DataFrame) -> float:
    """
    Predicts the revenue for the immediate next month.

    Parameters:
        monthly_revenue_df (pd.DataFrame): Aggregated monthly revenue history.

    Returns:
        float: Predicted revenue for the next month.
    """
    # 1. Load model
    model = load_revenue_model()
    
    # 2. Extract features for the latest month (predicting next month)
    df_features = create_revenue_features(monthly_revenue_df)
    latest_features = df_features.tail(1)[REVENUE_FEATURES]
    
    # 3. Predict
    prediction = model.predict(latest_features)
    return float(prediction[0])

def forecast_revenue(monthly_revenue_df: pd.DataFrame, horizons: List[str] = ["next_month", "next_quarter", "next_year"]) -> Dict[str, float]:
    """
    Predicts future revenue performance across multiple time horizons:
    - Next Month Revenue
    - Next Quarter Revenue
    - Annual Revenue Projection

    Parameters:
        monthly_revenue_df (pd.DataFrame): Aggregated monthly revenue history.
        horizons (List[str]): Timeframes to project.

    Returns:
        Dict[str, float]: Predictions for each horizon.
    """
    results = {}
    
    # Simple recursive forecasting or aggregations based on predictions:
    # 1. Next Month
    if "next_month" in horizons:
        next_month_val = predict_revenue(monthly_revenue_df)
        results["next_month"] = next_month_val
        
    # 2. Next Quarter (Sum of next 3 months)
    if "next_quarter" in horizons:
        # Stub logic: recursively predict 3 months ahead or scale next month prediction
        next_month_val = results.get("next_month", predict_revenue(monthly_revenue_df))
        results["next_quarter"] = next_month_val * 3.0  # Placeholder stub
        
    # 3. Next Year (Sum of next 12 months)
    if "next_year" in horizons:
        next_month_val = results.get("next_month", predict_revenue(monthly_revenue_df))
        results["next_year"] = next_month_val * 12.0  # Placeholder stub
        
    return results

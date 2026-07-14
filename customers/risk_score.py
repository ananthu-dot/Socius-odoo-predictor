import pandas as pd
import numpy as np
from customers.features import create_customer_features

def calculate_risk_score(orders_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a churn/atypical dormancy risk score (0-100) for each customer.
    Score is based on recency compared to the customer's average purchase cycle.

    Parameters:
        orders_df (pd.DataFrame): Cleaned orders.

    Returns:
        pd.DataFrame: DataFrame containing customer_id and risk_score.
    """
    features = create_customer_features(orders_df)
    
    if features.empty:
        return pd.DataFrame(columns=["customer_id", "risk_score"])
        
    # Risk score heuristic:
    # 1. Average purchase interval (in days)
    # tenure / frequency (or a default of 30 if frequency is 1)
    avg_interval = np.where(
        features["frequency"] > 1,
        (features["tenure"] - features["recency"]) / (features["frequency"] - 1),
        45.0  # Default assumed purchase cycle for single-time buyers
    )
    
    # 2. Risk = Recency / Avg Interval
    # E.g., if recency is twice their average interval, risk is very high
    # We clip it to scale smoothly up to 100.
    ratio = features["recency"] / (avg_interval + 1e-5)
    
    # Risk factor: Sigmoid-like scaling of the ratio
    # If ratio is 1.0 (at their average interval), risk should be around 30-40%
    # If ratio is 2.0 (missed cycle), risk goes to 70-80%
    risk_score = (1.0 / (1.0 + np.exp(-1.5 * (ratio - 1.2)))) * 100.0
    
    # Single-time buyers with very long recency default to high churn risk
    single_buyer_mask = features["frequency"] == 1
    long_dormant_mask = features["recency"] > 90
    risk_score = np.where(single_buyer_mask & long_dormant_mask, 85.0, risk_score)
    
    features["risk_score"] = np.clip(risk_score, 0.0, 100.0).round(1)
    
    return features[["customer_id", "risk_score"]]

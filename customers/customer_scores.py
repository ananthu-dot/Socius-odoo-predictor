import pandas as pd
import numpy as np
from customers.features import create_customer_segment_features

def calculate_activity_score(orders_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes an engagement/activity score (0-100) for each customer.
    Score is calculated using recency (decaying activity) and purchase frequency.

    Parameters:
        orders_df (pd.DataFrame): Cleaned orders.

    Returns:
        pd.DataFrame: DataFrame containing customer_id and activity_score.
    """
    features = create_customer_segment_features(orders_df)
    
    if features.empty:
        return pd.DataFrame(columns=["customer", "activity_score"])
        
    # Activity score heuristic:
    # 1. Recency component: Activity decays as recency increases
    # e.g., e^(-recency / 30) for a 30-day half-life
    recency_factor = np.exp(-features["recency"] / 60.0)  # type: ignore[index]
    
    # 2. Frequency component: Log-scaled frequency of purchases
    # normalize frequency relative to the cohort's maximum
    max_freq = features["frequency"].max()  # type: ignore[index]
    freq_factor = np.log1p(features["frequency"]) / np.log1p(max_freq if max_freq > 0 else 1)  # type: ignore[index]
    
    # Combine (e.g. 50% recency factor, 50% frequency factor)
    raw_score = (0.6 * recency_factor + 0.4 * freq_factor) * 100.0
    
    # Clip to 0-100 and round
    features["activity_score"] = np.clip(raw_score, 0.0, 100.0).round(1)
    
    return features.reset_index()[["customer", "activity_score"]]


def calculate_risk_score(orders_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a churn/atypical dormancy risk score (0-100) for each customer.
    Score is based on recency compared to the customer's average purchase cycle.

    Parameters:
        orders_df (pd.DataFrame): Cleaned orders.

    Returns:
        pd.DataFrame: DataFrame containing customer and risk_score.
    """
    raise NotImplementedError("calculate_risk_score is not yet implemented.")
import pandas as pd
import numpy as np
from typing import Dict, Any
from customers.features import create_customer_features

def segment_customers(orders_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns customers into segments (e.g., Champions, Loyal, At Risk, Hibernating)
    using RFM quantile scores.

    Parameters:
        orders_df (pd.DataFrame): Cleaned orders.

    Returns:
        pd.DataFrame: DataFrame containing customer_id, RFM scores, and segment label.
    """
    # 1. Generate features
    features = create_customer_features(orders_df)
    
    if features.empty:
        return pd.DataFrame(columns=["customer_id", "recency", "frequency", "monetary_value", "segment"])
        
    # 2. Compute quintiles for Recency, Frequency, and Monetary (1-5 score)
    # Recency: Lower is better, so label 5 is lowest recency days, label 1 is highest recency days
    # Frequency & Monetary: Higher is better, so label 5 is highest frequency/monetary, label 1 is lowest
    
    # Use qcut with handles for duplicates
    def safe_qcut(series, q=5):
        try:
            return pd.qcut(series, q=q, labels=False, duplicates="drop") + 1
        except Exception:
            # Fallback if there are too few unique values for quantiles
            ranks = series.rank(method="first")
            return pd.qcut(ranks, q=q, labels=False) + 1

    features["R_score"] = safe_qcut(features["recency"], 5).map({1:5, 2:4, 3:3, 4:2, 5:1})
    features["F_score"] = safe_qcut(features["frequency"], 5)
    features["M_score"] = safe_qcut(features["monetary_value"], 5)
    
    # 3. Simple heuristic rules for segment classification
    # Concatenate R and F scores
    features["RF"] = features["R_score"].astype(str) + features["F_score"].astype(str)
    
    segment_map = {
        r"[4-5][4-5]": "Champions",
        r"[2-4][3-5]": "Loyal Customers",
        r"[3-5][1-2]": "Recent / Promising",
        r"[1-2][3-5]": "At Risk / Slipping",
        r"[1-2][1-2]": "Hibernating / Churned"
    }
    
    features["segment"] = "Standard Customers"
    for pattern, label in segment_map.items():
        features.loc[features["RF"].str.match(pattern), "segment"] = label
        
    return features[["customer_id", "recency", "frequency", "monetary_value", "segment"]]

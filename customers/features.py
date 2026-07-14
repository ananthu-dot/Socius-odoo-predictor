import pandas as pd
import numpy as np
from config.feature_lists import CUSTOMER_FEATURES

def create_customer_features(orders_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes RFM (Recency, Frequency, Monetary) and engagement features per customer.

    Parameters:
        orders_df (pd.DataFrame): Cleaned orders DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing RFM features per customer.
    """
    df = orders_df.copy()
    
    # Calculate current/max reference date for Recency
    ref_date = df["date"].max() + pd.Timedelta(days=1)
    
    # Calculate basic RFM aggregations
    customer_rfm = df.groupby("customer_id").agg(
        last_purchase_date=("date", "max"),
        first_purchase_date=("date", "min"),
        frequency=("order_id", "count"),
        monetary_value=("amount_total", "sum")
    ).reset_index()
    
    # Recency (Days since last purchase)
    customer_rfm["recency"] = (ref_date - customer_rfm["last_purchase_date"]).dt.days
    
    # Average Order Value
    customer_rfm["average_order_val"] = customer_rfm["monetary_value"] / customer_rfm["frequency"]
    
    # Tenure (Days active)
    customer_rfm["tenure"] = (ref_date - customer_rfm["first_purchase_date"]).dt.days
    
    # Active Months (unique months customer made purchases)
    df["year_month"] = df["date"].dt.to_period("M")
    active_months = df.groupby("customer_id")["year_month"].nunique().reset_index()
    active_months.rename(columns={"year_month": "active_months"}, inplace=True)
    
    # Merge active months
    customer_features = pd.merge(customer_rfm, active_months, on="customer_id", how="left")
    
    # Standardize columns matching CUSTOMER_FEATURES
    # Fill defaults/NaNs
    customer_features = customer_features.fillna(0.0)
    
    # Keep only customer_id and feature columns
    cols_to_keep = ["customer_id"] + CUSTOMER_FEATURES
    return customer_features[cols_to_keep]

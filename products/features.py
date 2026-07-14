import pandas as pd
import numpy as np
from config.feature_lists import PRODUCT_FEATURES
from data_processing.feature_engineering import create_lag_features, create_rolling_features, extract_datetime_components

def create_product_features(order_lines_df: pd.DataFrame, orders_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds product-level historical features from raw transactions.

    Parameters:
        order_lines_df (pd.DataFrame): Cleaned order lines.
        orders_df (pd.DataFrame): Cleaned orders.

    Returns:
        pd.DataFrame: DataFrame containing product feature records.
    """
    # Merge order lines with orders to get order dates
    transactions = pd.merge(
        order_lines_df,
        orders_df[["order_id", "date"]],
        on="order_id",
        how="inner"
    )
    
    # Aggregate to weekly/monthly product demand
    transactions["date_weekly"] = transactions["date"].dt.to_period("W").dt.to_timestamp()
    
    product_weekly = transactions.groupby(["product_id", "date_weekly"]).agg(
        qty_demanded=("qty", "sum"),
        revenue_generated=("line_subtotal", "sum")
    ).reset_index()
    
    # Sort chronologically within products
    product_weekly = product_weekly.sort_values(["product_id", "date_weekly"]).reset_index(drop=True)
    
    # Add calendar components
    product_weekly = extract_datetime_components(product_weekly, "date_weekly")
    
    # Add dummy price column for feature matching
    product_weekly["price"] = product_weekly["revenue_generated"] / (product_weekly["qty_demanded"] + 1e-5)
    
    # Generate lag features
    product_weekly = create_lag_features(
        product_weekly, 
        target_col="qty_demanded", 
        lag_steps=[1, 4], 
        group_col="product_id"
    )
    product_weekly.rename(columns={"qty_demanded_lag_1": "lag_1w_demand", "qty_demanded_lag_4": "lag_4w_demand"}, inplace=True)
    
    # Generate rolling features
    # Shift by 1 first to avoid data leakage
    product_weekly["qty_demanded_shifted_1"] = product_weekly.groupby("product_id")["qty_demanded"].shift(1)
    product_weekly = create_rolling_features(
        product_weekly, 
        target_col="qty_demanded_shifted_1", 
        windows=[4], 
        functions=["mean"], 
        group_col="product_id"
    )
    product_weekly.rename(columns={"qty_demanded_shifted_1_rolling_mean_4": "rolling_mean_4w"}, inplace=True)
    
    # Drop temp cols
    product_weekly.drop(columns=["qty_demanded_shifted_1"], inplace=True, errors="ignore")
    
    # Add a stub placeholder for category_id if not present
    if "category_id" not in product_weekly.columns:
        product_weekly["category_id"] = 0
        
    return product_weekly.fillna(0.0)

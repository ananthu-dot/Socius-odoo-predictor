import pandas as pd
import numpy as np
from config.feature_lists import REVENUE_FEATURES
from data_processing.feature_engineering import create_lag_features, create_rolling_features, extract_datetime_components

def create_monthly_revenue(orders_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates cleaned orders into a monthly time series format.

    Parameters:
        orders_df (pd.DataFrame): Cleaned orders DataFrame.

    Returns:
        pd.DataFrame: Monthly aggregated revenue DataFrame with columns ['year_month', 'revenue', 'total_orders', 'unique_customers'].
    """
    df = orders_df.copy()
    
    # Extract year_month
    df["year_month"] = df["date"].dt.to_period("M")
    
    monthly = df.groupby("year_month").agg(
        revenue=("amount_total", "sum"),
        total_orders=("order_id", "count"),
        unique_customers=("customer_id", "nunique")
    ).reset_index()
    
    # Sort chronologically
    monthly = monthly.sort_values("year_month").reset_index(drop=True)
    return monthly

def create_revenue_features(monthly_revenue_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates time-series features (lags, rolling averages, datetime parts) for revenue forecasting.

    Parameters:
        monthly_revenue_df (pd.DataFrame): Monthly aggregated revenue DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing generated features.
    """
    df = monthly_revenue_df.copy()
    
    # Ensure year_month is datetime-like
    if not isinstance(df["year_month"].dtype, pd.DatetimeTZDtype) and not pd.api.types.is_datetime64_any_dtype(df["year_month"]):
        df["date_timestamp"] = df["year_month"].dt.to_timestamp()
    else:
        df["date_timestamp"] = df["year_month"]
        
    df = extract_datetime_components(df, "date_timestamp")
    
    # Lags on revenue
    df = create_lag_features(df, target_col="revenue", lag_steps=[1, 2, 3])
    df.rename(columns={"revenue_lag_1": "prev_month_revenue", "revenue_lag_2": "revenue_lag_2", "revenue_lag_3": "revenue_lag_3"}, inplace=True)
    
    # Rolling stats
    # Use shifting by 1 month first to prevent data leakage during rolling calculation for prediction
    df["revenue_shifted_1"] = df["revenue"].shift(1)
    df = create_rolling_features(df, target_col="revenue_shifted_1", windows=[3], functions=["mean", "std"])
    df.rename(columns={"revenue_shifted_1_rolling_mean_3": "rolling_mean_3m", "revenue_shifted_1_rolling_std_3": "rolling_std_3m"}, inplace=True)
    
    # Drop intermediate columns
    df.drop(columns=["revenue_shifted_1", "date_timestamp"], inplace=True, errors="ignore")
    
    # Fill any NaNs resulting from lags/rolling calculations
    df = df.fillna(0.0)
    
    return df

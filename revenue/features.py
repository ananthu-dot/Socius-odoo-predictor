import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
from config.feature_lists import REVENUE_FEATURES

def create_revenue_features(orders_df: pd.DataFrame, order_lines_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates time-series features (lags, rolling averages, datetime parts) for revenue forecasting.

    Parameters:
        orders_df (pd.DataFrame): orders DataFrame.
        order_lines_df (pd.DataFrame): order lines DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing generated features.
    """

    df = (
        orders_df
        .set_index("order_date")
        .resample("ME")["total"]
        .sum()
        .rename("revenue")
        .to_frame()
    )
    
    # Calendar features
    df["month"] = df.index.month
    df["quarter"] = df.index.quarter
    df["year"] = df.index.year

    df["month_sin"] = np.sin(2*np.pi*df.index.month/12)
    df["month_cos"] = np.cos(2*np.pi*df.index.month/12)

    # Lag features
    df["revenue_lag_1"] = df["revenue"].shift(1)
    df["revenue_lag_2"] = df["revenue"].shift(2)
    df["revenue_lag_3"] = df["revenue"].shift(3)
    df["revenue_lag_6"] = df["revenue"].shift(6)

    # Rolling stats
    df["rolling_mean_3"] = (
        df["revenue"]
        .rolling(3)
        .mean()
    )

    df["rolling_mean_6"] = (
        df["revenue"]
        .rolling(6)
        .mean()
    )

    df["rolling_std_3"] = (
        df["revenue"]
        .rolling(3)
        .std()
    )

    # Business Feaures

    # Monthly order count
    monthly_orders = (
        orders_df
        .set_index("order_date")
        .resample("ME")
        .size()
    )

    # Monthly Customers
    monthly_customers = (
        orders_df
        .set_index("order_date")
        .resample("ME")["customer"]
        .nunique()
    )

    # Merge data
    orders_and_lines = orders_df.merge(order_lines_df, how='inner', on='order_reference')

    # Num unique products sold a month
    monthly_products = (
        orders_and_lines
        .set_index("order_date")
        .resample("ME")["product"]
        .nunique()
    )

    # total monthly units sold
    monthly_units = (
        orders_and_lines
        .set_index("order_date")
        .resample("ME")["quantity"]
        .sum()
    )

    # Shift features 

    for lag in [1, 3, 6]:

        df[f"order_count_lag_{lag}"] = (
            monthly_orders.shift(lag)
        )

        df[f"active_customers_lag_{lag}"] = (
            monthly_customers.shift(lag)
        )

        df[f"unique_products_lag_{lag}"] = (
            monthly_products.shift(lag)
        )

        df[f"units_lag_{lag}"] = (
            monthly_units.shift(lag)
        )

        df[f"avg_order_value_lag_{lag}"] = (
            df[f"revenue_lag_{lag}"] /
            df[f"order_count_lag_{lag}"]
        )

    df = df.dropna()
    
    return df

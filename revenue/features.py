import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
from config.feature_lists import REVENUE_FEATURES

def extract_business_metrics(orders_df: pd.DataFrame, order_lines_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts monthly business activity metrics from orders and order lines.
    """
    monthly_orders = (
        orders_df
        .set_index("order_date")
        .resample("ME")
        .size()
        .rename("order_count")
    )
    monthly_customers = (
        orders_df
        .set_index("order_date")
        .resample("ME")["customer"]
        .nunique()
        .rename("active_customers")
    )

    orders_and_lines = orders_df.merge(order_lines_df, how="inner", on="order_reference")

    monthly_products = (
        orders_and_lines
        .set_index("order_date")
        .resample("ME")["product_name" if "product_name" in orders_and_lines.columns else "product"]
        .nunique()
        .rename("unique_products")
    )
    monthly_units = (
        orders_and_lines
        .set_index("order_date")
        .resample("ME")["quantity"]
        .sum()
        .rename("units")
    )

    return pd.concat(
        [monthly_orders, monthly_customers, monthly_products, monthly_units], axis=1
    ).fillna(0)

def create_revenue_features(orders_df: pd.DataFrame, order_lines_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates time-series features (lags, rolling averages, datetime parts) for revenue forecasting.
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

    df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)

    # Lag features
    df["revenue_lag_1"] = df["revenue"].shift(1)
    df["revenue_lag_2"] = df["revenue"].shift(2)
    df["revenue_lag_3"] = df["revenue"].shift(3)
    df["revenue_lag_6"] = df["revenue"].shift(6)

    # Rolling stats
    df["rolling_mean_3"] = df["revenue"].rolling(3).mean()
    df["rolling_mean_6"] = df["revenue"].rolling(6).mean()
    df["rolling_std_3"]  = df["revenue"].rolling(3).std()

    # Business Features
    biz = extract_business_metrics(orders_df, order_lines_df)

    for lag in [1, 3, 6]:
        df[f"order_count_lag_{lag}"]      = biz["order_count"].shift(lag)
        df[f"active_customers_lag_{lag}"] = biz["active_customers"].shift(lag)
        df[f"unique_products_lag_{lag}"]  = biz["unique_products"].shift(lag)
        df[f"units_lag_{lag}"]            = biz["units"].shift(lag)
        df[f"avg_order_value_lag_{lag}"]  = (
            df[f"revenue_lag_{lag}"] / df[f"order_count_lag_{lag}"]
        )

    df = df.dropna()
    return df

def _lag_date(date: pd.Timestamp, months: int) -> pd.Timestamp:
    shifted = date - pd.DateOffset(months=months)
    return shifted + pd.offsets.MonthEnd(0)

def build_forecast_row(
    next_date: pd.Timestamp,
    revenue_history: pd.Series,
    business_history: pd.DataFrame,
) -> dict:
    row = {}
    row["month"]     = next_date.month
    row["quarter"]   = next_date.quarter
    row["year"]      = next_date.year
    row["month_sin"] = np.sin(2 * np.pi * next_date.month / 12)
    row["month_cos"] = np.cos(2 * np.pi * next_date.month / 12)

    for lag in [1, 2, 3, 6]:
        lag_d = _lag_date(next_date, lag)
        if lag_d in revenue_history.index:
            row[f"revenue_lag_{lag}"] = float(revenue_history[lag_d])
        elif len(revenue_history) >= lag:
            row[f"revenue_lag_{lag}"] = float(revenue_history.iloc[-lag])
        else:
            row[f"revenue_lag_{lag}"] = float(revenue_history.mean())

    n3 = min(3, len(revenue_history))
    n6 = min(6, len(revenue_history))
    row["rolling_mean_3"] = float(revenue_history.iloc[-n3:].mean())
    row["rolling_mean_6"] = float(revenue_history.iloc[-n6:].mean())
    row["rolling_std_3"]  = float(revenue_history.iloc[-n3:].std()) if n3 >= 2 else 0.0

    trailing_biz = business_history.tail(6).mean()

    for lag in [1, 3, 6]:
        lag_d = _lag_date(next_date, lag)
        for col in ["order_count", "active_customers", "unique_products", "units"]:
            if lag_d in business_history.index:
                row[f"{col}_lag_{lag}"] = float(business_history.loc[lag_d, col])
            else:
                row[f"{col}_lag_{lag}"] = float(trailing_biz[col])

        o = row[f"order_count_lag_{lag}"]
        row[f"avg_order_value_lag_{lag}"] = row[f"revenue_lag_{lag}"] / o if o > 0 else 0.0

    return row

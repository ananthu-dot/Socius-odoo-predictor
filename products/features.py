import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from config.feature_lists import PRODUCT_FEATURES

def create_product_features(order_lines_df: pd.DataFrame, orders_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates product-level monthly time-series features for demand forecasting.

    Parameters:
        order_lines_df (pd.DataFrame): Cleaned order lines DataFrame.
        orders_df (pd.DataFrame): Cleaned orders DataFrame.

    Returns:
        pd.DataFrame: Processed product features DataFrame.
    """
    transactions = order_lines_df.merge(
        orders_df[['order_reference', 'order_date']],
        on='order_reference',
        how='left'
    )

    transactions["month"] = (
        transactions["order_date"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    # Drop unnecessary cols if present
    cols_to_drop = [c for c in ['order_lines', 'product', 'delivery_quantity', 'product_internal_ref'] if c in transactions.columns]
    if cols_to_drop:
        transactions.drop(columns=cols_to_drop, inplace=True)

    # Format data for XGB
    monthly_products = (
        transactions
        .groupby(
            ["product_name", "product_category", "month"],
            as_index=False
        )
        .agg(
            units_sold=("quantity", "sum"),
            revenue=("subtotal", "sum"),
            order_count=("order_reference", "nunique")
        )
    )
    
    # Fill months with no data with 0
    completed_products = []

    # Loop through each product, find date range
    for product, group in monthly_products.groupby("product_name"):
        start = group["month"].min()
        end = group["month"].max()

        if pd.isna(start) or pd.isna(end):
            continue

        all_months = pd.date_range(
            start=start,
            end=end,
            freq="MS"
        )

        product_df = pd.DataFrame({
            "month": all_months
        })

        product_df["product_name"] = product

        # Get product with months that it was sold
        product_df = product_df.merge(
            group,
            on=["product_name", "month"],
            how="left"
        )

        # Fill missing months with 0
        product_df["units_sold"] = product_df["units_sold"].fillna(0)
        product_df["revenue"] = product_df["revenue"].fillna(0)
        product_df["order_count"] = product_df["order_count"].fillna(0)

        # Filling category for new rows
        product_df["product_category"] = group["product_category"].iloc[0]

        completed_products.append(product_df)

    if not completed_products:
        return pd.DataFrame()

    monthly_products = pd.concat(
        completed_products,
        ignore_index=True
    )

    # Lag features
    for lag in [1, 2, 3, 6]:
        monthly_products[f"units_lag_{lag}"] = (
            monthly_products
            .groupby("product_name")["units_sold"]
            .shift(lag)
        )

    for lag in [1, 3, 6]:
        monthly_products[f"revenue_lag_{lag}"] = (
            monthly_products
            .groupby("product_name")["revenue"]
            .shift(lag)
        )

    for lag in [1, 3]:
        monthly_products[f"orders_lag_{lag}"] = (
            monthly_products
            .groupby("product_name")["order_count"]
            .shift(lag)
        )

    # Rolling stats
    monthly_products["rolling_mean_3"] = (
        monthly_products
        .groupby("product_name")["units_sold"]
        .transform(
            lambda s: s.shift(1).rolling(3).mean()
        )
    )

    monthly_products["rolling_mean_6"] = (
        monthly_products
        .groupby("product_name")["units_sold"]
        .transform(
            lambda s: s.shift(1).rolling(6).mean()
        )
    )

    monthly_products["rolling_std_3"] = (
        monthly_products
        .groupby("product_name")["units_sold"]
        .transform(
            lambda s: s.shift(1).rolling(3).std()
        )
    )

    # Calendar features
    monthly_products["year"] = monthly_products["month"].dt.year
    monthly_products["month_num"] = monthly_products["month"].dt.month
    monthly_products["quarter"] = monthly_products["month"].dt.quarter

    # Cyclic encoding
    monthly_products["month_sin"] = np.sin(2 * np.pi * monthly_products["month_num"] / 12)
    monthly_products["month_cos"] = np.cos(2 * np.pi * monthly_products["month_num"] / 12)

    # Label encoder for product name
    product_encoder = LabelEncoder()
    monthly_products["product_id"] = product_encoder.fit_transform(
        monthly_products["product_name"]
    )

    # One Hot encoder for product category
    monthly_products = pd.get_dummies(
        monthly_products,
        columns=["product_category"],
        dtype=int
    )

    # Filter out very low selling items
    product_totals = (
        monthly_products
        .groupby("product_name")["units_sold"]
        .sum()
    )

    products_to_keep = product_totals[
        product_totals >= 50
    ].index

    monthly_products = monthly_products[
        monthly_products["product_name"].isin(products_to_keep)
    ].copy()

    return monthly_products.dropna()

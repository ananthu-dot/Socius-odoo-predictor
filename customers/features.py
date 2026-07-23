import pandas as pd
import numpy as np
from config.feature_lists import CUSTOMER_FEATURES

def create_customer_segment_features(orders_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes customer segment features per customer.

    Parameters:
        orders_df (pd.DataFrame): Cleaned orders DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing customer segment features per customer.
    """
    frequency = (
    orders_df
    .groupby("customer")
    .agg(
        frequency=("order_reference", "nunique")
    )
)

    # Total Spent
    monetary = (
        orders_df
        .groupby("customer")
        .agg(
            monetary=("total", "sum")
        )
    )
    
    # Time since last order placed (using last date from dataset)
    snapshot_date = (
        orders_df["order_date"].max()
    )

    recency = (
        orders_df
        .groupby("customer")
        .agg(
            last_purchase=("order_date", "max")
        )
    )

    recency["recency"] = (
        snapshot_date -
        recency["last_purchase"]
    ).dt.days

    # Main feature df
    customer_features = (
        frequency
        .join(monetary)
        .join(recency["recency"])
    )

    # Average order value
    customer_features["avg_order_value"] = (
        customer_features["monetary"] /
        customer_features["frequency"]
    )

    # Total units purchased
    units = (
        orders_df
        .groupby("customer")
        .agg(
            total_units=("quantity", "sum")
        )
    )

    customer_features = customer_features.join(units)

    # Average units per order
    customer_features["avg_units_per_order"] = (
        customer_features["total_units"] /
        customer_features["frequency"]
    )

    # Unique products ordered
    unique_products = (
        orders_df
        .groupby("customer")
        .agg(
            unique_products=("product_name", "nunique")
        )
    )

    customer_features = customer_features.join(
        unique_products
    )

    # Total customer lifetime
    lifetime = (
        orders_df
        .groupby("customer")
        .agg(
            first_purchase=("order_date", "min"),
            last_purchase=("order_date", "max")
        )
    )

    lifetime["customer_lifetime"] = (
        lifetime["last_purchase"] -
        lifetime["first_purchase"]
    ).dt.days


    customer_features = customer_features.join(
        lifetime["customer_lifetime"]
    )

    # Number of months with orders
    orders_df["purchase_month"] = (
        orders_df["order_date"]
        .dt.to_period("M")
    )

    months = (
        orders_df
        .groupby("customer")
        .agg(
            months_active=("purchase_month", "nunique")
        )
    )

    customer_features = customer_features.join(
        months
    )

    # Average Days Between Purchases
    orders_sorted = (
        orders_df
        .sort_values(
            ["customer", "order_date"]
        )
    )

    orders_sorted["days_between"] = (
        orders_sorted
        .groupby("customer")["order_date"]
        .diff()
        .dt.days
    )

    avg_days = (
        orders_sorted
        .groupby("customer")
        .agg(
            avg_days_between=("days_between", "mean")
        )
    )

    customer_features = customer_features.join(
        avg_days
    )

    # Handle missing/null data 
    customer_features = (
        customer_features
        .replace([np.inf, -np.inf], np.nan)
    )

    customer_features.fillna(0, inplace=True)


    return customer_features


def create_customer_repeat_features(orders_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes customer repeat purchase features per customer.

    Parameters:
        orders_df (pd.DataFrame): Cleaned orders DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing customer repeat purchase features per customer.
    """

    # Get monthly sales for each customer as seperate rows
    monthly_customer_repeat = (
        orders_df
        .assign(
            month=orders_df["order_date"].dt.to_period("M").dt.to_timestamp("M")
        )
        .groupby(
            ["customer", "month"],
            as_index=False
        )
        .agg(
            revenue=("total", "sum"),
            order_count=("order_reference", "nunique")
        )
    )

    completed_customers = []

    # Create entry customer by customer as to not have any rows before customer's first purchase
    for customer, group in monthly_customer_repeat.groupby("customer"):

        start = group["month"].min()
        end = group["month"].max()

        months = pd.date_range(
            start=start,
            end=end,
            freq="ME"
        )

        customer_df = pd.DataFrame({
            "customer": customer,
            "month": months
        })

        customer_df = customer_df.merge(
            group,
            on=["customer", "month"],
            how="left"
        )

        # Fill NA with 0
        customer_df["order_count"] = customer_df["order_count"].fillna(0)
        customer_df["revenue"] = customer_df["revenue"].fillna(0)

        completed_customers.append(customer_df)

    monthly_customer_repeat = pd.concat(
        completed_customers,
        ignore_index=True
    )

    # Cumulative Features

    # Lifetime orders
    monthly_customer_repeat["lifetime_orders"] = (
        monthly_customer_repeat
        .groupby("customer")["order_count"]
        .cumsum()
    )

    # Lifetime revenue
    monthly_customer_repeat["lifetime_revenue"] = (
        monthly_customer_repeat
        .groupby("customer")["revenue"]
        .cumsum()
    )


    # Rolling Features

    # 3-month Revenue
    monthly_customer_repeat["revenue_last_3m"] = (
        monthly_customer_repeat
        .groupby("customer")["revenue"]
        .transform(
            lambda x:
                x.rolling(
                    3,
                    min_periods=1
                ).sum()
        )
    )

    # 6-month Revenue
    monthly_customer_repeat["revenue_last_6m"] = (
        monthly_customer_repeat
        .groupby("customer")["revenue"]
        .transform(
            lambda x:
                x.rolling(
                    6,
                    min_periods=1
                ).sum()
        )
    )

    # Avg orders over last 3 months
    monthly_customer_repeat["avg_orders_last_3m"] = (
        monthly_customer_repeat
        .groupby("customer")["order_count"]
        .transform(
            lambda x:
                x.rolling(
                    3,
                    min_periods=1
                ).mean()
        )
    )

    # Trend Features

    # Revenue growth
    monthly_customer_repeat["revenue_growth"] = (
        monthly_customer_repeat
        .groupby("customer")["revenue"]
        .pct_change()
    )

    # Order growth
    monthly_customer_repeat["order_growth"] = (
        monthly_customer_repeat
        .groupby("customer")["order_count"]
        .pct_change()
    )

    monthly_customer_repeat["revenue_growth"] = (
        monthly_customer_repeat["revenue_growth"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    monthly_customer_repeat["order_growth"] = (
        monthly_customer_repeat["order_growth"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    # Purchase recency
    monthly_customer_repeat["purchase"] = (
        monthly_customer_repeat["order_count"] > 0
    )

    monthly_customer_repeat["last_purchase"] = (
        monthly_customer_repeat["month"]
        .where(monthly_customer_repeat["purchase"])
    )

    monthly_customer_repeat["last_purchase"] = (
        monthly_customer_repeat
        .groupby("customer")["last_purchase"]
        .ffill()
    )

    monthly_customer_repeat["recency"] = (
        (
            monthly_customer_repeat["month"]
            -
            monthly_customer_repeat["last_purchase"]
        )
        .dt.days
    )

    # Fill NA with very high value
    monthly_customer_repeat["recency"] = (
        monthly_customer_repeat["recency"]
        .fillna(9999)
    )

    # Customer Lifetime
    first_purchase = (
        monthly_customer_repeat
        .groupby("customer")["last_purchase"]
        .transform("min")
    )

    monthly_customer_repeat["customer_lifetime"] = (
        (
            monthly_customer_repeat["month"]
            -
            first_purchase
        )
        .dt.days
    )

    # Future target (target for predictions: 0 for no purchase in 3 months, 1 for a predicted purchase)
    monthly_customer_repeat["purchase"] = (
        monthly_customer_repeat["order_count"] > 0
    ).astype(int)

    monthly_customer_repeat["repeat_purchase"] = (
        monthly_customer_repeat
        .groupby("customer")["purchase"]
        .transform(
            lambda x: (
                x.shift(-1) +
                x.shift(-2) +
                x.shift(-3)
            ) > 0
        )
        .astype(int)
    )

    # Remove leakage from end months
    last_month = monthly_customer_repeat["month"].max()

    cutoff = last_month - pd.DateOffset(months=3)

    monthly_customer_repeat = (
        monthly_customer_repeat[
            monthly_customer_repeat["month"] <= cutoff
        ]
    )

    return monthly_customer_repeat
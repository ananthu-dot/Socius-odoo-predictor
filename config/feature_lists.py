# Centralized lists of features for machine learning models

# Features used for training and predicting revenue
REVENUE_FEATURES = [
    "month",
    "year",
    "prev_month_revenue",
    "revenue_lag_2",
    "revenue_lag_3",
    "rolling_mean_3m",
    "rolling_std_3m",
    "total_orders",
    "unique_customers"
]

# Features used for predicting product demand
PRODUCT_FEATURES = [
    "product_id",
    "category_id",
    "price",
    "month",
    "day_of_week",
    "lag_1w_demand",
    "lag_4w_demand",
    "rolling_mean_4w"
]

# Features used for predicting customer repeat purchase behavior
CUSTOMER_FEATURES = [
    "recency",           # Days since last purchase
    "frequency",         # Number of orders
    "monetary_value",    # Total spend
    "average_order_val", # Monetary / Frequency
    "tenure",            # Days active
    "active_months"      # Number of months with at least one purchase
]

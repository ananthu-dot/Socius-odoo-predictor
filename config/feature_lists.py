# Centralized lists of features for machine learning models

# Features used for training and predicting revenue
REVENUE_FEATURES = [
    # Calendar
    "month",
    "quarter",
    "year",
    "month_sin",
    "month_cos",
    # Revenue lags
    "revenue_lag_1",
    "revenue_lag_2",
    "revenue_lag_3",
    "revenue_lag_6",
    # Rolling revenue stats
    "rolling_mean_3",
    "rolling_mean_6",
    "rolling_std_3",
    # Lagged business metrics (lag 1, 3, 6)
    "order_count_lag_1",
    "order_count_lag_3",
    "order_count_lag_6",
    "active_customers_lag_1",
    "active_customers_lag_3",
    "active_customers_lag_6",
    "unique_products_lag_1",
    "unique_products_lag_3",
    "unique_products_lag_6",
    "units_lag_1",
    "units_lag_3",
    "units_lag_6",
    "avg_order_value_lag_1",
    "avg_order_value_lag_3",
    "avg_order_value_lag_6",
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

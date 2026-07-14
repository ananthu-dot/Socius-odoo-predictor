# Hyperparameters for all forecasting and prediction models

# Revenue Forecasting Model Parameters (e.g., XGBoost Regressor)
REVENUE_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "max_depth": 4,
    "random_state": 42,
    "subsample": 0.8,
    "colsample_bytree": 0.8
}

# Product Demand Forecasting Model Parameters (e.g., Prophet or XGBoost)
PRODUCT_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 5,
    "random_state": 42
}

# Customer Churn / Repeat Purchase Model Parameters (e.g., Random Forest Classifier)
CUSTOMER_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "min_samples_split": 5,
    "random_state": 42
}

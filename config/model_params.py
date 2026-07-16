# Hyperparameters for all forecasting and prediction models

# Default Revenue Forecasting Model Parameters (e.g., XGBoost Regressor)
REVENUE_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "max_depth": 4,
    "random_state": 42,
    "subsample": 0.8,
    "colsample_bytree": 0.8
}

# Param grid for GridSearchCV
REVENUE_PARAM_GRID = {
    "n_estimators":    [100, 300, 500],
    "learning_rate":   [0.01, 0.05, 0.1],
    "max_depth":       [2, 3, 4, 5, 6],
    "subsample":       [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree":[0.6, 0.7, 0.8, 0.9, 1.0],
}

# Default Product Demand Forecasting Model Parameters (e.g., Prophet or XGBoost)
PRODUCT_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 5,
    "random_state": 42
}

# Default Repeat Purchase Model Parameters (e.g., Random Forest Classifier)
CUSTOMER_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "min_samples_split": 5,
    "random_state": 42
}

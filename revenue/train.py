import pandas as pd
from typing import Dict, Any
from config.settings import REVENUE_MODEL_PATH
from config.feature_lists import REVENUE_FEATURES
from revenue.model import RevenueModel
from revenue.features import create_revenue_features
from utils.saving import save_models
from utils.metrics import calculate_metrics

def train_revenue_model(monthly_revenue_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Trains the revenue prediction model on aggregated monthly revenue data.
    Saves the trained model to the configured file path.

    Parameters:
        monthly_revenue_df (pd.DataFrame): The monthly aggregated revenue dataframe.

    Returns:
        Dict[str, Any]: Dict containing training metrics and performance results.
    """
    # 1. Feature Engineering
    df_features = create_revenue_features(monthly_revenue_df)
    
    # Ensure we have enough data points to train
    if len(df_features) < 4:
        raise ValueError("Insufficient data to train revenue model (need at least 4 monthly periods).")
        
    # 2. Split into features and target
    X = df_features[REVENUE_FEATURES]
    y = df_features["revenue"]
    
    # Simple temporal split for evaluation (e.g., last 2 months for test)
    train_size = len(df_features) - 2
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    # 3. Model Training
    model_wrapper = RevenueModel()
    model_wrapper.fit(X_train, y_train)
    
    # 4. Evaluation
    y_pred = model_wrapper.predict(X_test)
    test_metrics = calculate_metrics(y_test, y_pred, model_type="regression")
    
    # Retrain on the entire dataset before saving
    final_model = RevenueModel()
    final_model.fit(X, y)
    
    # 5. Persist Model
    save_models(final_model, REVENUE_MODEL_PATH)
    
    return {
        "status": "success",
        "saved_path": str(REVENUE_MODEL_PATH),
        "test_metrics": test_metrics
    }

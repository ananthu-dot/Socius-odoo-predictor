import pandas as pd
from typing import Dict, Any
from config.settings import PRODUCT_MODEL_PATH
from config.feature_lists import PRODUCT_FEATURES
from products.model import ProductModel
from products.features import create_product_features
from utils.saving import save_models
from utils.metrics import calculate_metrics

def train_product_model(order_lines_df: pd.DataFrame, orders_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Trains a product demand forecasting model from order history.
    Saves the trained model and returns metrics.

    Parameters:
        order_lines_df (pd.DataFrame): Cleaned order lines.
        orders_df (pd.DataFrame): Cleaned orders.

    Returns:
        Dict[str, Any]: Execution status, saved path, and metrics.
    """
    # 1. Feature Engineering
    product_features_df = create_product_features(order_lines_df, orders_df)
    
    if len(product_features_df) < 10:
        raise ValueError("Insufficient data to train product demand model.")
        
    # 2. Split into features and target
    X = product_features_df[PRODUCT_FEATURES]
    y = product_features_df["qty_demanded"]
    
    # 3. Model Training
    model = ProductModel()
    model.fit(X, y)
    
    # Calculate simple training metrics for performance baseline
    y_pred = model.predict(X)
    train_metrics = calculate_metrics(y, y_pred, model_type="regression")
    
    # 4. Save model
    save_models(model, PRODUCT_MODEL_PATH)
    
    return {
        "status": "success",
        "saved_path": str(PRODUCT_MODEL_PATH),
        "train_metrics": train_metrics
    }

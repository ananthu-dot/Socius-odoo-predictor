import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.ensemble import RandomForestClassifier
from config.settings import REPEAT_PURCHASE_MODEL_PATH
from customers.features import create_customer_segment_features
from utils.saving import save_models, load_models

def train_repeat_purchase_model(orders_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Trains a model to predict the probability that a customer will make a repeat purchase.

    Parameters:
        orders_df (pd.DataFrame): Cleaned orders.

    Returns:
        Dict[str, Any]: Model training results and metadata.
    """
    features_df = create_customer_segment_features(orders_df)
    
    if len(features_df) < 5:
        raise ValueError("Insufficient customer data to train repeat purchase model.")
        
    y = (features_df["frequency"] > 1).astype(int)
    feature_cols = [c for c in features_df.columns if c not in ["customer", "customer_id"]]
    X = features_df[feature_cols]
    
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X, y)
    
    save_models(model, REPEAT_PURCHASE_MODEL_PATH)
    
    return {
        "status": "success",
        "saved_path": str(REPEAT_PURCHASE_MODEL_PATH),
        "model_type": "RandomForestClassifier",
        "n_samples": len(features_df)
    }

def predict_repeat_purchase(orders_df: pd.DataFrame) -> pd.DataFrame:
    """
    Predicts the probability of a repeat purchase for each customer.

    Parameters:
        orders_df (pd.DataFrame): Cleaned orders.

    Returns:
        pd.DataFrame: DataFrame containing customer and repeat_probability.
    """
    features_df = create_customer_segment_features(orders_df)
    
    if features_df.empty:
        return pd.DataFrame(columns=["customer", "repeat_probability"])
        
    feature_cols = [c for c in features_df.columns if c not in ["customer", "customer_id"]]
    X = features_df[feature_cols]
    customers = features_df.index
    
    try:
        model = load_models(REPEAT_PURCHASE_MODEL_PATH)
        probs = model.predict_proba(X)[:, 1]
    except (FileNotFoundError, Exception):
        # Fallback heuristic if model is not trained/found
        probs = (features_df["frequency"] > 1).astype(float) * 0.8 + 0.1
        probs = np.clip(probs, 0.0, 1.0)

    result = pd.DataFrame({
        "customer": customers,
        "repeat_probability": probs
    })
    
    return result

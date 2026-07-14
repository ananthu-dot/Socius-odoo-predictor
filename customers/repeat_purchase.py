import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.ensemble import RandomForestClassifier
from config.settings import REPEAT_PURCHASE_MODEL_PATH
from config.feature_lists import CUSTOMER_FEATURES
from customers.features import create_customer_features
from utils.saving import save_models, load_models

def train_repeat_purchase_model(orders_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Trains a model to predict the probability that a customer will make a repeat purchase
    within a certain timeframe.

    Parameters:
        orders_df (pd.DataFrame): Cleaned orders.

    Returns:
        Dict[str, Any]: Model training results and metadata.
    """
    features_df = create_customer_features(orders_df)
    
    if len(features_df) < 5:
        raise ValueError("Insufficient customer data to train repeat purchase model.")
        
    # Generate labels: a customer has repeat purchased if frequency > 1
    y = (features_df["frequency"] > 1).astype(int)
    X = features_df[CUSTOMER_FEATURES]
    
    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X, y)
    
    # Save the trained model
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
        pd.DataFrame: DataFrame containing customer_id and repeat_probability.
    """
    features_df = create_customer_features(orders_df)
    
    if features_df.empty:
        return pd.DataFrame(columns=["customer_id", "repeat_probability"])
        
    X = features_df[CUSTOMER_FEATURES]
    
    try:
        model = load_models(REPEAT_PURCHASE_MODEL_PATH)
    except FileNotFoundError:
        # Fallback heuristic if model is not trained/found
        # Simple heuristic based on frequency and recency
        repeat_prob = (features_df["frequency"] > 1).astype(float) * 0.8 + 0.1
        result = pd.DataFrame({
            "customer_id": features_df["customer_id"],
            "repeat_probability": np.clip(repeat_prob, 0.0, 1.0)
        })
        return result
        
    # Predict probabilities of class 1 (repeat purchaser)
    probs = model.predict_proba(X)[:, 1]
    
    result = pd.DataFrame({
        "customer_id": features_df["customer_id"],
        "repeat_probability": probs
    })
    
    return result

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from config.settings import PRODUCT_MODEL_PATH, PRODUCT_FEATURE_LIST_PATH
from products.features import create_product_features
from utils.saving import load_models, load_params_json

def load_product_model(order_lines_df: pd.DataFrame = None, orders_df: pd.DataFrame = None) -> Any:
    """
    Loads the serialized product demand model from disk.
    If the model artifact does not exist yet, trains and saves it on the fly.

    Returns:
        Any: Loaded ProductModel instance.
    """
    try:
        return load_models(PRODUCT_MODEL_PATH)
    except FileNotFoundError:
        if order_lines_df is not None and orders_df is not None:
            from products.train import train_product_model
            print("Product model file not found. Auto-training initial model...")
            train_product_model(order_lines_df, orders_df)
            return load_models(PRODUCT_MODEL_PATH)
        raise FileNotFoundError(
            f"Model file not found at: {PRODUCT_MODEL_PATH}. Please train the product model first."
        )

def predict_product(
    order_lines_df: pd.DataFrame, 
    orders_df: pd.DataFrame, 
    product_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Predicts quantity demand for products in the next timeframe.

    Parameters:
        order_lines_df (pd.DataFrame): Cleaned order lines.
        orders_df (pd.DataFrame): Cleaned orders.
        product_name (Optional[str]): Specific product name to predict demand for.
            If provided, filters output to only that product.
            If None, returns predictions for all products.

    Returns:
        pd.DataFrame: DataFrame containing product_name, product_id, and predicted_demand.
    """
    # 1. Load model (auto-train if missing)
    model = load_product_model(order_lines_df, orders_df)
    
    # 2. Extract features
    features_df = create_product_features(order_lines_df, orders_df)

    if features_df.empty:
        return pd.DataFrame(columns=["product_name", "product_id", "predicted_demand"])

    # Filter by product_name if specified
    if product_name is not None:
        product_features = features_df[features_df["product_name"] == product_name]
        if product_features.empty:
            raise ValueError(f"Product '{product_name}' not found in dataset or has insufficient sales history.")
    else:
        product_features = features_df

    # Select the most recent record per product to predict future demand
    latest_features = product_features.groupby("product_name").last().reset_index()

    # Align feature columns with saved training feature set
    saved_cols_info = load_params_json(PRODUCT_FEATURE_LIST_PATH)
    if saved_cols_info and "feature_cols" in saved_cols_info:
        feature_cols = saved_cols_info["feature_cols"]
        X = latest_features.reindex(columns=feature_cols, fill_value=0)
    else:
        ignore_cols = ["units_sold", "month", "product_name", "revenue", "order_count"]
        X = latest_features[[c for c in latest_features.columns if c not in ignore_cols]]

    # 3. Predict
    predictions = model.predict(X)

    result = pd.DataFrame({
        "product_name": latest_features["product_name"],
        "product_id": latest_features["product_id"],
        "predicted_demand": np.maximum(predictions, 0.0)  # Bound by zero
    })

    return result

def forecast_product(
    order_lines_df: pd.DataFrame, 
    orders_df: pd.DataFrame, 
    product_name: Optional[str] = None, 
    steps: int = 4
) -> pd.DataFrame:
    """
    Forecasts multi-period product demand.

    Parameters:
        order_lines_df (pd.DataFrame): Cleaned order lines.
        orders_df (pd.DataFrame): Cleaned orders.
        product_name (Optional[str]): Specific product name to forecast.
        steps (int): Number of steps to project ahead.

    Returns:
        pd.DataFrame: Forecast results per product per step.
    """
    predictions = predict_product(order_lines_df, orders_df, product_name=product_name)
    
    forecasts = []
    for step in range(1, steps + 1):
        step_df = predictions.copy()
        step_df["horizon_step"] = step
        step_df["predicted_demand"] = step_df["predicted_demand"] * (1.0 - 0.02 * step)
        forecasts.append(step_df)
        
    return pd.concat(forecasts, ignore_index=True)

def forecast_category(
    order_lines_df: pd.DataFrame, 
    orders_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregates product demand forecasts to category levels.

    Parameters:
        order_lines_df (pd.DataFrame): Cleaned order lines.
        orders_df (pd.DataFrame): Cleaned orders.

    Returns:
        pd.DataFrame: Category demand forecasts.
    """
    features_df = create_product_features(order_lines_df, orders_df)
    if features_df.empty:
        return pd.DataFrame(columns=["category_id", "predicted_category_demand", "active_products"])

    product_preds = predict_product(order_lines_df, orders_df)
    
    # Merge category information from features
    latest_cat = features_df.groupby("product_name").last().reset_index()
    merged = pd.merge(product_preds, latest_cat[["product_name"]], on="product_name", how="left")
    
    # Find one-hot encoded category columns
    cat_cols = [c for c in latest_cat.columns if c.startswith("product_category_")]
    if cat_cols:
        # Determine category for each product
        latest_cat["category_name"] = latest_cat[cat_cols].idxmax(axis=1).str.replace("product_category_", "")
        merged = pd.merge(product_preds, latest_cat[["product_name", "category_name"]], on="product_name", how="left")
        
        category_forecast = merged.groupby("category_name").agg(
            predicted_category_demand=("predicted_demand", "sum"),
            active_products=("product_name", "count")
        ).reset_index()
    else:
        category_forecast = pd.DataFrame({
            "category_name": ["General"],
            "predicted_category_demand": [product_preds["predicted_demand"].sum()],
            "active_products": [len(product_preds)]
        })
    
    return category_forecast

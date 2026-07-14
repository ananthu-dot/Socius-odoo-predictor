import pandas as pd
import numpy as np
from typing import Dict, Any, List
from config.settings import PRODUCT_MODEL_PATH
from config.feature_lists import PRODUCT_FEATURES
from products.features import create_product_features
from utils.saving import load_models

def load_product_model() -> Any:
    """
    Loads the serialized product demand model from disk.
    """
    return load_models(PRODUCT_MODEL_PATH)

def predict_product(order_lines_df: pd.DataFrame, orders_df: pd.DataFrame) -> pd.DataFrame:
    """
    Predicts quantity demand for all products in the next timeframe (e.g., next week/month).

    Parameters:
        order_lines_df (pd.DataFrame): Cleaned order lines.
        orders_df (pd.DataFrame): Cleaned orders.

    Returns:
        pd.DataFrame: DataFrame containing product_id and predicted_demand.
    """
    # 1. Load model
    model = load_product_model()
    
    # 2. Extract features
    features_df = create_product_features(order_lines_df, orders_df)
    
    # Select the most recent record per product to predict future demand
    latest_features = features_df.groupby("product_id").last().reset_index()
    
    # 3. Predict
    X = latest_features[PRODUCT_FEATURES]
    predictions = model.predict(X)
    
    result = pd.DataFrame({
        "product_id": latest_features["product_id"],
        "predicted_demand": np.maximum(predictions, 0.0)  # Bound by zero
    })
    
    return result

def forecast_product(order_lines_df: pd.DataFrame, orders_df: pd.DataFrame, steps: int = 4) -> pd.DataFrame:
    """
    Forecasts multi-period product demand.

    Parameters:
        order_lines_df (pd.DataFrame): Cleaned order lines.
        orders_df (pd.DataFrame): Cleaned orders.
        steps (int): Number of weeks/months to project ahead.

    Returns:
        pd.DataFrame: Forecast results per product per step.
    """
    # Placeholder stub implementation
    predictions = predict_product(order_lines_df, orders_df)
    
    forecasts = []
    for step in range(1, steps + 1):
        step_df = predictions.copy()
        step_df["horizon_step"] = step
        # Decay or trends can be added here
        step_df["predicted_demand"] = step_df["predicted_demand"] * (1.0 - 0.02 * step)
        forecasts.append(step_df)
        
    return pd.concat(forecasts, ignore_index=True)

def forecast_category(
    order_lines_df: pd.DataFrame, 
    orders_df: pd.DataFrame, 
    products_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregates product demand forecasts to category levels.

    Parameters:
        order_lines_df (pd.DataFrame): Cleaned order lines.
        orders_df (pd.DataFrame): Cleaned orders.
        products_df (pd.DataFrame): Cleaned products metadata (for category lookup).

    Returns:
        pd.DataFrame: Category demand forecasts.
    """
    # 1. Get product predictions
    product_preds = predict_product(order_lines_df, orders_df)
    
    # 2. Merge categories
    merged = pd.merge(product_preds, products_df[["product_id", "category_id"]], on="product_id", how="left")
    
    # 3. Aggregate
    category_forecast = merged.groupby("category_id").agg(
        predicted_category_demand=("predicted_demand", "sum"),
        active_products=("product_id", "count")
    ).reset_index()
    
    return category_forecast

def identify_fast_movers(order_lines_df: pd.DataFrame, threshold_qty: float = 100.0) -> List[Dict[str, Any]]:
    """
    Identifies high-demand products based on recent unit sales volume.

    Parameters:
        order_lines_df (pd.DataFrame): Cleaned order lines.
        threshold_qty (float): Sales quantity above which a product is "fast-moving".

    Returns:
        List[Dict[str, Any]]: List of fast-moving products.
    """
    # Sum quantity by product
    grouped = order_lines_df.groupby("product_id")["qty"].sum().reset_index()
    fast_movers = grouped[grouped["qty"] >= threshold_qty]
    return fast_movers.to_dict(orient="records")

def identify_slow_movers(order_lines_df: pd.DataFrame, threshold_qty: float = 10.0) -> List[Dict[str, Any]]:
    """
    Identifies low-demand products based on recent unit sales volume.

    Parameters:
        order_lines_df (pd.DataFrame): Cleaned order lines.
        threshold_qty (float): Sales quantity below which a product is "slow-moving".

    Returns:
        List[Dict[str, Any]]: List of slow-moving products.
    """
    grouped = order_lines_df.groupby("product_id")["qty"].sum().reset_index()
    slow_movers = grouped[grouped["qty"] < threshold_qty]
    return slow_movers.to_dict(orient="records")

def identify_dead_stock(
    order_lines_df: pd.DataFrame, 
    orders_df: pd.DataFrame, 
    products_df: pd.DataFrame, 
    days_threshold: int = 90
) -> List[Dict[str, Any]]:
    """
    Identifies product stock with zero sales over a specified historical window.

    Parameters:
        order_lines_df (pd.DataFrame): Cleaned order lines.
        orders_df (pd.DataFrame): Cleaned orders (with timestamps).
        products_df (pd.DataFrame): Full products catalog.
        days_threshold (int): Idle days threshold.

    Returns:
        List[Dict[str, Any]]: List of dead stock items.
    """
    # Find last sale date for each product
    merged = pd.merge(order_lines_df, orders_df[["order_id", "date"]], on="order_id", how="inner")
    last_sales = merged.groupby("product_id")["date"].max().reset_index()
    
    # Merge with catalog to find products with no sales at all or long-dormant
    catalog = products_df[["product_id", "name"]].copy()
    catalog = pd.merge(catalog, last_sales, on="product_id", how="left")
    
    current_time = pd.Timestamp.now()
    
    # A product has dead stock if it has never been sold OR was last sold more than `days_threshold` ago
    catalog["days_since_last_sale"] = (current_time - catalog["date"]).dt.days
    
    dead_stock = catalog[
        (catalog["date"].isna()) | 
        (catalog["days_since_last_sale"] > days_threshold)
    ]
    
    return dead_stock[["product_id", "name", "days_since_last_sale"]].fillna(-1).to_dict(orient="records")

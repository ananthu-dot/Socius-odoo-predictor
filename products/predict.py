import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from config.settings import PRODUCT_MODEL_PATH, PRODUCT_FEATURE_LIST_PATH
from products.features import create_product_features
from utils.saving import load_models, load_params_json

def load_product_model() -> Any:
    """
    Loads the serialized product demand model from disk.
    """
    return load_models(PRODUCT_MODEL_PATH)

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
    # 1. Load model
    model = load_product_model()
    
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
        # Decay factor across horizon steps
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
    product_preds = predict_product(order_lines_df, orders_df)
    
    merged = pd.merge(product_preds, products_df[["product_id", "category_id"]], on="product_id", how="left")
    
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
    grouped = order_lines_df.groupby("product_id")["qty"].sum().reset_index() if "product_id" in order_lines_df.columns else order_lines_df.groupby("product")["quantity"].sum().reset_index()
    qty_col = "qty" if "qty" in grouped.columns else "quantity"
    fast_movers = grouped[grouped[qty_col] >= threshold_qty]
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
    grouped = order_lines_df.groupby("product_id")["qty"].sum().reset_index() if "product_id" in order_lines_df.columns else order_lines_df.groupby("product")["quantity"].sum().reset_index()
    qty_col = "qty" if "qty" in grouped.columns else "quantity"
    slow_movers = grouped[grouped[qty_col] < threshold_qty]
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
        orders_df (pd.DataFrame): Cleaned orders.
        products_df (pd.DataFrame): Full products catalog.
        days_threshold (int): Idle days threshold.

    Returns:
        List[Dict[str, Any]]: List of dead stock items.
    """
    date_col = "date" if "date" in orders_df.columns else "order_date"
    ref_col = "order_id" if "order_id" in orders_df.columns else "order_reference"

    merged = pd.merge(order_lines_df, orders_df[[ref_col, date_col]], on=ref_col, how="inner")
    
    prod_id_col = "product_id" if "product_id" in merged.columns else "product"
    last_sales = merged.groupby(prod_id_col)[date_col].max().reset_index()
    
    catalog = products_df[["product_id", "name"]].copy() if "product_id" in products_df.columns else products_df.copy()
    catalog_prod_col = "product_id" if "product_id" in catalog.columns else "product"
    catalog = pd.merge(catalog, last_sales, left_on=catalog_prod_col, right_on=prod_id_col, how="left")
    
    current_time = pd.Timestamp.now()
    catalog["days_since_last_sale"] = (current_time - catalog[date_col]).dt.days
    
    dead_stock = catalog[
        (catalog[date_col].isna()) | 
        (catalog["days_since_last_sale"] > days_threshold)
    ]
    
    return dead_stock.fillna(-1).to_dict(orient="records")

# API endpoint stubs for Product Demand Forecasting and Inventory Analytics
from typing import Dict, Any
import pandas as pd
from products.train import train_product_model
from products.predict import (
    predict_product,
    forecast_product,
    forecast_category,
    identify_fast_movers,
    identify_slow_movers,
    identify_dead_stock
)

def api_train_product(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    POST /api/products/train
    Triggers product demand model retraining.
    """
    if "order_lines" not in payload or "orders" not in payload:
        return {"error": "Missing key 'order_lines' or 'orders' in request body."}
        
    try:
        order_lines_df = pd.DataFrame(payload["order_lines"])
        orders_df = pd.DataFrame(payload["orders"])
        
        result = train_product_model(order_lines_df, orders_df)
        return result
    except Exception as e:
        return {"error": f"Training failed: {str(e)}"}

def api_predict_product(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    POST /api/products/predict
    Predicts next timeframe's quantity demanded for products (optionally filtered by product_name).
    """
    if "order_lines" not in payload or "orders" not in payload:
        return {"error": "Missing data."}
        
    try:
        order_lines_df = pd.DataFrame(payload["order_lines"])
        orders_df = pd.DataFrame(payload["orders"])
        product_name = payload.get("product_name")
        
        preds_df = predict_product(order_lines_df, orders_df, product_name=product_name)
        return {
            "status": "success",
            "predictions": preds_df.to_dict(orient="records")
        }
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

def api_forecast_product(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    POST /api/products/forecast
    Forecasts multi-period product demand (optionally filtered by product_name).
    """
    if "order_lines" not in payload or "orders" not in payload:
        return {"error": "Missing data."}
        
    try:
        order_lines_df = pd.DataFrame(payload["order_lines"])
        orders_df = pd.DataFrame(payload["orders"])
        product_name = payload.get("product_name")
        steps = payload.get("steps", 4)
        
        forecast_df = forecast_product(order_lines_df, orders_df, product_name=product_name, steps=steps)
        return {
            "status": "success",
            "forecasts": forecast_df.to_dict(orient="records")
        }
    except Exception as e:
        return {"error": f"Forecasting failed: {str(e)}"}

def api_forecast_category(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    POST /api/products/forecast/category
    Forecasts demand aggregated by category.
    """
    if "order_lines" not in payload or "orders" not in payload or "products" not in payload:
        return {"error": "Missing data."}
        
    try:
        order_lines_df = pd.DataFrame(payload["order_lines"])
        orders_df = pd.DataFrame(payload["orders"])
        products_df = pd.DataFrame(payload["products"])
        
        cat_forecast_df = forecast_category(order_lines_df, orders_df, products_df)
        return {
            "status": "success",
            "category_forecasts": cat_forecast_df.to_dict(orient="records")
        }
    except Exception as e:
        return {"error": f"Category forecasting failed: {str(e)}"}

def api_identify_movers(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    POST /api/products/movers
    Identifies fast-moving, slow-moving, and dead-stock products.
    """
    if "order_lines" not in payload:
        return {"error": "Missing order lines."}
        
    try:
        order_lines_df = pd.DataFrame(payload["order_lines"])
        orders_df = pd.DataFrame(payload.get("orders", []))
        products_df = pd.DataFrame(payload.get("products", []))
        
        fast_threshold = payload.get("fast_threshold", 100.0)
        slow_threshold = payload.get("slow_threshold", 10.0)
        dead_days_threshold = payload.get("dead_days_threshold", 90)
        
        fast = identify_fast_movers(order_lines_df, fast_threshold)
        slow = identify_slow_movers(order_lines_df, slow_threshold)
        
        dead = []
        if not orders_df.empty and not products_df.empty:
            dead = identify_dead_stock(order_lines_df, orders_df, products_df, dead_days_threshold)
            
        return {
            "status": "success",
            "fast_moving": fast,
            "slow_moving": slow,
            "dead_stock": dead
        }
    except Exception as e:
        return {"error": f"Movers identification failed: {str(e)}"}

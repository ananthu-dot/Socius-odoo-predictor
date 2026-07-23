# API endpoint stubs for Revenue Forecasting
from typing import Dict, Any
import pandas as pd
from revenue.predict import predict_revenue, forecast_revenue
from revenue.train import train_revenue_model

# Example structure using standard Python dictionary payloads (compatible with FastAPI, Flask, etc.)

def api_train_revenue(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    POST /api/revenue/train
    Triggers revenue model retraining.
    Expects payload with 'orders' and 'order_lines' keys.
    """
    if "orders" not in payload or "order_lines" not in payload:
        return {"error": "Missing key 'orders' or 'order_lines' in request body."}
        
    try:
        orders_df = pd.DataFrame(payload["orders"])
        order_lines_df = pd.DataFrame(payload["order_lines"])
        result = train_revenue_model(orders_df, order_lines_df)
        return result
    except Exception as e:
        return {"error": f"Training failed: {str(e)}"}

def api_predict_revenue(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    POST /api/revenue/predict
    Predicts next month's revenue.
    Expects payload with 'orders' and 'order_lines' keys.
    """
    if "orders" not in payload or "order_lines" not in payload:
        return {"error": "Missing key 'orders' or 'order_lines' in request body."}
        
    try:
        orders_df = pd.DataFrame(payload["orders"])
        order_lines_df = pd.DataFrame(payload["order_lines"])
        pred_val = predict_revenue(orders_df, order_lines_df)
        return {
            "status": "success",
            "prediction": pred_val
        }
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

def api_forecast_revenue(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    POST /api/revenue/forecast
    Recursively forecasts revenue for the next N months.
    Expects payload with 'orders', 'order_lines', and optional 'months' (default 12).
    """
    if "orders" not in payload or "order_lines" not in payload:
        return {"error": "Missing key 'orders' or 'order_lines' in request body."}
        
    try:
        orders_df = pd.DataFrame(payload["orders"])
        order_lines_df = pd.DataFrame(payload["order_lines"])
        months = payload.get("months", 12)

        forecasts = forecast_revenue(orders_df, order_lines_df, months=months)
        return {
            "status": "success",
            "forecasts": forecasts
        }
    except Exception as e:
        return {"error": f"Forecasting failed: {str(e)}"}

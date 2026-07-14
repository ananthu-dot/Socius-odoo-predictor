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
    Expects payload with historical monthly revenue data.
    """
    if "data" not in payload:
        return {"error": "Missing key 'data' in request body."}
        
    try:
        # Convert incoming JSON payload to DataFrame
        monthly_df = pd.DataFrame(payload["data"])
        
        # Trigger business/ML logic
        result = train_revenue_model(monthly_df)
        return result
    except Exception as e:
        return {"error": f"Training failed: {str(e)}"}

def api_predict_revenue(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    POST /api/revenue/predict
    Predicts next month's revenue.
    """
    if "data" not in payload:
        return {"error": "Missing key 'data' in request body."}
        
    try:
        monthly_df = pd.DataFrame(payload["data"])
        pred_val = predict_revenue(monthly_df)
        return {
            "status": "success",
            "prediction": pred_val
        }
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

def api_forecast_revenue(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    POST /api/revenue/forecast
    Forecasts revenue over multiple time horizons.
    """
    if "data" not in payload:
        return {"error": "Missing key 'data' in request body."}
        
    try:
        monthly_df = pd.DataFrame(payload["data"])
        horizons = payload.get("horizons", ["next_month", "next_quarter", "next_year"])
        
        forecasts = forecast_revenue(monthly_df, horizons=horizons)
        return {
            "status": "success",
            "forecasts": forecasts
        }
    except Exception as e:
        return {"error": f"Forecasting failed: {str(e)}"}

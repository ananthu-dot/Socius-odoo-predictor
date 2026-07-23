import os
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pandas as pd

from api.revenue import api_train_revenue, api_predict_revenue, api_forecast_revenue
from api.products import api_train_product, api_predict_product, api_forecast_product, api_forecast_category
from api.customers import api_train_customer_model, api_analyze_customers
from generate_sample_data import generate_demo_dataset

app = FastAPI(
    title="Socius Odoo Predictor API & Demo UI",
    description="Machine Learning forecasting dashboard for Revenue, Product Demand, and Customer Analytics.",
    version="1.0.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure static directory exists
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# ── Sample Data Endpoint ───────────────────────────────────────────────────────

@app.get("/api/sample-data")
def get_sample_data():
    """
    Returns 24 months of synthetic historical orders and order lines 
    so the UI can run predictions out-of-the-box with one click.
    """
    try:
        orders_df, order_lines_df = generate_demo_dataset(months=24)
        return {
            "status": "success",
            "orders": orders_df.to_dict(orient="records"),
            "order_lines": order_lines_df.to_dict(orient="records"),
            "summary": {
                "total_orders": len(orders_df),
                "total_order_lines": len(order_lines_df),
                "start_date": str(orders_df["order_date"].min()),
                "end_date": str(orders_df["order_date"].max()),
                "unique_customers": int(orders_df["customer"].nunique()),
                "unique_products": int(order_lines_df["product_name"].nunique())
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate sample data: {str(e)}")

# ── Revenue Endpoints ──────────────────────────────────────────────────────────

@app.post("/api/revenue/predict")
def route_revenue_predict(payload: Dict[str, Any]):
    return api_predict_revenue(payload)

@app.post("/api/revenue/forecast")
def route_revenue_forecast(payload: Dict[str, Any]):
    return api_forecast_revenue(payload)

@app.post("/api/revenue/train")
def route_revenue_train(payload: Dict[str, Any]):
    return api_train_revenue(payload)

# ── Product Endpoints ──────────────────────────────────────────────────────────

@app.post("/api/products/predict")
def route_product_predict(payload: Dict[str, Any]):
    return api_predict_product(payload)

@app.post("/api/products/forecast")
def route_product_forecast(payload: Dict[str, Any]):
    return api_forecast_product(payload)

@app.post("/api/products/forecast/category")
def route_product_forecast_category(payload: Dict[str, Any]):
    return api_forecast_category(payload)

@app.post("/api/products/train")
def route_product_train(payload: Dict[str, Any]):
    return api_train_product(payload)

# ── Customer Endpoints ─────────────────────────────────────────────────────────

@app.post("/api/customers/analyze")
def route_customer_analyze(payload: Dict[str, Any]):
    return api_analyze_customers(payload)

@app.post("/api/customers/train")
def route_customer_train(payload: Dict[str, Any]):
    return api_train_customer_model(payload)

# ── Static UI Route ───────────────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_ui():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Socius Odoo Predictor API is running. UI file missing."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)

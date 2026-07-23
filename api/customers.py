# API endpoint stubs for Customer Behavior Analytics
from typing import Dict, Any
import pandas as pd
from customers.segmentation import segment_customers
from customers.repeat_purchase import predict_repeat_purchase, train_repeat_purchase_model
from customers.customer_scores import calculate_activity_score

def api_train_customer_model(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    POST /api/customers/train
    Triggers repeat purchase model retraining.
    """
    if "orders" not in payload:
        return {"error": "Missing key 'orders' in request body."}
        
    try:
        orders_df = pd.DataFrame(payload["orders"])
        result = train_repeat_purchase_model(orders_df)
        return result
    except Exception as e:
        return {"error": f"Training failed: {str(e)}"}

def api_analyze_customers(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    POST /api/customers/analyze
    Analyzes customers to return:
    - Segments
    - Repeat purchase probabilities
    - Activity/Engagement scores
    """
    if "orders" not in payload:
        return {"error": "Missing key 'orders' in request body."}
        
    try:
        orders_df = pd.DataFrame(payload["orders"])
        
        # 1. Segments — returns ["customer", "segment"]
        segments_df = segment_customers(orders_df)
        
        # 2. Repeat purchase — returns ["customer_id", "repeat_probability"]
        # Note: repeat_purchase still uses old schema; key as "customer" for merging
        repeat_df = predict_repeat_purchase(orders_df)
        if "customer_id" in repeat_df.columns:
            repeat_df = repeat_df.rename(columns={"customer_id": "customer"})

        # 3. Activity score — returns ["customer", "activity_score"]
        activity_df = calculate_activity_score(orders_df)

        # Merge all metrics together on "customer"
        merged = pd.merge(segments_df, repeat_df, on="customer", how="outer")
        merged = pd.merge(merged, activity_df, on="customer", how="outer")

        return {
            "status": "success",
            "customer_analysis": merged.to_dict(orient="records")
        }
    except Exception as e:
        return {"error": f"Customer analysis failed: {str(e)}"}

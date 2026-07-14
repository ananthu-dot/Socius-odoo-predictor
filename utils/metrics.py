import numpy as np
from typing import Dict, Any
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

def calculate_metrics(y_true: Any, y_pred: Any, model_type: str = "regression") -> Dict[str, float]:
    """
    Computes performance metrics based on model type.

    Parameters:
        y_true (array-like): Ground truth values.
        y_pred (array-like): Predicted values/probabilities.
        model_type (str): Type of model, either "regression" or "classification".

    Returns:
        Dict[str, float]: Calculated metrics.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    
    if len(y_true_arr) == 0 or len(y_pred_arr) == 0:
        return {}
        
    metrics = {}
    
    if model_type == "regression":
        # Regression metrics
        metrics["mae"] = float(mean_absolute_error(y_true_arr, y_pred_arr))
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
        metrics["r2"] = float(r2_score(y_true_arr, y_pred_arr))
        
        # Mean Absolute Percentage Error (MAPE)
        mask = y_true_arr != 0
        if np.any(mask):
            metrics["mape"] = float(np.mean(np.abs((y_true_arr[mask] - y_pred_arr[mask]) / y_true_arr[mask])))
        else:
            metrics["mape"] = 0.0
            
    elif model_type == "classification":
        # Handle if y_pred is probabilities or class predictions
        if y_pred_arr.ndim > 1 or (y_pred_arr.dtype.kind in "fc" and np.any((y_pred_arr > 0) & (y_pred_arr < 1))):
            y_pred_class = (y_pred_arr >= 0.5).astype(int)
            metrics["roc_auc"] = float(roc_auc_score(y_true_arr, y_pred_arr))
        else:
            y_pred_class = y_pred_arr.astype(int)
            metrics["roc_auc"] = 0.0
            
        metrics["accuracy"] = float(accuracy_score(y_true_arr, y_pred_class))
        metrics["precision"] = float(precision_score(y_true_arr, y_pred_class, zero_division=0))
        metrics["recall"] = float(recall_score(y_true_arr, y_pred_class, zero_division=0))
        metrics["f1"] = float(f1_score(y_true_arr, y_pred_class, zero_division=0))
        
    return metrics

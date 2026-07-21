import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
SCALERS_DIR = MODELS_DIR / "scalers"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
SCALERS_DIR.mkdir(exist_ok=True)

# Configuration settings
RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2

# File paths for models and preprocessing artifacts
REVENUE_MODEL_PATH = MODELS_DIR / "revenue.pkl"
REVENUE_BEST_PARAMS_PATH = MODELS_DIR / "revenue_best_params.json"
PRODUCT_MODEL_PATH = MODELS_DIR / "product.pkl"
PRODUCT_BEST_PARAMS_PATH = MODELS_DIR / "product_best_params.json"
PRODUCT_FEATURE_LIST_PATH = MODELS_DIR / "product_features.json"
REPEAT_PURCHASE_MODEL_PATH = MODELS_DIR / "repeat_purchase.pkl"

REVENUE_ENCODER_PATH = MODELS_DIR / "revenue_encoder.pkl"
PRODUCT_ENCODER_PATH = MODELS_DIR / "product_encoder.pkl"
CUSTOMER_SCALER_PATH = MODELS_DIR / "customer_scaler.pkl"

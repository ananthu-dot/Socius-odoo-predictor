from typing import Any, Optional
import xgboost as xgb
from config.model_params import PRODUCT_PARAMS
from config.settings import PRODUCT_BEST_PARAMS_PATH
from utils.saving import load_params_json

def _load_best_product_params() -> dict:
    """
    Attempts to load tuned hyperparameters from the persisted JSON file.
    Falls back to the hardcoded PRODUCT_PARAMS defaults if the file is absent.
    """
    saved = load_params_json(PRODUCT_BEST_PARAMS_PATH)
    if saved is not None:
        saved.setdefault("random_state", 42)
        return saved
    return PRODUCT_PARAMS

class ProductModel:
    """
    Wrapper class around the product-level demand forecasting model.

    When instantiated without explicit params, automatically loads the best
    hyperparameters from product_best_params.json. Falls back to PRODUCT_PARAMS if absent.
    """
    def __init__(self, params: Optional[dict] = None):
        self.params = params if params is not None else _load_best_product_params()
        self.model = xgb.XGBRegressor(**self.params)
        
    def fit(self, X: Any, y: Any) -> None:
        """
        Trains the product demand forecasting model.
        """
        self.model.fit(X, y)
        
    def predict(self, X: Any) -> Any:
        """
        Predicts future quantity demanded for given product features.
        """
        return self.model.predict(X)

    def get_booster(self) -> Any:
        """
        Returns the underlying booster.
        """
        return self.model.get_booster()

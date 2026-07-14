from typing import Any
import xgboost as xgb
from config.model_params import PRODUCT_PARAMS

class ProductModel:
    """
    Wrapper class around the product-level demand forecasting model.
    """
    def __init__(self, params: dict = PRODUCT_PARAMS):
        self.params = params
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

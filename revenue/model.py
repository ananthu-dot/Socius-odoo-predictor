from typing import Any
import xgboost as xgb
from config.model_params import REVENUE_PARAMS

class RevenueModel:
    """
    Wrapper class around the revenue forecasting model.
    """
    def __init__(self, params: dict = REVENUE_PARAMS):
        self.params = params
        self.model = xgb.XGBRegressor(**self.params)
        
    def fit(self, X: Any, y: Any) -> None:
        """
        Trains the revenue regressor model.
        """
        self.model.fit(X, y)
        
    def predict(self, X: Any) -> Any:
        """
        Performs predictions on the input features.
        """
        return self.model.predict(X)
        
    def get_booster(self) -> Any:
        """
        Returns the underlying booster.
        """
        return self.model.get_booster()

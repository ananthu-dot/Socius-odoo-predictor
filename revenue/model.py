from typing import Any, Optional
import xgboost as xgb
from config.model_params import REVENUE_PARAMS
from config.settings import REVENUE_BEST_PARAMS_PATH
from utils.saving import load_params_json

def _load_best_params() -> dict:
    """
    Attempts to load tuned hyperparameters from the persisted JSON file.
    Falls back to the hardcoded REVENUE_PARAMS defaults if the file is absent.
    """
    saved = load_params_json(REVENUE_BEST_PARAMS_PATH)
    if saved is not None:
        # Ensure random_state is always present (may have been stripped by GridSearchCV)
        saved.setdefault("random_state", 42)
        return saved
    return REVENUE_PARAMS

class RevenueModel:
    """
    Wrapper class around the revenue forecasting model.

    When instantiated without explicit params, automatically loads the best
    hyperparameters from ``revenue_best_params.json`` (written by the tuning
    pipeline). Falls back to the hardcoded defaults in REVENUE_PARAMS if the
    JSON file does not yet exist.
    """
    def __init__(self, params: Optional[dict] = None):
        self.params = params if params is not None else _load_best_params()
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

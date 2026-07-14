import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Optional

def plot_revenue_forecast(historical: pd.DataFrame, forecast: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """
    Plots historical revenue alongside predictions.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot history
    plt.plot(historical["year_month"].astype(str), historical["revenue"], label="Historical", marker="o")
    
    # Plot forecast
    plt.plot(forecast["year_month"].astype(str), forecast["predicted_revenue"], label="Forecasted", linestyle="--", marker="s")
    
    plt.title("Revenue Forecasting")
    plt.xlabel("Date")
    plt.ylabel("Revenue")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_feature_importance(importances: pd.Series, title: str = "Feature Importance", save_path: Optional[str] = None) -> None:
    """
    Plots a horizontal bar chart of feature importances.
    """
    plt.figure(figsize=(8, 6))
    importances.sort_values().plot(kind="barh")
    plt.title(title)
    plt.xlabel("Importance Score")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

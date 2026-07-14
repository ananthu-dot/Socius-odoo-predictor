import pandas as pd
import numpy as np
from typing import List, Optional

def create_lag_features(df: pd.DataFrame, target_col: str, lag_steps: List[int], group_col: Optional[str] = None) -> pd.DataFrame:
    """
    Creates lag features for a specific column. Can group by a column (e.g., product_id).

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        target_col (str): Column to create lags from.
        lag_steps (List[int]): List of lag integers (e.g., [1, 2, 3]).
        group_col (Optional[str]): Column name to group by (e.g. for panels).

    Returns:
        pd.DataFrame: DataFrame with new lag features.
    """
    df_out = df.copy()
    for lag in lag_steps:
        col_name = f"{target_col}_lag_{lag}"
        if group_col:
            df_out[col_name] = df_out.groupby(group_col)[target_col].shift(lag)
        else:
            df_out[col_name] = df_out[target_col].shift(lag)
    return df_out

def create_rolling_features(df: pd.DataFrame, target_col: str, windows: List[int], functions: List[str] = ["mean"], group_col: Optional[str] = None) -> pd.DataFrame:
    """
    Creates rolling aggregation features.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        target_col (str): Column to apply rolling operations.
        windows (List[int]): Window sizes (e.g. [3, 7]).
        functions (List[str]): Aggregate functions (e.g. ["mean", "std"]).
        group_col (Optional[str]): Column name to group by.

    Returns:
        pd.DataFrame: DataFrame with new rolling features.
    """
    df_out = df.copy()
    for window in windows:
        for func in functions:
            col_name = f"{target_col}_rolling_{func}_{window}"
            if group_col:
                roller = df_out.groupby(group_col)[target_col].transform(lambda x: x.rolling(window, min_periods=1).agg(func))
            else:
                roller = df_out[target_col].rolling(window, min_periods=1).agg(func)
            df_out[col_name] = roller
    return df_out

def extract_datetime_components(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Extracts year, month, day, day_of_week, and quarter from a datetime column.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        date_col (str): Name of datetime column.

    Returns:
        pd.DataFrame: DataFrame with datetime components added.
    """
    df_out = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_out[date_col]):
        df_out[date_col] = pd.to_datetime(df_out[date_col], errors="coerce")
    
    df_out["year"] = df_out[date_col].dt.year
    df_out["month"] = df_out[date_col].dt.month
    df_out["day"] = df_out[date_col].dt.day
    df_out["day_of_week"] = df_out[date_col].dt.dayofweek
    df_out["quarter"] = df_out[date_col].dt.quarter
    return df_out

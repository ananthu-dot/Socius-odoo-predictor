import pandas as pd
from typing import List, Tuple, Dict, Any
from sklearn.preprocessing import LabelEncoder
from utils.saving import save_models

def encode_categories(df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Fits and transforms category columns into integer labels using LabelEncoder.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        columns (List[str]): Columns to label encode.

    Returns:
        Tuple[pd.DataFrame, Dict[str, LabelEncoder]]: Encoded DataFrame, and dictionary of encoders.
    """
    df_out = df.copy()
    encoders = {}
    
    for col in columns:
        if col in df_out.columns:
            le = LabelEncoder()
            # Handle NaNs before encoding
            df_out[col] = df_out[col].astype(str).fillna("Unknown")
            df_out[col] = le.fit_transform(df_out[col])
            encoders[col] = le
            
    return df_out, encoders

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional

def validate_data(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validates that a DataFrame contains all required columns and is not empty.

    Parameters:
        df (pd.DataFrame): The DataFrame to validate.
        required_columns (List[str]): List of column names that must exist.

    Returns:
        bool: True if the DataFrame is valid, raises ValueError otherwise.
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None")
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame is missing required columns: {missing_cols}")
    
    return True

def clean_orders(orders_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw order data: parses dates, handles missing values, and drops duplicates.

    Parameters:
        orders_df (pd.DataFrame): Raw orders DataFrame.

    Returns:
        pd.DataFrame: Cleaned orders DataFrame.
    """
    # 1. Validate structure
    required_cols = ["order_id", "customer_id", "date", "amount_total"]
    validate_data(orders_df, required_cols)
    
    cleaned_df = orders_df.copy()
    
    # 2. Date conversion and handling invalid dates
    cleaned_df["date"] = pd.to_datetime(cleaned_df["date"], errors="coerce")
    cleaned_df = cleaned_df.dropna(subset=["date"])
    
    # 3. Deduplicate
    cleaned_df = cleaned_df.drop_duplicates(subset=["order_id"])
    
    # 4. Fill missing values or drop invalid rows
    cleaned_df["amount_total"] = pd.to_numeric(cleaned_df["amount_total"], errors="coerce").fillna(0.0)
    
    return cleaned_df

def clean_order_lines(order_lines_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw order line item data (product-level transaction items).

    Parameters:
        order_lines_df (pd.DataFrame): Raw order lines DataFrame.

    Returns:
        pd.DataFrame: Cleaned order lines DataFrame.
    """
    required_cols = ["order_line_id", "order_id", "product_id", "qty", "price_unit"]
    validate_data(order_lines_df, required_cols)
    
    cleaned_df = order_lines_df.copy()
    cleaned_df = cleaned_df.drop_duplicates(subset=["order_line_id"])
    
    # Preprocessing numerical columns
    cleaned_df["qty"] = pd.to_numeric(cleaned_df["qty"], errors="coerce").fillna(0).astype(int)
    cleaned_df["price_unit"] = pd.to_numeric(cleaned_df["price_unit"], errors="coerce").fillna(0.0)
    cleaned_df["line_subtotal"] = cleaned_df["qty"] * cleaned_df["price_unit"]
    
    return cleaned_df

def clean_products(products_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw product data (metadata, categories, names).

    Parameters:
        products_df (pd.DataFrame): Raw products DataFrame.

    Returns:
        pd.DataFrame: Cleaned products DataFrame.
    """
    required_cols = ["product_id", "name", "category_id"]
    validate_data(products_df, required_cols)
    
    cleaned_df = products_df.copy()
    cleaned_df = cleaned_df.drop_duplicates(subset=["product_id"])
    
    # Standardize string fields
    cleaned_df["name"] = cleaned_df["name"].fillna("Unknown Product").astype(str).str.strip()
    cleaned_df["category_id"] = cleaned_df["category_id"].fillna(-1).astype(int)
    
    return cleaned_df

def clean_customers(customers_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw customer data.

    Parameters:
        customers_df (pd.DataFrame): Raw customers DataFrame.

    Returns:
        pd.DataFrame: Cleaned customers DataFrame.
    """
    required_cols = ["customer_id", "name", "email"]
    validate_data(customers_df, required_cols)
    
    cleaned_df = customers_df.copy()
    cleaned_df = cleaned_df.drop_duplicates(subset=["customer_id"])
    
    cleaned_df["name"] = cleaned_df["name"].fillna("Guest Customer").astype(str).str.strip()
    cleaned_df["email"] = cleaned_df["email"].fillna("").astype(str).str.strip().str.lower()
    
    return cleaned_df

def preprocess_data(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    """
    General preprocessing helper for handling nulls or scaling (if required).
    """
    processed_df = df.copy()
    # Simple example of handling missing columns globally
    if strategy == "mean":
        processed_df = processed_df.fillna(processed_df.mean(numeric_only=True))
    elif strategy == "median":
        processed_df = processed_df.fillna(processed_df.median(numeric_only=True))
    return processed_df

def clean_data(
    orders_df: pd.DataFrame,
    order_lines_df: pd.DataFrame,
    products_df: pd.DataFrame,
    customers_df: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """
    Orchestrates the data cleaning pipeline. Cleans and validates all inputs.

    Parameters:
        orders_df (pd.DataFrame): Raw orders.
        order_lines_df (pd.DataFrame): Raw order lines.
        products_df (pd.DataFrame): Raw products.
        customers_df (pd.DataFrame): Raw customers.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing cleaned datasets.
    """
    cleaned_orders = clean_orders(orders_df)
    cleaned_order_lines = clean_order_lines(order_lines_df)
    cleaned_products = clean_products(products_df)
    cleaned_customers = clean_customers(customers_df)
    
    return {
        "orders": cleaned_orders,
        "order_lines": cleaned_order_lines,
        "products": cleaned_products,
        "customers": cleaned_customers
    }

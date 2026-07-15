import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional

def get_odoo_data() -> Dict[str, pd.DataFrame]:
    """
    Gets all required data from Odoo database
    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing cleaned datasets.
        
    """
    # TODO: Implement Odoo data retrieval
    pass
    
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans column names: strips whitespace, converts to lowercase, replaces spaces with underscores.

    Parameters:
        df (pd.DataFrame): The DataFrame to clean.

    Returns:
        pd.DataFrame: The DataFrame with cleaned column names.
    """

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("/", "_")
        .str.replace("-", "_")
    )

    return df

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
    # 0. Make a copy
    cleaned_df = orders_df.copy()

    # 1. Clean column names
    cleaned_df = clean_columns(cleaned_df)

    # 2. Validate structure
    required_cols = ["order_date", "order_reference", "customer", "total", "untaxed_amount", "status"]
    validate_data(cleaned_df, required_cols)
    
    # 3. Date conversion and handling invalid dates
    cleaned_df["order_date"] = pd.to_datetime(cleaned_df["order_date"], errors="coerce")
    cleaned_df = cleaned_df.dropna(subset=["order_date", "order_reference", "total"])
    
    # 4. Deduplicate
    cleaned_df = cleaned_df.drop_duplicates(subset=["order_reference"])
    
    # 5. Fill missing values or drop invalid rows
    cleaned_df["total"] = pd.to_numeric(cleaned_df["total"], errors="coerce").fillna(0.0)
    
    # 6. remove non-fulfilled orders Quotation
    cleaned_df = cleaned_df[cleaned_df["status"] != "Cancelled"]
    cleaned_df = cleaned_df[cleaned_df["status"] != "Quotation"]
    cleaned_df = cleaned_df[cleaned_df["status"] != "Quotation Sent"]

    # 7. Standardize customer names
    cleaned_df["customer"] = (
        cleaned_df["customer"]
        .str.split(",", n=1)
        .str[0]
        .str.strip()
    )

    return cleaned_df

def clean_order_lines(order_lines_df: pd.DataFrame, orders_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw order line item data (product-level transaction items).

    Parameters:
        order_lines_df (pd.DataFrame): Raw order lines DataFrame.
        orders_df (pd.DataFrame): Cleaned orders DataFrame.

    Returns:
        pd.DataFrame: Cleaned order lines DataFrame.
    """

    # 0. Make a copy
    order_lines_df_clean = order_lines_df.copy()

    # 1. Clean column names
    order_lines_df_clean = clean_columns(order_lines_df_clean)

    #1. Remove suffix from column names
    order_lines_df_clean.columns = order_lines_df_clean.columns.str.removeprefix("order_lines_")
    
    # 2. Validate structure
    required_cols = ["order_lines", "order_reference", "quantity", "product", "delivery_quantity", "unit_price", "subtotal", "discount_(%)"]
    validate_data(order_lines_df_clean, required_cols)
    
    # 3. Deduplicate
    order_lines_df_clean = order_lines_df_clean.drop_duplicates(subset=["order_reference"])

    # 4. Fill missing values or drop invalid rows
    order_lines_df_clean = order_lines_df_clean.dropna(subset=["quantity", "order_reference", "product","subtotal"])
    
    # 5. Filter out order lines that don't have a corresponding order in orders_df
    order_lines_df_clean = order_lines_df_clean[order_lines_df_clean["order_reference"].isin(orders_df["order_reference"])]
    
    return order_lines_df_clean

def clean_products(products_df: pd.DataFrame, product_template_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw product data (metadata, categories, names).

    Parameters:
        products_df (pd.DataFrame): Raw products DataFrame.
        product_template_df (pd.DataFrame): Raw product template DataFrame.

    Returns:
        pd.DataFrame: Cleaned products DataFrame.
    """

    # 0. Make a copy
    products_df_clean = products_df.copy()
    product_template_df_clean = product_template_df.copy()

    # 1. Clean column names
    products_df_clean = clean_columns(products_df_clean)
    product_template_df_clean = clean_columns(product_template_df_clean)

    # 2. Remove suffix from column names
    products_df_clean.columns = products_df_clean.columns.str.removeprefix("product_")

    # 3. Validate structure
    required_cols = ["internal_reference", "name", "cost", "sales_price", "product_category", "active"]
    validate_data(products_df_clean, required_cols)

    required_cols = ["internal_reference", "product_category", "name", "sales_price", "cost", "product_type", "unit"]
    validate_data(product_template_df_clean, required_cols)
    
    # 4. Deduplicate & drop NA
    products_df_clean = products_df_clean.drop_duplicates(subset=["internal_reference", ])
    products_df_clean = products_df_clean.dropna(subset=["internal_reference", "name", "cost", "sales_price"])
    
    product_template_df_clean = product_template_df_clean.drop_duplicates(subset=["internal_reference", ])
    product_template_df_clean = product_template_df_clean.dropna(subset=["internal_reference", "product_category", "name", "sales_price", "cost"])

    # merge products and product templates to get unique product information for all products
    products_merged_df = products_df_clean.merge(
        product_template_df_clean,
        on="internal_reference",
        how="outer",
        suffixes=("_prod", "_templ")
    )

    products_merged_df.drop(columns=["name_templ", "product_category_templ", "sales_price_templ", "cost_templ"], inplace=True)
    products_merged_df.columns = products_merged_df.columns.str.removesuffix("_prod")

    return products_merged_df

def clean_customers(customers_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw customer data.

    Parameters:
        customers_df (pd.DataFrame): Raw customers DataFrame.

    Returns:
        pd.DataFrame: Cleaned customers DataFrame.
    """
    # NOTE: customer df might be completely unnecessary as customer names are already included in the orders df
    # However, if we need more customer information, we can use this df

    # 0. Make a copy
    customers_df_clean = customers_df.copy()

    # 1. Clean column names
    customers_df_clean = clean_columns(customers_df_clean)

    # 2. Validate structure
    required_cols = ["display_name", "country"]
    validate_data(customers_df_clean, required_cols)
    
    # 3. drop NA
    customers_df_clean = customers_df_clean.dropna(subset=["display_name", "country"])
    
    # 4. Standardize company names
    customers_df_clean["company_name"] = customers_df_clean["display_name"].str.split(",", n=1).str[0].str.strip()
    
    # 5. Deduplicate based on company name
    customers_df_clean = (
        customers_df_clean
        .sort_values("display_name")
        .drop_duplicates(subset="company_name", keep="first")
    )
    
    return customers_df_clean

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
        (all above dfs are raw from Odoo export)

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

import pandas as pd
import numpy as np
from typing import Dict, Any
from customers.features import create_customer_segment_features
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def segment_customers(orders_df: pd.DataFrame, desired_features: list[str] = None, k: int = 4) -> pd.DataFrame:
    """
    Clusters customers into segments based on their RFM (Recency, Frequency, Monetary) values.

    Args:
        orders_df (pd.DataFrame): DataFrame containing order information.
        desired_features (list[str], optional): List of features to use for clustering.
            If None, uses default features from create_customer_segment_features.
            Defaults to None.
        k (int, optional): Number of clusters to create.
            Defaults to 4.

    Returns:
        pd.DataFrame: DataFrame with customer segments.
    """
    customer_features = create_customer_segment_features(orders_df)  

    if desired_features:
        X = customer_features[desired_features]
    else:
        X = customer_features
    
    # Apply log transformation to account for skew
    customer_features["monetary"] = np.log1p(customer_features["monetary"])
    customer_features["avg_order_value"] = np.log1p(customer_features["avg_order_value"])
    customer_features["total_units"] = np.log1p(customer_features["total_units"])

    # Scale fatures (accounts for large differences in scale)
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # Train Model
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    customer_features["cluster"] = (
        kmeans.fit_predict(X_scaled)
    )

    # Determine optimal segment names by sorting clusters
    cluster_summary = (
        customer_features
        .groupby("cluster")
        .agg(
            frequency=("frequency", "mean"),
            monetary=("monetary", "mean"),
            recency=("recency", "mean")
        )
    )

    # Add a composite score for ranking segments (frequency x monetary)
    cluster_summary["score"] = (
        cluster_summary["frequency"]
        * cluster_summary["monetary"]
    )

    # Map clusters to segment names based on score (high to low)
    cluster_summary = cluster_summary.sort_values(
        "score",
        ascending=False
    )

    if k == 4:
        labels = [
            "VIP Customers",
            "Loyal Customers",
            "Regular Customers",
            "At Risk Customers"
        ]

    elif k == 3:
        labels = [
            "VIP Customers",
            "Loyal Customers",
            "At Risk Customers"
        ]

    elif k == 5:
        labels = [
            "VIP Customers",
            "Loyal Customers",
            "Regular Customers",
            "Occasional Customers",
            "At Risk Customers"
        ]

    if k in [4, 3, 5]:
        mapping = {}

        for label, cluster in zip(labels, cluster_summary.index):
            mapping[cluster] = label

    else:
        # Fallback to default labeling if k is not 3, 4, or 5
        labels = [f"Segment {i}" for i in range(k)]
        
        mapping = {}

        for label, cluster in zip(labels, cluster_summary.index):
            mapping[cluster] = label

    customer_features["segment"] = customer_features["cluster"].map(mapping)

    return customer_features.reset_index()[["customer", "segment"]]

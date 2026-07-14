import pickle
import os
from pathlib import Path
from typing import Any

def save_models(model: Any, filepath: str) -> None:
    """
    Serializes a model or encoder to disk using pickle.

    Parameters:
        model (Any): Object to save.
        filepath (str): Destination path.
    """
    path = Path(filepath)
    # Ensure parent folders exist
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "wb") as f:
        pickle.dump(model, f)

def load_models(filepath: str) -> Any:
    """
    Loads a serialized model or encoder from disk.

    Parameters:
        filepath (str): Source path.

    Returns:
        Any: Loaded Python object.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at: {filepath}")
        
    with open(path, "rb") as f:
        return pickle.load(f)

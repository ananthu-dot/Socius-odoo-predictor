import pickle
import json
import os
from pathlib import Path
from typing import Any, Optional

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

def save_params_json(params: dict, filepath) -> None:
    """
    Serialises a hyperparameter dict to a JSON file.

    Parameters:
        params (dict): Parameter dictionary to save.
        filepath: Destination path (str or Path).
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

def load_params_json(filepath) -> Optional[dict]:
    """
    Loads a hyperparameter dict from a JSON file.
    Returns None if the file does not yet exist (caller should fall back to defaults).

    Parameters:
        filepath: Source path (str or Path).

    Returns:
        Optional[dict]: Loaded parameter dict, or None.
    """
    path = Path(filepath)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

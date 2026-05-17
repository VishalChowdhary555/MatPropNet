"""
save_load.py

Helper functions for saving and loading models and project artifacts.
"""

import os
import joblib
import torch


def ensure_dir(path: str):
    """
    Create a directory if it does not exist.
    """

    os.makedirs(path, exist_ok=True)


def save_joblib_object(obj, path: str):
    """
    Save a Python object using joblib.
    """

    directory = os.path.dirname(path)

    if directory:
        ensure_dir(directory)

    joblib.dump(obj, path)

    print(f"Saved object to {path}")


def load_joblib_object(path: str):
    """
    Load a joblib object.
    """

    return joblib.load(path)


def save_torch_model(model, path: str):
    """
    Save PyTorch model state dict.
    """

    directory = os.path.dirname(path)

    if directory:
        ensure_dir(directory)

    torch.save(model.state_dict(), path)

    print(f"Saved PyTorch model to {path}")


def load_torch_model(model, path: str, device=None):
    """
    Load PyTorch model state dict.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.load_state_dict(
        torch.load(path, map_location=device)
    )

    model.to(device)

    return model

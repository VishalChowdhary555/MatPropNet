"""
load_jarvis.py

Utilities for loading the JARVIS-DFT 3D materials dataset.
"""

import pandas as pd
from jarvis.db.figshare import data


def load_jarvis_dft_3d() -> pd.DataFrame:
    """
    Load the JARVIS-DFT 3D dataset and return it as a pandas DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame containing JARVIS materials records.
    """

    print("Loading JARVIS-DFT 3D dataset...")

    dataset = data("dft_3d")
    df = pd.DataFrame(dataset)

    print(f"Dataset loaded successfully. Shape: {df.shape}")

    return df


if __name__ == "__main__":
    df = load_jarvis_dft_3d()
    print(df.head())

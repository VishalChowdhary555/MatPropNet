"""
preprocess.py

Data cleaning and preprocessing utilities for materials datasets.
"""

import pandas as pd
import numpy as np


TARGET_COLUMNS = [
    "formation_energy_peratom",
    "optb88vdw_bandgap",
    "bulk_modulus_kv",
    "shear_modulus_gv"
]


USEFUL_COLUMNS = [
    "jid",
    "formula",
    "atoms",
    "density",
    "spg_number",
    "spg_symbol",
    "crys",
    "nat"
] + TARGET_COLUMNS


def select_relevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select useful columns from the raw JARVIS dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.

    Returns
    -------
    pd.DataFrame
        Reduced dataset with important columns.
    """

    return df[USEFUL_COLUMNS].copy()


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert selected columns to numeric types.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    numeric_columns = [
        "density",
        "nat"
    ] + TARGET_COLUMNS

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def remove_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows containing missing required values.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    required_columns = [
        "atoms",
        "formula",
        "density",
        "nat"
    ] + TARGET_COLUMNS

    df = df.dropna(subset=required_columns)

    return df.reset_index(drop=True)


def apply_physical_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove physically invalid samples.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    df = df[
        (df["density"] > 0) &
        (df["nat"] > 0) &
        (df["bulk_modulus_kv"] > 0) &
        (df["shear_modulus_gv"] > 0)
    ]

    return df.reset_index(drop=True)


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete preprocessing pipeline.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Cleaned dataset.
    """

    print("Selecting relevant columns...")
    df = select_relevant_columns(df)

    print("Converting numeric columns...")
    df = convert_numeric_columns(df)

    print("Removing missing values...")
    df = remove_missing_values(df)

    print("Applying physical filters...")
    df = apply_physical_filters(df)

    print(f"Final cleaned dataset shape: {df.shape}")

    return df


if __name__ == "__main__":

    from load_jarvis import load_jarvis_dft_3d

    raw_df = load_jarvis_dft_3d()

    clean_df = preprocess_dataset(raw_df)

    print(clean_df.head())

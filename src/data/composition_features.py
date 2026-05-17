"""
composition_features.py

Composition-based feature engineering using pymatgen and matminer.
"""

import pandas as pd
import numpy as np

from pymatgen.core.composition import Composition

from matminer.featurizers.composition import (
    ElementProperty,
    Stoichiometry,
    ValenceOrbital,
    IonProperty
)

from matminer.featurizers.base import MultipleFeaturizer
from sklearn.feature_selection import VarianceThreshold


def add_composition_objects(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert chemical formulas into pymatgen Composition objects.
    """

    df = df.copy()
    df["composition"] = df["formula"].apply(Composition)

    return df


def build_composition_featurizer() -> MultipleFeaturizer:
    """
    Build a matminer composition featurizer.
    """

    featurizer = MultipleFeaturizer([
        Stoichiometry(),
        ElementProperty.from_preset("magpie"),
        ValenceOrbital(),
        IonProperty()
    ])

    return featurizer


def generate_composition_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate composition descriptors using matminer.
    """

    df = add_composition_objects(df)

    featurizer = build_composition_featurizer()

    print("Generating composition features...")

    featured_df = featurizer.featurize_dataframe(
        df,
        col_id="composition",
        ignore_errors=True
    )

    print("Composition feature generation completed.")

    return featured_df


def clean_feature_matrix(
    featured_df: pd.DataFrame,
    target_columns=None,
    missing_threshold: float = 0.1
) -> pd.DataFrame:
    """
    Keep numeric descriptors, remove leakage columns, handle missing values,
    and remove constant features.
    """

    if target_columns is None:
        target_columns = [
            "formation_energy_peratom",
            "optb88vdw_bandgap",
            "bulk_modulus_kv",
            "shear_modulus_gv"
        ]

    feature_df = featured_df.select_dtypes(include=[np.number]).copy()

    feature_df = feature_df.drop(
        columns=target_columns,
        errors="ignore"
    )

    threshold = missing_threshold * len(feature_df)

    valid_columns = feature_df.columns[
        feature_df.isna().sum() < threshold
    ]

    feature_df = feature_df[valid_columns]
    feature_df = feature_df.fillna(feature_df.median())

    selector = VarianceThreshold(threshold=0)

    X_features = selector.fit_transform(feature_df)

    selected_columns = feature_df.columns[selector.get_support()]

    feature_df = pd.DataFrame(
        X_features,
        columns=selected_columns
    )

    print(f"Final composition feature matrix shape: {feature_df.shape}")

    return feature_df


if __name__ == "__main__":

    from src.data.load_jarvis import load_jarvis_dft_3d
    from src.data.preprocess import preprocess_dataset

    raw_df = load_jarvis_dft_3d()
    clean_df = preprocess_dataset(raw_df)

    featured_df = generate_composition_features(clean_df)
    feature_df = clean_feature_matrix(featured_df)

    print(feature_df.head())

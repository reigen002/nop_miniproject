"""
Dataset loading and preprocessing for House Prices regression.

Uses the California Housing dataset from scikit-learn as a high-dimensional
regression benchmark (can be swapped for the Kaggle House Prices dataset).
Applies feature engineering, missing-value imputation, and normalization.
"""

import ssl
import os
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer

# Fix SSL certificate verification on macOS / Python 3.13
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context



def load_house_prices(poly_degree=2, interaction_only=False, test_size=0.2, val_size=0.15, random_state=42):
    """
    Load and preprocess the housing dataset with polynomial feature expansion
    to create a high-dimensional regression problem suitable for sparse optimization.

    Parameters
    ----------
    poly_degree : int
        Degree for polynomial feature expansion (default=2 creates ~100+ features).
    interaction_only : bool
        If True, only interaction features are produced (no x^2 terms).
    test_size : float
        Fraction of data reserved for final test evaluation.
    val_size : float
        Fraction of remaining data reserved for validation.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        'X_train', 'X_val', 'X_test' : np.ndarray — feature matrices
        'y_train', 'y_val', 'y_test' : np.ndarray — target vectors
        'feature_names' : list[str]   — names of all features (after expansion)
        'scaler' : StandardScaler     — fitted scaler for inverse transforms
        'n_original_features' : int   — number of features before expansion
        'n_total_features' : int      — number of features after expansion
    """
    # --- Load base dataset ---
    housing = fetch_california_housing()
    X_raw = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = housing.target.copy()

    # --- Handle missing values (for robustness if swapped to Kaggle data) ---
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X_raw), columns=X_raw.columns)

    n_original = X_imputed.shape[1]

    # --- Polynomial feature expansion to create high-dimensional problem ---
    poly = PolynomialFeatures(degree=poly_degree, interaction_only=interaction_only, include_bias=False)
    X_poly = poly.fit_transform(X_imputed)
    feature_names = poly.get_feature_names_out(X_imputed.columns)

    print(f"[Dataset] Original features: {n_original}")
    print(f"[Dataset] After degree-{poly_degree} expansion: {X_poly.shape[1]} features")
    print(f"[Dataset] Samples: {X_poly.shape[0]}")

    # --- Train / Validation / Test split ---
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_poly, y, test_size=test_size, random_state=random_state
    )
    relative_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val, random_state=random_state
    )

    # --- Normalize features (critical for gradient-based optimization) ---
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print(f"[Dataset] Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'feature_names': list(feature_names),
        'scaler': scaler,
        'n_original_features': n_original,
        'n_total_features': X_poly.shape[1],
    }


if __name__ == "__main__":
    data = load_house_prices(poly_degree=2)
    print(f"\nFeature names (first 20): {data['feature_names'][:20]}")
    print(f"Target range: [{data['y_train'].min():.2f}, {data['y_train'].max():.2f}]")

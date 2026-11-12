"""Raw-feature encoding for training and inference.

Tree models do not require scaling. Persist a fitted pipeline if a future model
needs learned transformations; never fit a scaler on an uploaded batch.
"""
import numpy as np
import pandas as pd

NUMERIC_FEATURES = (
    "duration", "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss", "spkts", "dpkts"
)
CATEGORICAL_FEATURES = ("proto", "service", "state")
MAX_ROWS = 100_000


def encode_traffic(df, feature_names):
    """Validate raw traffic and encode it in the model's exact feature order."""
    required = list(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    if df.empty or len(df) > MAX_ROWS:
        raise ValueError(f"Provide between 1 and {MAX_ROWS:,} rows.")
    if not df.columns.is_unique:
        raise ValueError("Duplicate column names are not supported.")
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    expected = list(feature_names)
    if len(expected) != len(set(expected)) or not set(NUMERIC_FEATURES).issubset(expected):
        raise ValueError("The model feature schema is invalid.")
    supported = set(NUMERIC_FEATURES)
    for col in CATEGORICAL_FEATURES:
        supported.update(f for f in expected if f.startswith(col + "_"))
    if set(expected) != supported:
        raise ValueError("The model feature schema contains unsupported features.")

    clean = df[required].copy()
    for col in NUMERIC_FEATURES:
        values = pd.to_numeric(clean[col], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"Column '{col}' must contain finite numbers with no missing values.")
        if (values < 0).any():
            raise ValueError(f"Column '{col}' cannot contain negative values.")
        clean[col] = values
    for col in CATEGORICAL_FEATURES:
        if clean[col].isna().any():
            raise ValueError(f"Column '{col}' contains missing values.")
        allowed = {f[len(col) + 1:] for f in expected if f.startswith(col + "_")}
        if not clean[col].isin(allowed).all():
            raise ValueError(f"Column '{col}' must use these categories: {', '.join(sorted(allowed))}.")
    encoded = pd.get_dummies(clean, columns=list(CATEGORICAL_FEATURES), dtype=float)
    return encoded.reindex(columns=expected, fill_value=0).astype(float)


def preprocess_features(df, drop_cols=("label", "attack_cat", "severity_score")):
    """Encode training features without introducing an unsaved scaling step."""
    X = df.drop(columns=list(drop_cols), errors="ignore")
    X_encoded = pd.get_dummies(X, dtype=float)
    return X_encoded.to_numpy(), X_encoded.columns

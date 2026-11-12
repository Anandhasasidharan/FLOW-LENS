"""CSV parsing and prediction checks, independent of the UI."""
import csv
import io

import numpy as np
import pandas as pd

from src.data_preprocessing import MAX_ROWS

MAX_FILE_SIZE = 50 * 1024 * 1024


def read_traffic_csv(content):
    if len(content) > MAX_FILE_SIZE:
        raise ValueError("Maximum file size is 50 MB.")
    try:
        text = content.decode("utf-8-sig")
        headers = next(csv.reader(io.StringIO(text)))
        if len(headers) != len(set(headers)):
            raise ValueError("Duplicate column names are not supported.")
        df = pd.read_csv(io.StringIO(text), nrows=MAX_ROWS + 1)
    except (UnicodeError, StopIteration, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError("Provide a valid UTF-8 CSV with a header and data rows.") from exc
    if df.empty or len(df) > MAX_ROWS:
        raise ValueError(f"Provide between 1 and {MAX_ROWS:,} rows.")
    return df


def predict_severity(model, encoded):
    # Older artifacts were fitted on arrays, newer models may carry feature names.
    features = encoded if hasattr(model, "feature_names_in_") else encoded.to_numpy()
    scores = np.asarray(model.predict(features), dtype=float)
    if scores.shape != (len(encoded),) or not np.isfinite(scores).all():
        raise ValueError("The model returned invalid predictions.")
    if ((scores < 0) | (scores > 10)).any():
        raise ValueError("The model returned scores outside the supported 0–10 range.")
    return scores


def risk_categories(scores):
    """Display bands are fixed conventions, pending domain validation."""
    return np.where(np.asarray(scores) >= 7, "High", np.where(np.asarray(scores) >= 3, "Medium", "Low"))


def export_csv(df):
    """Escape spreadsheet formulas in text cells when exporting uploaded data."""
    safe = df.copy()
    for col in safe.select_dtypes(include=["object", "string"]).columns:
        safe[col] = safe[col].map(
            lambda v: "'" + v if isinstance(v, str) and v.lstrip().startswith(("=", "+", "-", "@")) else v
        )
    return safe.to_csv(index=False)

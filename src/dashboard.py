"""Batch analysis for the prototype dashboard."""
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd

BANDS = ("Low", "Medium", "High")


def filter_traffic(results, bands, protocols, severity_range):
    mask = (
        results['Risk_Category'].isin(bands)
        & results['proto'].isin(protocols)
        & results['Risk_Score'].between(*severity_range)
    )
    return results.loc[mask].copy()


def severity_distribution(results):
    counts, _ = np.histogram(results['Risk_Score'], bins=np.arange(11))
    return pd.DataFrame({"Risk index interval": [f"{n}–{n+1}" for n in range(10)], "Records": counts}).set_index('Risk index interval')


def analysis_report(results, model_name, source):
    scores = results['Risk_Score']
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "model": model_name,
        "records": len(results),
        "risk_score": {"mean": float(scores.mean()), "min": float(scores.min()), "max": float(scores.max())},
        "risk_counts": {band: int((results['Risk_Category'] == band).sum()) for band in BANDS},
        "protocol_counts": {str(k): int(v) for k, v in results['proto'].value_counts().items()},
        "thresholds": {"medium_at": 3, "high_at": 7},
        "limitations": "Research prototype. Risk score is estimated attack probability multiplied by 10, not severity. Display bands are not validated alert thresholds. Conformal coverage is not guaranteed under drift.",
    }
    return json.dumps(report, indent=2, allow_nan=False)

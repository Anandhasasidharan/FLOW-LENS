"""Validated inference for the independently trained NetFlow detector."""
import numpy as np
import pandas as pd

FEATURES = (
    'L4_DST_PORT', 'PROTOCOL', 'L7_PROTO', 'IN_BYTES', 'OUT_BYTES',
    'IN_PKTS', 'OUT_PKTS', 'TCP_FLAGS', 'FLOW_DURATION_MILLISECONDS',
)
CATEGORICAL = ('PROTOCOL', 'L7_PROTO')


def validate_netflow(raw):
    if raw.empty or len(raw) > 100_000:
        raise ValueError('Provide between 1 and 100,000 records.')
    if not raw.columns.is_unique:
        raise ValueError('Duplicate column names are not supported.')
    missing = sorted(set(FEATURES) - set(raw.columns))
    if missing:
        raise ValueError('Missing required columns: ' + ', '.join(missing))
    clean = raw[list(FEATURES)].apply(pd.to_numeric, errors='coerce')
    if not np.isfinite(clean.to_numpy(dtype=float)).all() or (clean < 0).any().any():
        raise ValueError('All model inputs must be finite, nonnegative numbers with no missing values.')
    for col, maximum in [('L4_DST_PORT', 65535), ('PROTOCOL', 255), ('TCP_FLAGS', 255)]:
        if ((clean[col] > maximum) | (clean[col] % 1 != 0)).any():
            raise ValueError(f'{col} must contain integers from 0 to {maximum}.')
    for col in ('IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS'):
        if (clean[col] % 1 != 0).any():
            raise ValueError(f'{col} must contain whole-number counts.')
    return clean.astype(float)


def score_netflow(bundle, clean):
    probabilities = np.asarray(bundle['model'].predict_proba(clean), dtype=float)
    if probabilities.shape != (len(clean), 2) or not np.isfinite(probabilities).all():
        raise ValueError('The model returned invalid probabilities.')
    if ((probabilities < 0) | (probabilities > 1)).any() or not np.allclose(probabilities.sum(axis=1), 1):
        raise ValueError('The model returned invalid probabilities.')
    attack = probabilities[:, 1]
    result = pd.DataFrame({
        'Attack_Probability': attack,
        'Risk_Score': attack * 10,
        'Predicted_Attack': attack >= bundle['threshold'],
    }, index=clean.index)
    if 'conformal_q' in bundle:
        sets = (1 - probabilities) <= bundle['conformal_q']
        labels = np.array(['Empty set — review', 'Benign', 'Attack', 'Benign or Attack'])
        codes = sets[:, 0].astype(int) + 2 * sets[:, 1].astype(int)
        result['Prediction_Set'] = labels[codes]
    return result


def display_traffic(raw, clean):
    """Actual NetFlow values exposed under dashboard display names only."""
    view = raw.copy()
    view['proto'] = clean['PROTOCOL'].map({1: 'icmp', 6: 'tcp', 17: 'udp'}).fillna('other')
    view['service'] = clean['L4_DST_PORT'].map(lambda port: f'Port {int(port)}' if pd.notna(port) else 'Unknown port')
    view['state'] = clean['TCP_FLAGS'].map(lambda flag: f'TCP flags {int(flag)}' if pd.notna(flag) else 'Unknown flags')
    view['duration'] = clean['FLOW_DURATION_MILLISECONDS'] / 1000
    for alias, original in [('sbytes', 'IN_BYTES'), ('dbytes', 'OUT_BYTES'), ('spkts', 'IN_PKTS'), ('dpkts', 'OUT_PKTS')]:
        view[alias] = clean[original]
    return view


def tree_explanations(bundle, clean):
    import shap
    pipeline = bundle['base_model']
    transformed = pipeline.named_steps['features'].transform(clean)
    names = pipeline.named_steps['features'].get_feature_names_out()
    values = shap.TreeExplainer(pipeline.named_steps['estimator'])(transformed)
    if values.values.ndim == 3:
        values = values[:, :, 1]
    values.feature_names = list(names)
    return values

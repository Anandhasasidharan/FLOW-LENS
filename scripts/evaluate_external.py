"""Evaluate the fixed NF-ToN-IoT model on NF-BoT-IoT without retraining."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from scripts.train_detector import metrics
from src.network_detector import FEATURES


def main():
    artifact = ROOT / 'models/experiments/nf_ton_iot/netflow_detector.joblib'
    training_report = json.loads((artifact.parent / 'training_report.json').read_text())
    if 'NF-BoT-IoT' in training_report['dataset']:
        raise ValueError('NF-BoT-IoT was part of development; it cannot be called an independent external test.')
    path = ROOT / 'data/raw/NF-BoT-IoT.csv'
    bundle = joblib.load(artifact)
    df = pd.read_csv(path)
    X = df[list(FEATURES)].apply(pd.to_numeric, errors='raise').astype(float)
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError('External dataset contains nonfinite inputs.')
    y = df.Label.to_numpy(dtype=int)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError('Expected both benign and attack labels.')
    with threadpool_limits(limits=2):
        probability = np.concatenate([bundle['model'].predict_proba(X.iloc[i:i+50_000])[:, 1] for i in range(0, len(X), 50_000)])
    result = metrics(y, probability, bundle['threshold'])
    sets = (1 - np.column_stack([1-probability, probability])) <= bundle['conformal_q']
    classes = {}
    for label in sorted(df.Attack.unique()):
        mask = df.Attack.to_numpy() == label
        classes[str(label)] = {'records': int(mask.sum()), 'correct_binary_decision_rate': float(((probability[mask] >= bundle['threshold']) == y[mask]).mean())}
    report = {
        'dataset': 'NF-BoT-IoT v1', 'records': len(df),
        'dataset_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'artifact_sha256': hashlib.sha256(artifact.read_bytes()).hexdigest(),
        'threshold': bundle['threshold'], 'retrained_or_tuned_on_external_data': False,
        'metrics': result, 'per_attack': classes,
        'conformal_coverage': float(sets[np.arange(len(df)), y].mean()),
        'cleaning': 'None; full external dataset used.',
        'limitations': [
            'Different lab dataset, features and attack prevalence; not production traffic.',
            'Scores reflect domain shift. No retraining, threshold changes, or calibrator fitting use these labels.',
            'Per-attack rates measure binary decisions, not family classification accuracy.',
        ],
    }
    destination = ROOT / 'reports/initial_external_evaluation.json'
    destination.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

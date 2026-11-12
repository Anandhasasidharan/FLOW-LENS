"""Inspect a trusted legacy forest; this is not an accuracy evaluation."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def audit_forest(model, feature_names):
    features = []
    inconsistent_binary = []
    for index, name in enumerate(feature_names):
        thresholds = np.concatenate([tree.tree_.threshold[tree.tree_.feature == index] for tree in model.estimators_])
        features.append({
            'feature': name, 'split_count': len(thresholds),
            'minimum_threshold': float(thresholds.min()) if len(thresholds) else None,
            'maximum_threshold': float(thresholds.max()) if len(thresholds) else None,
        })
        if name.startswith(('proto_', 'service_', 'state_')) and len(thresholds) and not np.allclose(thresholds, 0.5):
            inconsistent_binary.append(name)
    single = np.concatenate([
        tree.tree_.value[(tree.tree_.children_left == -1) & (tree.tree_.n_node_samples == 1)].reshape(-1)
        for tree in model.estimators_
    ])
    fraction = float(np.mean(~np.isclose(single, np.round(single)))) if len(single) else None
    return {
        'status': 'incompatible_with_documented_preprocessing' if inconsistent_binary else 'unverified',
        'accuracy_evaluated': False,
        'trees': len(model.estimators_),
        'feature_thresholds': features,
        'binary_features_with_non_midpoint_splits': inconsistent_binary,
        'single_observation_leaves': len(single),
        'noninteger_single_observation_leaf_fraction': fraction,
        'interpretation': [
            'Binary 0/1 encoded fields should split at 0.5. Other thresholds are inconsistent with the documented raw/one-hot training path.',
            'A one-observation leaf with a noninteger target is inconsistent with the documented integer severity mapping.',
            'This does not prove the origin of the model or whether an unrecorded transformation was used.',
            'Neither structural checks nor explanations measure predictive accuracy.',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('models/risk_score_model.pkl'))
    parser.add_argument('--features', type=Path, default=Path('data/feature_columns.csv'))
    parser.add_argument('--output', type=Path, default=Path('reports/legacy_model_audit.json'))
    args = parser.parse_args()
    report = audit_forest(joblib.load(args.model), pd.read_csv(args.features).feature.tolist())
    report['artifact_sha256'] = hashlib.sha256(args.model.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f"{report['status']}: saved {args.output}")


if __name__ == '__main__':
    main()

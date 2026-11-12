"""Evaluate frozen v2 models once on the previously unopened chronological lockbox."""
from datetime import datetime, timezone
import hashlib
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from scripts.train_detector import metrics
from src.interpretable_detector import native_record_explanation


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(model, features, threshold, frame):
    start = time.perf_counter()
    probability = np.concatenate([model.predict_proba(frame[features].iloc[i:i+25_000])[:, 1] for i in range(0, len(frame), 25_000)])
    y = frame.Label.to_numpy()
    result = metrics(y, probability, threshold)
    result['inference_seconds'] = time.perf_counter()-start
    per_family = {}
    for family, ids in frame.groupby('Attack').indices.items():
        truth = y[ids]
        prediction = probability[ids] >= threshold
        per_family[family] = {'records': len(ids), 'binary_correctness': float((prediction == truth).mean())}
    result['per_family'] = per_family
    attack_recalls = [v['binary_correctness'] for k,v in per_family.items() if k != 'Benign']
    result['macro_attack_recall'] = float(np.mean(attack_recalls))
    # Descriptive reliability bins, kept even when calibration is imperfect.
    bins = np.minimum((probability * 10).astype(int), 9)
    reliability = []
    for index in range(10):
        mask = bins == index
        if mask.any():
            reliability.append({'bin': index, 'records': int(mask.sum()), 'mean_probability': float(probability[mask].mean()), 'attack_fraction': float(y[mask].mean())})
    result['reliability_bins'] = reliability
    result['expected_calibration_error'] = float(sum(v['records']/len(frame)*abs(v['mean_probability']-v['attack_fraction']) for v in reliability))
    # Estimate how often the final records have patterns seen in the actual fitting data.
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir', type=Path, default=ROOT / 'models/v2')
    args = parser.parse_args()
    directory = args.experiment_dir
    destination = directory / 'final_evaluation.json'
    if destination.exists():
        raise RuntimeError('Final evaluation already exists; preserve it rather than reusing the lockbox for tuning.')
    artifact = directory / 'detector.joblib'
    report = json.loads((directory / 'development_report.json').read_text())
    if digest(artifact) != report['artifact_sha256']:
        raise ValueError('Model differs from the frozen development report.')
    plan_path = directory / 'evaluation_plan.json'
    if digest(plan_path) != report['evaluation_plan_sha256']:
        raise ValueError('Evaluation plan changed after model freezing.')
    references_path = directory / 'reference_models.joblib'
    marker = {'artifact_sha256': digest(artifact), 'references_sha256': digest(references_path), 'plan_sha256': digest(plan_path)}
    marker_path = directory / 'lockbox_started.json'
    if marker_path.exists() and json.loads(marker_path.read_text()) != marker:
        raise RuntimeError('A different model already opened this lockbox.')
    marker_path.write_text(json.dumps(marker, indent=2)+'\n')
    bundle = joblib.load(artifact)
    references = joblib.load(references_path)
    frame = pd.read_parquet(ROOT / 'data/raw/v2/prepared/lockbox.parquet')
    plan = json.loads(plan_path.read_text())
    if len(frame) != plan['partitions']['lockbox']['records'] or (frame.FLOW_START_MILLISECONDS < plan['cutoffs_ms'][-1]).any():
        raise ValueError('Final data do not match the locked temporal plan.')
    selected = evaluate(bundle['model'], bundle['features'], bundle['threshold'], frame)
    comparisons = {name: evaluate(ref['model'], ref['features'], ref['threshold'], frame) for name,ref in references.items()}
    native_validated = False
    if bundle['model_name'] == 'explainable_boosting':
        native_record_explanation(bundle, frame[bundle['features']].iloc[:100])
        native_validated = True
    expected_families = sorted(pd.read_parquet(ROOT / 'data/raw/v2/prepared/train.parquet', columns=['Attack']).Attack.unique())
    absent = sorted(set(expected_families) - set(selected['per_family']))
    checks = {
        'attack_recall_at_least_95_percent': selected['recall'] >= .95,
        'benign_false_positive_rate_at_most_1_percent': selected['false_positive_rate'] <= .01,
        'macro_attack_recall_at_least_90_percent': selected['macro_attack_recall'] >= .9,
        'every_attack_family_at_least_90_percent': all(v['binary_correctness'] >= .9 for k,v in selected['per_family'].items() if k != 'Benign'),
        'all_development_families_have_test_support': not absent,
        'native_additive_explanation_verified': native_validated,
    }
    result = {
        'evaluated_at': datetime.now(timezone.utc).isoformat(), **marker,
        'evaluation_kind': report.get('evaluation_kind', 'prospectively frozen chronological lockbox'),
        'dataset': report['dataset'], 'model_name': report['model_name'], 'probability_policy': report['probability_policy'],
        'threshold': report['threshold'], 'features': report['features'],
        'evaluation_design': plan['split'], 'test_records': len(frame),
        'selected_model': selected, 'references': comparisons,
        'quality_checks': checks, 'status': 'benchmark_targets_met' if all(checks.values()) else 'benchmark_targets_partially_met',
        'absent_families': absent,
        'limitations': [
            'This is a chronological holdout from one laboratory dataset, not a live deployment or a cross-dataset result.',
            'The original traffic dates to 2015; richer extraction and a newer model do not make the traffic contemporary.',
            'Per-family rates describe binary detection, not attack-family classification.',
            'Rare-family estimates need more independent examples; correlated flows do not provide independent trials.',
            'Feature contributions explain this fitted model and do not establish causal effects or remediation advice.',
        ],
    }
    destination.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'selected_model': selected, 'quality_checks': checks}, indent=2), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()

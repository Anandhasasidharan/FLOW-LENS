"""Freeze feature-ablation references before the v2 lockbox is evaluated."""
import gc
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import joblib
import pandas as pd
from threadpoolctl import threadpool_limits

from scripts.train_interpretable_v2 import fit_candidate, operating_point
from src.network_detector import FEATURES


def fit_reference(columns, parameters):
    # Project parquet columns rather than loading unused full-flow measurements.
    projection = list(columns) + ['Label', 'Attack']
    train = pd.read_parquet(ROOT / 'data/raw/v2/prepared/train.parquet', columns=projection)
    val = pd.read_parquet(ROOT / 'data/raw/v2/prepared/validation.parquet', columns=projection)
    model = fit_candidate('lgbm', list(columns), parameters, train, val, full=True)
    point = operating_point(val.Label.to_numpy(), model.predict_proba(val[list(columns)])[:, 1], val.Attack.to_numpy())
    result = {'model': model, 'features': list(model.named_steps['features'].retained_),
              'threshold': point['threshold'], 'validation': point,
              'fitting_records': len(train), 'derived_feature_count': 8}
    del train, val
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir', type=Path, default=ROOT / 'models/v2')
    args = parser.parse_args()
    directory = args.experiment_dir
    if (directory / 'final_evaluation.json').exists():
        raise RuntimeError('Reference models must be frozen before final evaluation.')
    report = json.loads((directory / 'development_report.json').read_text())
    plan = json.loads((directory / 'evaluation_plan.json').read_text())
    parameters = report['best_parameters']['lgbm']
    print('Fitting nine-input LightGBM reference on all chronological training records.', flush=True)
    baseline = fit_reference(FEATURES, parameters)
    print('Freezing rich-input LightGBM using the same development parameters.', flush=True)
    challenger = fit_reference(plan['features'], parameters)
    joblib.dump({'nine_input_lightgbm': baseline, 'rich_lightgbm': challenger}, directory / 'reference_models.joblib', compress=3)
    print('References frozen; no final-test records were loaded.', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()

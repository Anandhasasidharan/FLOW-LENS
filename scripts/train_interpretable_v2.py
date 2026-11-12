"""Develop an intrinsically interpretable detector without opening the final lockbox."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interpret.glassbox import ExplainableBoostingClassifier
import interpret
import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, roc_curve
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from scripts.train_detector import metrics
from src.interpretable_detector import RichFlowFeatures, native_record_explanation

SEED = 20261112
MODEL_DIRECTORY = ROOT / 'models/v2'
DATA_DIRECTORY = ROOT / 'data/raw/v2/prepared'


def stratified_screen(frame, maximum, minimum_family=100):
    if len(frame) <= maximum:
        return frame
    parts = []
    for label, group in frame.groupby('Attack', sort=True):
        count = min(len(group), max(minimum_family, int(maximum * len(group) / len(frame))))
        parts.append(group.sample(count, random_state=SEED))
    return pd.concat(parts).sort_index()


def operating_point(y, probability, families, maximum_fpr=.01):
    fpr, recall, thresholds = roc_curve(y, probability, drop_intermediate=False)
    family_recalls = []
    for family in np.unique(families[y == 1]):
        values = np.sort(probability[(y == 1) & (families == family)])
        family_recalls.append((len(values) - np.searchsorted(values, thresholds, side='left')) / len(values))
    macro = np.mean(family_recalls, axis=0)
    candidates = np.flatnonzero(np.isfinite(thresholds) & (fpr <= maximum_fpr))
    if not len(candidates):
        return {'threshold': float(np.nextafter(1., 2.)), 'macro_attack_recall': 0., 'false_positive_rate': 0., 'attack_recall': 0.}
    best = max(candidates, key=lambda i: (macro[i], recall[i], -fpr[i]))
    return {'threshold': float(thresholds[best]), 'macro_attack_recall': float(macro[best]), 'false_positive_rate': float(fpr[best]), 'attack_recall': float(recall[best])}


def sample_weights(frame, exponent):
    if exponent == 0:
        return None
    sizes = frame.Attack.value_counts()
    attack_sizes = sizes.drop('Benign', errors='ignore')
    reference = float(attack_sizes.max())
    weight = frame.Attack.map({label: min(20., (reference / count) ** exponent) if label != 'Benign' else 1. for label, count in sizes.items()}).to_numpy()
    return weight / weight.mean()


def make_model(kind, columns, parameters):
    parameters = dict(parameters)
    exclude_ttl = parameters.pop('exclude_ttl', False)
    parameters.pop('weight_exponent', None)
    if kind == 'ebm':
        estimator = ExplainableBoostingClassifier(
            max_bins=128, max_interaction_bins=32, outer_bags=2,
            early_stopping_rounds=50, smoothing_rounds=50,
            interaction_smoothing_rounds=50, n_jobs=4, random_state=SEED,
            **parameters,
        )
    else:
        estimator = lgb.LGBMClassifier(n_jobs=4, verbosity=-1, random_state=SEED, **parameters)
    return Pipeline([('features', RichFlowFeatures(columns, exclude_ttl=exclude_ttl)), ('estimator', estimator)])


def fit_candidate(kind, columns, parameters, train, validation, full=False):
    pipeline = make_model(kind, columns, parameters)
    X = train[columns]
    y = train.Label.to_numpy()
    weight = sample_weights(train, parameters.get('weight_exponent', 0))
    # Transform inside the fitted pipeline; no fitted preprocessing comes from future partitions.
    if kind == 'lgbm':
        builder = pipeline.named_steps['features']
        transformed = builder.fit_transform(X, y)
        val = builder.transform(validation[columns])
        pipeline.named_steps['estimator'].fit(
            transformed, y, sample_weight=weight, eval_set=[(val, validation.Label)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
    else:
        pipeline.fit(X, y, estimator__sample_weight=weight)
    return pipeline


def inspect_validation(pipeline, validation, columns):
    y = validation.Label.to_numpy()
    p = pipeline.predict_proba(validation[columns])[:, 1]
    point = operating_point(y, p, validation.Attack.to_numpy())
    result = metrics(y, p, point['threshold'])
    result.update(point)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=MODEL_DIRECTORY)
    parser.add_argument('--ebm-trials', type=int, default=4)
    parser.add_argument('--lgbm-trials', type=int, default=6)
    args = parser.parse_args()
    directory = args.output
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'final_evaluation.json').exists():
        raise RuntimeError('Scored release retained. Use --output models/experiments/v2-reproduction to replay this public benchmark without overwriting the active release.')
    plan_path = directory / 'evaluation_plan.json'
    replay = directory.resolve() != MODEL_DIRECTORY.resolve() and (MODEL_DIRECTORY / 'final_evaluation.json').exists()
    if not plan_path.exists():
        plan_path.write_bytes((MODEL_DIRECTORY / 'evaluation_plan.json').read_bytes())
    plan = json.loads(plan_path.read_text())
    columns = plan['features']
    train = pd.read_parquet(DATA_DIRECTORY / 'train.parquet')
    validation = pd.read_parquet(DATA_DIRECTORY / 'validation.parquet')
    calibration = pd.read_parquet(DATA_DIRECTORY / 'calibration.parquet')
    for role, data in [('train', train), ('validation', validation), ('calibration', calibration)]:
        if set(data.Label.unique()) != {0, 1}:
            raise ValueError(f'{role} requires both labels.')
    screen_train = stratified_screen(train, 140_000)
    screen_val = stratified_screen(validation, 100_000)
    records = []
    best_parameters = {}
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(kind):
        def run(trial):
            if kind == 'ebm':
                parameters = {
                    'interactions': trial.suggest_categorical('interactions', [0, 8, 16]),
                    'learning_rate': trial.suggest_float('learning_rate', .03, .08),
                    'max_leaves': trial.suggest_int('max_leaves', 2, 3),
                    'reg_lambda': trial.suggest_float('reg_lambda', .1, 5, log=True),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 30),
                    'max_rounds': 1200,
                    'weight_exponent': trial.suggest_categorical('weight_exponent', [0., .35]),
                    'exclude_ttl': trial.suggest_categorical('exclude_ttl', [True, False]),
                }
            else:
                parameters = {
                    'n_estimators': 1200,
                    'num_leaves': trial.suggest_int('num_leaves', 15, 63),
                    'learning_rate': trial.suggest_float('learning_rate', .025, .1),
                    'min_child_samples': trial.suggest_int('min_child_samples', 20, 150),
                    'reg_lambda': trial.suggest_float('reg_lambda', .1, 10, log=True),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', .7, 1.),
                    'weight_exponent': trial.suggest_categorical('weight_exponent', [0., .35]),
                    'exclude_ttl': trial.suggest_categorical('exclude_ttl', [True, False]),
                }
            started = time.perf_counter()
            pipeline = fit_candidate(kind, columns, parameters, screen_train, screen_val)
            result = inspect_validation(pipeline, screen_val, columns)
            result.update({'kind': kind, 'trial': trial.number, 'parameters': parameters, 'fit_seconds': time.perf_counter()-started})
            records.append(result)
            (directory / 'development_trials.json').write_text(json.dumps(records, indent=2)+'\n')
            trial.set_user_attr('parameters', parameters)
            print(f"{kind} trial {trial.number}: AP={result['average_precision']:.5f}, macro recall @1% FPR={result['macro_attack_recall']:.5f}, {result['fit_seconds']:.1f}s", flush=True)
            return result['macro_attack_recall'] + .001 * result['average_precision']
        return run

    for kind, count in [('lgbm', args.lgbm_trials), ('ebm', args.ebm_trials)]:
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
        # Include an explicitly TTL-free candidate to test dependence on lab-specific TTL signatures.
        if kind == 'ebm':
            study.enqueue_trial({'interactions': 8, 'learning_rate': .05, 'max_leaves': 3, 'reg_lambda': 1., 'min_samples_leaf': 10, 'weight_exponent': .35, 'exclude_ttl': True})
            study.enqueue_trial({'interactions': 16, 'learning_rate': .05, 'max_leaves': 3, 'reg_lambda': 1., 'min_samples_leaf': 10, 'weight_exponent': 0., 'exclude_ttl': False})
        study.optimize(objective(kind), n_trials=count)
        best_parameters[kind] = study.best_trial.user_attrs['parameters']
    # Refit finalists on every eligible training row; screening is not a cap on final fitting.
    pipelines = {}
    full_validation = {}
    for kind, parameters in best_parameters.items():
        print(f'Full-data fit: {kind} on {len(train):,} records', flush=True)
        started = time.perf_counter()
        pipelines[kind] = fit_candidate(kind, columns, parameters, train, validation, full=True)
        full_validation[kind] = inspect_validation(pipelines[kind], validation, columns)
        full_validation[kind]['fit_seconds'] = time.perf_counter() - started
        print(f"Full {kind}: {full_validation[kind]}", flush=True)
    strongest = max(full_validation, key=lambda k: (full_validation[k]['macro_attack_recall'], full_validation[k]['average_precision']))
    ebm = full_validation['ebm']
    champion = full_validation[strongest]
    selected = 'ebm' if ebm['average_precision'] >= champion['average_precision']-.01 and ebm['macro_attack_recall'] >= champion['macro_attack_recall']-.01 else strongest
    # Preserve the independently interpretable EBM even if the predictive challenger wins.
    policies = {}
    fitted = {}
    for kind, pipeline in pipelines.items():
        for method in ['raw', 'sigmoid', 'isotonic']:
            model = pipeline
            if method != 'raw':
                model = CalibratedClassifierCV(FrozenEstimator(pipeline), method=method)
                model.fit(calibration[columns], calibration.Label)
            probability = model.predict_proba(validation[columns])[:, 1]
            point = operating_point(validation.Label.to_numpy(), probability, validation.Attack.to_numpy())
            result = metrics(validation.Label.to_numpy(), probability, point['threshold'])
            result.update(point)
            policies[kind+'/'+method] = result
            fitted[kind+'/'+method] = model
    # Calibration should improve reliability without sacrificing validated discrimination.
    available = [key for key in policies if key.startswith(selected+'/') and policies[key]['macro_attack_recall'] >= full_validation[selected]['macro_attack_recall']-.005]
    key = min(available, key=lambda k: (policies[k]['brier'], -policies[k]['average_precision']))
    pipeline = pipelines[selected]
    raw_features = list(pipeline.named_steps['features'].retained_)
    bundle = {
        'schema_version': 2, 'model': fitted[key], 'base_model': pipeline,
        'features': raw_features, 'model_name': 'explainable_boosting' if selected == 'ebm' else 'lightgbm',
        'probability_policy': key.split('/')[1], 'threshold': policies[key]['threshold'],
        'interpretability': 'native additive terms' if selected == 'ebm' else 'native TreeSHAP contributions',
    }
    artifact = directory / 'detector.joblib'
    joblib.dump(bundle, artifact, compress=3)
    joblib.dump({'model': pipelines['ebm'], 'features': columns}, directory / 'ebm_companion.joblib', compress=3)
    # Validate explanation reconstruction on development records before publication.
    if selected == 'ebm':
        native_record_explanation(bundle, validation[raw_features].iloc[:20])
    else:
        transformed = pipeline.named_steps['features'].transform(validation[raw_features].iloc[:20])
        terms = pipeline.named_steps['estimator'].predict(transformed, pred_contrib=True)
        assert np.allclose(1/(1+np.exp(-terms.sum(axis=1))), pipeline.predict_proba(validation[raw_features].iloc[:20])[:, 1])
    report = {
        'created_at': datetime.now(timezone.utc).isoformat(), 'dataset': plan['dataset'],
        'evaluation_kind': 'reproduction of previously evaluated public split' if replay else 'prospectively frozen chronological lockbox',
        'model_name': bundle['model_name'], 'probability_policy': bundle['probability_policy'],
        'threshold': bundle['threshold'], 'features': raw_features,
        'engineered_feature_count': len(pipeline.named_steps['features'].get_feature_names_out()),
        'training_records': len(train), 'screening_records': len(screen_train),
        'seed': SEED, 'best_parameters': best_parameters,
        'finalist_validation': full_validation, 'probability_policy_validation': policies,
        'selection': plan['selection'], 'interpretability': bundle['interpretability'],
        'artifact_sha256': hashlib.sha256(artifact.read_bytes()).hexdigest(),
        'evaluation_plan_sha256': hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        'versions': {'python': platform.python_version(), 'interpret': interpret.__version__, 'lightgbm': lgb.__version__, 'optuna': optuna.__version__},
        'lockbox_evaluated': False,
    }
    (directory / 'development_report.json').write_text(json.dumps(report, indent=2)+'\n')
    # Demonstration records come only from development data, never from the lockbox.
    examples = validation.groupby('Attack', sort=True).head(1).head(10)
    sample = examples[raw_features].replace([np.inf, -np.inf], np.nan)
    sample.to_csv(directory / 'sample.csv', index=False)
    if directory.resolve() == MODEL_DIRECTORY.resolve():
        sample.to_csv(ROOT / 'data/rich_flow_sample.csv', index=False)
    (ROOT / 'data/netflow_v3_features.csv').write_bytes((ROOT / 'data/raw/v2/NetFlow_v3_Features.csv').read_bytes())
    print(f"Frozen v2 model: {key}; {bundle['interpretability']}. Final lockbox remains unopened.", flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()

"""Train and benchmark a binary NetFlow detector with untouched host-pair holdout."""
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

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, log_loss, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from src.network_detector import FEATURES, CATEGORICAL

SEED = 42


def make_pipeline(estimator, scale=False):
    numerical = [f for f in FEATURES if f not in CATEGORICAL]
    transformer = ColumnTransformer([
        ('numeric', StandardScaler() if scale else 'passthrough', numerical),
        ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), list(CATEGORICAL)),
    ])
    return Pipeline([('features', transformer), ('estimator', estimator)])


def grouped_partitions(groups):
    indexes = np.arange(len(groups))
    def split(ids, fraction, seed):
        left, right = next(GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed).split(ids, groups=groups[ids]))
        return ids[left], ids[right]
    rest, test = split(indexes, .2, SEED)
    rest, conformal = split(rest, .125, SEED + 1)
    rest, calibration = split(rest, 1 / 7, SEED + 2)
    train, validation = split(rest, .2, SEED + 3)
    return dict(train=train, validation=validation, calibration=calibration, conformal=conformal, test=test)


def metrics(y, probability, threshold):
    predicted = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    return {
        'records': len(y), 'attack_prevalence': float(np.mean(y)),
        'average_precision': float(average_precision_score(y, probability)) if both else None,
        'roc_auc': float(roc_auc_score(y, probability)) if both else None,
        'precision': float(precision_score(y, predicted, zero_division=0)),
        'recall': float(recall_score(y, predicted, zero_division=0)),
        'f1': float(f1_score(y, predicted, zero_division=0)),
        'false_positive_rate': float(fp / (fp + tn)) if fp + tn else None,
        'brier': float(brier_score_loss(y, probability)),
        'log_loss': float(log_loss(y, probability, labels=[0, 1])),
        'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def choose_threshold(y, probability, max_fpr=.05, sources=None):
    fpr, recall, thresholds = roc_curve(y, probability, drop_intermediate=False)
    allowed = np.isfinite(thresholds) & (fpr <= max_fpr)
    if sources is not None:
        for source in np.unique(sources):
            negatives = np.sort(probability[(y == 0) & (sources == source)])
            if len(negatives):
                source_fpr = (len(negatives) - np.searchsorted(negatives, thresholds, side='left')) / len(negatives)
                allowed &= source_fpr <= max_fpr
    candidates = np.flatnonzero(allowed)
    if not len(candidates):
        return float(np.nextafter(1., 2.))
    best = max(candidates, key=lambda i: (recall[i], -fpr[i], thresholds[i]))
    return float(thresholds[best])


def conformal_quantile(scores, alpha=.1):
    n = len(scores)
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    if rank > n:
        return 1.0
    return float(np.sort(scores)[rank - 1])


def group_bootstrap_intervals(y, prediction, groups, repetitions=500):
    counts = pd.DataFrame({
        'group': groups, 'tn': (y == 0) & ~prediction, 'fp': (y == 0) & prediction,
        'fn': (y == 1) & ~prediction, 'tp': (y == 1) & prediction,
    }).groupby('group')[['tn', 'fp', 'fn', 'tp']].sum().to_numpy()
    rng = np.random.default_rng(SEED)
    results = {'recall': [], 'false_positive_rate': [], 'precision': []}
    for _ in range(repetitions):
        tn, fp, fn, tp = counts[rng.integers(0, len(counts), len(counts))].sum(axis=0)
        for name, numerator, denominator in [('recall', tp, tp+fn), ('false_positive_rate', fp, fp+tn), ('precision', tp, tp+fp)]:
            if denominator:
                results[name].append(numerator / denominator)
    return {name: [float(x) for x in np.quantile(values, [.025, .975])] for name, values in results.items()}


def prepare_data(path, additional_path=None):
    primary = pd.read_csv(path)
    primary['_Dataset'] = 'NF-ToN-IoT'
    if additional_path is not None:
        secondary = pd.read_csv(additional_path)
        secondary['_Dataset'] = 'NF-BoT-IoT'
        df = pd.concat([primary, secondary], ignore_index=True)
    else:
        df = primary
    required = set(FEATURES) | {'Label', 'Attack', 'IPV4_SRC_ADDR', 'IPV4_DST_ADDR'}
    if not required.issubset(df.columns):
        raise ValueError('Dataset is missing required fields.')
    n_original = len(df)
    X = df[list(FEATURES)].apply(pd.to_numeric, errors='coerce')
    valid = np.isfinite(X.to_numpy()).all(axis=1) & (X >= 0).all(axis=1) & df.Label.isin([0, 1])
    invalid_count = int((~valid).sum())
    df, X = df.loc[valid].copy(), X.loc[valid]
    y = df.Label.astype(int)
    # Keep conflicting/repeated patterns in holdout data; do not select test rows using their labels.
    df, X = df.reset_index(drop=True), X.reset_index(drop=True).astype(float)
    src, dst = df.IPV4_SRC_ADDR.astype(str), df.IPV4_DST_ADDR.astype(str)
    pair = df['_Dataset'] + '|' + pd.Series(np.where(src < dst, src + '|' + dst, dst + '|' + src))
    groups = pd.factorize(pair)[0]
    return df, X, groups, {
        'original_records': n_original, 'invalid_records_removed': invalid_count,
        'ambiguous_feature_records_removed': 0,
        'duplicate_feature_records_removed': 0,
        'retained_records': len(df), 'host_pairs': int(len(np.unique(groups))),
        'excluded_inputs': ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'L4_SRC_PORT', 'Label', 'Attack'],
        'caveat': 'No label-based filtering or deduplication is applied to holdouts. Only fitting data are deduplicated. Natural repeated feature patterns may exist across distinct host pairs; novelty metrics are reported separately.',
    }


def train(path, output, max_train_rows, additional_path=None):
    output.mkdir(parents=True, exist_ok=True)
    df, X, groups, cleaning = prepare_data(path, additional_path)
    y = df.Label.to_numpy(dtype=int)
    partitions = grouped_partitions(groups)
    for role, ids in partitions.items():
        if len(np.unique(y[ids])) != 2:
            raise ValueError(f'{role} split lacks benign or attack labels; a meaningful evaluation cannot proceed.')
    train_ids = partitions['train']
    fit_unique = ~pd.concat([X.iloc[train_ids].reset_index(drop=True), pd.Series(y[train_ids], name='Label')], axis=1).duplicated()
    fit_duplicates_removed = int((~fit_unique).sum())
    train_ids = train_ids[fit_unique.to_numpy()]
    if len(train_ids) > max_train_rows:
        train_ids, _ = train_test_split(train_ids, train_size=max_train_rows, stratify=y[train_ids], random_state=SEED)
    val = partitions['validation']
    candidates = {
        'dummy_prior': make_pipeline(DummyClassifier(strategy='prior')),
        'logistic_regression': make_pipeline(LogisticRegression(max_iter=600, random_state=SEED), scale=True),
        'extra_trees': make_pipeline(ExtraTreesClassifier(n_estimators=120, max_depth=24, min_samples_leaf=2, n_jobs=2, random_state=SEED)),
        'hist_gradient_boosting': make_pipeline(HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=31, l2_regularization=1, early_stopping=False, random_state=SEED)),
    }
    comparisons = {}
    policies = {}
    fitted_policies = {}
    calibration = partitions['calibration']
    validation_sources = df.iloc[val]['_Dataset'].to_numpy()
    for name, pipeline in candidates.items():
        started = time.perf_counter()
        pipeline.fit(X.iloc[train_ids], y[train_ids])
        probability = pipeline.predict_proba(X.iloc[val])[:, 1]
        comparisons[name] = metrics(y[val], probability, .5)
        comparisons[name]['fit_seconds'] = time.perf_counter() - started
        print(f"{name}: validation AP={comparisons[name]['average_precision']:.5f}", flush=True)
        if name == 'dummy_prior':
            continue
        for method in ['raw', 'sigmoid', 'isotonic']:
            policy = pipeline
            if method != 'raw':
                policy = CalibratedClassifierCV(FrozenEstimator(pipeline), method=method)
                policy.fit(X.iloc[calibration], y[calibration])
            p = policy.predict_proba(X.iloc[val])[:, 1]
            threshold = choose_threshold(y[val], p, sources=validation_sources)
            key = name + '/' + method
            policies[key] = metrics(y[val], p, threshold)
            policies[key]['threshold'] = threshold
            policies[key]['source_false_positive_rates'] = {
                source: float((p[(y[val] == 0) & (validation_sources == source)] >= threshold).mean())
                for source in np.unique(validation_sources) if ((y[val] == 0) & (validation_sources == source)).any()
            }
            fitted_policies[key] = policy
    selected_policy = max(policies, key=lambda key: (policies[key]['recall'], policies[key]['average_precision'], -policies[key]['brier']))
    winner, probability_policy = selected_policy.split('/')
    base_model = candidates[winner]
    calibrated = fitted_policies[selected_policy]
    threshold = policies[selected_policy]['threshold']
    conformal_ids = partitions['conformal']
    conformal_probability = calibrated.predict_proba(X.iloc[conformal_ids])
    nonconformity = 1 - conformal_probability[np.arange(len(conformal_ids)), y[conformal_ids]]
    q = conformal_quantile(nonconformity)
    bundle = {'model': calibrated, 'base_model': base_model, 'threshold': threshold,
              'features': list(FEATURES), 'conformal_q': q, 'conformal_alpha': .1, 'model_name': winner, 'probability_policy': probability_policy}
    test = partitions['test']
    started = time.perf_counter()
    probability = calibrated.predict_proba(X.iloc[test])[:, 1]
    seconds = time.perf_counter() - started
    test_metrics = metrics(y[test], probability, threshold)
    raw_probability = base_model.predict_proba(X.iloc[test])[:, 1]
    test_metrics['uncalibrated_brier'] = float(brier_score_loss(y[test], raw_probability))
    test_metrics['inference_seconds'] = seconds
    test_metrics['records_per_second'] = len(test) / seconds
    sets = (1 - np.column_stack([1 - probability, probability])) <= q
    coverage = float(sets[np.arange(len(test)), y[test]].mean())
    per_attack = {}
    attack_names = df['_Dataset'] + ' / ' + df['Attack']
    for label in sorted(attack_names.unique()):
        mask = attack_names.iloc[test].to_numpy() == label
        per_attack[str(label)] = {'records': int(mask.sum()), 'correct_binary_decision_rate': float(((probability[mask] >= threshold) == y[test][mask]).mean()) if mask.any() else None}
    audit = {}
    for role, ids in partitions.items():
        audit[role] = {'records': len(ids), 'host_pairs': int(len(np.unique(groups[ids]))),
                       'benign': int((y[ids] == 0).sum()), 'attack': int((y[ids] == 1).sum())}
    host_overlap = any(set(groups[a]) & set(groups[b]) for i, a in enumerate(partitions.values()) for b in list(partitions.values())[i + 1:])
    assert not host_overlap
    test_sample = X.iloc[test].sample(min(3000, len(test)), random_state=SEED)
    sample_y = y[test][X.iloc[test].index.get_indexer(test_sample.index)]
    importance = permutation_importance(calibrated, test_sample, sample_y, scoring='average_precision', n_repeats=3, random_state=SEED, n_jobs=1)
    development = np.concatenate([ids for name, ids in partitions.items() if name != 'test'])
    known_patterns = set(pd.util.hash_pandas_object(X.iloc[development], index=False))
    novel = ~pd.util.hash_pandas_object(X.iloc[test], index=False).isin(known_patterns).to_numpy()
    novelty = {'records': int(novel.sum()), 'fraction_of_test': float(novel.mean()), 'metrics': metrics(y[test][novel], probability[novel], threshold) if novel.any() else None}
    bootstrap = group_bootstrap_intervals(y[test], probability >= threshold, groups[test])
    checks = {
        'test_average_precision_above_prevalence': test_metrics['average_precision'] > test_metrics['attack_prevalence'],
        'test_recall_at_least_90_percent': test_metrics['recall'] >= .9,
        'test_false_positive_rate_at_most_5_percent': test_metrics['false_positive_rate'] <= .05,
        'host_pairs_disjoint': not host_overlap,
        'every_source_test_false_positive_rate_at_most_5_percent': all(metrics(y[test][df.iloc[test]['_Dataset'].to_numpy() == source], probability[df.iloc[test]['_Dataset'].to_numpy() == source], threshold)['false_positive_rate'] <= .05 for source in df.iloc[test]['_Dataset'].unique()),
        'every_attack_family_has_at_least_30_test_records': all(v['records'] >= 30 for k, v in per_attack.items() if not k.endswith(' / Benign')),
        'every_supported_attack_recall_at_least_90_percent': all(v['correct_binary_decision_rate'] >= .9 for k, v in per_attack.items() if not k.endswith(' / Benign') and v['records'] > 0),
    }
    report = {
        'created_at': datetime.now(timezone.utc).isoformat(), 'dataset': 'NF-ToN-IoT v1 + NF-BoT-IoT v1' if additional_path else 'NF-ToN-IoT v1',
        'dataset_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest(), **({additional_path.name: hashlib.sha256(additional_path.read_bytes()).hexdigest()} if additional_path else {})},
        'python': platform.python_version(), 'scikit_learn': sklearn.__version__,
        'numpy': np.__version__, 'pandas': pd.__version__, 'seed': SEED,
        'features': list(FEATURES), 'cleaning': cleaning,
        'split_method': 'Disjoint (dataset source, unordered IPv4 host pair) groups; holdouts retain all valid records, including repeated and ambiguous feature patterns. Only fitting records are deduplicated on (inputs, binary target).',
        'split_counts': audit, 'fitted_records': len(train_ids), 'fit_duplicates_removed': fit_duplicates_removed,
        'validation_comparison': comparisons, 'validation_policies': policies, 'probability_policy': probability_policy, 'selected_model': winner,
        'selection_criterion': 'Maximum validation attack recall subject to each source benign FPR <=5%; ties broken by validation average precision then Brier score. Raw/sigmoid/isotonic policies compared; test data excluded from selection.',
        'threshold': threshold, 'threshold_policy': 'Maximum validation recall with benign false-positive rate <=5% for each represented source; engineering target, not a validated operational SLA.',
        'test': test_metrics, 'novel_pattern_test': novelty, 'group_bootstrap_95_intervals': bootstrap, 'per_attack_test': per_attack,
        'per_dataset_test': {name: metrics(y[test][df.iloc[test]['_Dataset'].to_numpy() == name], probability[df.iloc[test]['_Dataset'].to_numpy() == name], threshold) for name in sorted(df['_Dataset'].unique()) if (df.iloc[test]['_Dataset'] == name).any()},
        'test_dummy': metrics(y[test], candidates['dummy_prior'].predict_proba(X.iloc[test])[:, 1], .5),
        'conformal': {'alpha': .1, 'q': q, 'test_coverage': coverage,
                      'test_mean_set_size': float(sets.sum(axis=1).mean()),
                      'test_ambiguous_fraction': float((sets.sum(axis=1) != 1).mean()),
                      'caveat': 'Empirical row-level coverage only. Correlated flows and distribution shift violate simple exchangeability assumptions; no deployment coverage guarantee.'},
        'permutation_importance': [{'feature': name, 'mean_ap_decrease': float(mean), 'std': float(std)} for name, mean, std in zip(FEATURES, importance.importances_mean, importance.importances_std)],
        'engineering_checks': checks,
        'status': 'benchmark_targets_met' if all(checks.values()) else 'benchmark_targets_not_met',
        'experimental_history': 'Initial single-source training failed an external NF-BoT-IoT challenge (7.87% recall). Combined-source training followed; sources had already been inspected. Full-raw-holdout design replaces the earlier cleaned-subset evaluation.',
        'limitations': [
            'Lab dataset with attack prevalence unlike many real networks; no real-world performance guarantee.',
            'This host-pair test partition was inspected across pipeline revisions. Results are exploratory and need confirmation on a fresh locked test source or capture.',
            'The combined-data experiment was designed after a failed NF-BoT-IoT external challenge. Its dataset sources were previously inspected; this is not a pristine external validation.',
            'Host pairs are disjoint, but individual hosts can occur in multiple splits; no claim of unseen-host, temporal, or cross-dataset generalization.',
            'Holdouts retain ambiguous and repeated patterns. Training-only deduplication changes fitting weights; probability calibration uses a separate raw holdout.',
            'Binary labels are ground truth; risk score is 10 times estimated attack probability, not attack severity.',
            'Raw, sigmoid and isotonic probability policies were compared on validation data. Their probability reliability and decision thresholds may fail under distribution shift.',
            'Per-attack detection rates are binary decisions, not attack-family classification accuracy.',
        ],
    }
    artifact = output / 'netflow_detector.joblib'
    joblib.dump(bundle, artifact, compress=3)
    report['artifact_sha256'] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (output / 'training_report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    # Only real holdout records are used for the downloadable demonstration input.
    examples = []
    for label in [0, 1]:
        subset = df.iloc[test].loc[df.iloc[test].Label == label]
        diverse = subset.drop_duplicates('PROTOCOL').head(3)
        rest = subset.drop(index=diverse.index).sample(n=4-len(diverse), random_state=SEED)
        examples.append(pd.concat([diverse, rest]))
    pd.concat(examples)[list(FEATURES)].to_csv(ROOT / 'data/netflow_sample.csv', index=False)
    print(f"Selected {selected_policy}; test AP={test_metrics['average_precision']:.5f}, recall={test_metrics['recall']:.5f}, FPR={test_metrics['false_positive_rate']:.5f}; {report['status']}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/raw/NF-ToN-IoT.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'models')
    parser.add_argument('--max-train-rows', type=int, default=250_000)
    parser.add_argument('--include-bot', action='store_true', help='Include NF-BoT-IoT in a combined-source experiment; external independence is then lost.')
    args = parser.parse_args()
    if args.max_train_rows < 100:
        parser.error('--max-train-rows must be at least 100')
    with threadpool_limits(limits=2):
        train(args.data, args.output, args.max_train_rows, ROOT / 'data/raw/NF-BoT-IoT.csv' if args.include_bot else None)


if __name__ == '__main__':
    main()

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from threadpoolctl import threadpool_limits

from scripts.train_detector import choose_threshold, conformal_quantile, grouped_partitions
from src.model_audit import audit_forest
from src.network_detector import FEATURES, validate_netflow, score_netflow, display_traffic

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def netflow():
    return pd.read_csv(ROOT / 'data/netflow_sample.csv')


def test_artifact_matches_evidence():
    path = ROOT / 'models/netflow_detector.joblib'
    report = json.loads((ROOT / 'models/training_report.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == report['artifact_sha256']
    assert report['features'] == list(FEATURES)
    assert report['test']['records'] > 1000
    matrix = np.asarray(report['test']['confusion_matrix'])
    assert matrix.sum() == report['test']['records']
    assert report['engineering_checks']['host_pairs_disjoint']


def test_predictions_match_saved_pipeline_and_are_batch_independent(netflow):
    bundle = joblib.load(ROOT / 'models/netflow_detector.joblib')
    clean = validate_netflow(netflow)
    with threadpool_limits(limits=2):
        expected = bundle['model'].predict_proba(clean)[:, 1]
        scored = score_netflow(bundle, clean)
        one = score_netflow(bundle, clean.iloc[:1])
    np.testing.assert_allclose(scored.Attack_Probability, expected)
    np.testing.assert_allclose(scored.Risk_Score, expected * 10)
    pd.testing.assert_frame_equal(scored.iloc[:1], one)
    assert scored.Predicted_Attack.tolist() == (expected >= bundle['threshold']).tolist()


@pytest.mark.parametrize('change', ['missing', 'nan', 'infinite', 'negative', 'bad_port', 'bad_protocol', 'bad_flags', 'fractional_packets'])
def test_netflow_rejects_invalid_measurements(netflow, change):
    if change == 'missing':
        netflow = netflow.drop(columns='IN_BYTES')
    else:
        column, value = {
            'nan': ('IN_BYTES', np.nan), 'infinite': ('IN_BYTES', np.inf),
            'negative': ('OUT_BYTES', -1), 'bad_port': ('L4_DST_PORT', 65536),
            'bad_protocol': ('PROTOCOL', 6.5), 'bad_flags': ('TCP_FLAGS', 256),
            'fractional_packets': ('IN_PKTS', 1.5),
        }[change]
        netflow[column] = netflow[column].astype(float)
        netflow.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_netflow(netflow)


def test_labels_and_identifiers_cannot_enter_model(netflow):
    encoded = validate_netflow(netflow.assign(Label=1, Attack='attack', IPV4_SRC_ADDR='1.2.3.4', L4_SRC_PORT=12345))
    assert encoded.columns.tolist() == list(FEATURES)
    pd.testing.assert_frame_equal(encoded, validate_netflow(netflow))
    original = netflow.copy()
    view = display_traffic(netflow, encoded)
    np.testing.assert_allclose(view.duration, encoded.FLOW_DURATION_MILLISECONDS / 1000)
    pd.testing.assert_frame_equal(netflow, original)


def test_prediction_set_membership():
    class Model:
        def predict_proba(self, data):
            return np.array([[.9, .1], [.1, .9], [.5, .5]])
    clean = pd.DataFrame(index=[1, 2, 3])
    result = score_netflow({'model': Model(), 'threshold': .5, 'conformal_q': .2}, clean)
    assert result.Prediction_Set.tolist() == ['Benign', 'Attack', 'Empty set — review']
    result = score_netflow({'model': Model(), 'threshold': .5, 'conformal_q': .5}, clean)
    assert result.Prediction_Set.iloc[-1] == 'Benign or Attack'


def test_split_groups_never_cross_partitions():
    groups = np.repeat(np.arange(100), 3)
    parts = grouped_partitions(groups)
    assert sorted(np.concatenate(list(parts.values()))) == list(range(300))
    sets = [set(groups[ids]) for ids in parts.values()]
    assert all(not a.intersection(b) for i, a in enumerate(sets) for b in sets[i+1:])


def test_threshold_controls_validation_false_positives():
    truth = np.array([0, 0, 0, 1, 1])
    probability = np.array([.05, .15, .2, .25, .9])
    threshold = choose_threshold(truth, probability)
    assert not (probability[truth == 0] >= threshold).any()
    assert (probability[truth == 1] >= threshold).all()


def test_conformal_finite_sample_rank():
    assert conformal_quantile(np.array([.1, .2, .3]), alpha=.5) == .2
    assert conformal_quantile(np.array([.1]), alpha=.1) == 1


def test_audit_detects_incompatible_binary_training():
    model = RandomForestRegressor(n_estimators=2, random_state=42).fit([[.1], [.2], [.8], [.9]], [.1, .2, .8, .9])
    report = audit_forest(model, ['proto_tcp'])
    assert report['status'] == 'incompatible_with_documented_preprocessing'
    assert report['binary_features_with_non_midpoint_splits'] == ['proto_tcp']
    assert not report['accuracy_evaluated']


def test_quality_gate_exposes_source_failures():
    report = json.loads((ROOT / 'models/training_report.json').read_text())
    expected = all(source['false_positive_rate'] <= .05 for source in report['per_dataset_test'].values())
    assert report['engineering_checks']['every_source_test_false_positive_rate_at_most_5_percent'] == expected
    assert report['status'] == ('benchmark_targets_met' if all(report['engineering_checks'].values()) else 'benchmark_targets_not_met')

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.pipeline import Pipeline

from src.interpretable_detector import RichFlowFeatures, validate_rich_flow, native_record_explanation, native_effect


def toy_flows():
    return pd.DataFrame({
        'PROTOCOL': [6., 17., 6., 17.] * 10,
        'IN_BYTES': [10., 30., 100., 300.] * 10,
        'OUT_BYTES': [5., 20., 10., 100.] * 10,
        'IN_PKTS': [1., 2., 5., 10.] * 10,
        'OUT_PKTS': [1., 2., 1., 5.] * 10,
        'FLOW_DURATION_MILLISECONDS': [5., 10., 20., 50.] * 10,
        'MIN_TTL': [30., 40., 50., 60.] * 10,
    })


def test_rich_features_are_fitted_and_row_local():
    raw = toy_flows()
    builder = RichFlowFeatures(raw.columns.tolist(), exclude_ttl=True).fit(raw)
    transformed = builder.transform(raw)
    assert 'MIN_TTL' not in transformed
    assert transformed.TOTAL_BYTES.iloc[0] == 15
    assert transformed.TOTAL_PACKETS.iloc[0] == 2
    pd.testing.assert_frame_equal(transformed.iloc[:1], builder.transform(raw.drop(columns='MIN_TTL').iloc[:1]))
    unknown = raw.iloc[:1].copy()
    unknown['PROTOCOL'] = 99
    assert builder.transform(unknown).PROTOCOL.isna().all()


def test_missing_native_inputs_are_explicit():
    raw = toy_flows()
    columns = raw.columns.tolist()
    raw.loc[0, 'IN_BYTES'] = np.nan
    assert np.isnan(validate_rich_flow(raw, columns).IN_BYTES.iloc[0])
    with pytest.raises(ValueError, match='Missing required'):
        validate_rich_flow(raw.drop(columns='PROTOCOL'), columns)
    with pytest.raises(ValueError, match='Infinite'):
        validate_rich_flow(raw.assign(IN_BYTES=np.inf), columns)
    with pytest.raises(ValueError, match='numbers'):
        validate_rich_flow(raw.assign(IN_BYTES='bad'), columns)


def test_derived_values_cannot_be_overridden_and_labels_are_excluded():
    raw = toy_flows()
    builder = RichFlowFeatures(raw.columns.tolist()).fit(raw)
    transformed = builder.transform(raw.assign(TOTAL_BYTES=999, Label=1, Attack='Worms'))
    assert transformed.TOTAL_BYTES.iloc[0] == 15
    assert not {'Label', 'Attack'}.intersection(transformed.columns)


def test_native_terms_reconstruct_prediction():
    interpret = pytest.importorskip('interpret')
    from interpret.glassbox import ExplainableBoostingClassifier
    raw = toy_flows()
    target = np.array([0, 0, 1, 1] * 10)
    pipeline = Pipeline([
        ('features', RichFlowFeatures(raw.columns.tolist())),
        ('estimator', ExplainableBoostingClassifier(interactions=0, outer_bags=1, max_rounds=30, n_jobs=1, random_state=42)),
    ]).fit(raw, target)
    bundle = {'base_model': pipeline, 'model': pipeline}
    explanation = native_record_explanation(bundle, raw.iloc[:4])
    np.testing.assert_allclose(expit(explanation['intercept']+explanation['contributions'].sum(axis=1)), pipeline.predict_proba(raw.iloc[:4])[:, 1])
    effect = native_effect(bundle, 1)
    assert len(effect) > 0
    assert np.isfinite(effect.Contribution).all()


def test_family_operating_point_does_not_discard_rare_attacks():
    pytest.importorskip('interpret')
    from scripts.train_interpretable_v2 import operating_point
    y = np.array([0, 0, 0, 1, 1, 1])
    families = np.array(['Benign', 'Benign', 'Benign', 'Common', 'Common', 'Rare'])
    p = np.array([.01, .02, .03, .9, .95, .04])
    selected = operating_point(y, p, families)
    assert selected['macro_attack_recall'] == 1.
    assert selected['false_positive_rate'] == 0.
    assert (p[y == 1] >= selected['threshold']).all()


def test_preparation_rebuilds_a_clone_without_changing_plan(tmp_path, monkeypatch):
    pytest.importorskip('pyarrow')
    from scripts import prepare_rich_data
    raw = tmp_path / 'data/raw/v2'
    raw.mkdir(parents=True)
    n = 100
    pd.DataFrame({
        'FLOW_START_MILLISECONDS': np.arange(n) * 100,
        'FLOW_END_MILLISECONDS': np.arange(n) * 100 + 1,
        'IN_BYTES': np.arange(n), 'Label': np.arange(n) % 2,
        'Attack': np.where(np.arange(n) % 2, 'Attack', 'Benign'),
    }).to_csv(raw / 'NF-UNSW-NB15-v3.csv', index=False)
    monkeypatch.setattr(prepare_rich_data, 'ROOT', tmp_path)
    prepare_rich_data.main()
    plan = (tmp_path / 'models/v2/evaluation_plan.json').read_bytes()
    (raw / 'prepared/train.parquet').unlink()
    prepare_rich_data.main()
    assert (raw / 'prepared/train.parquet').exists()
    assert (tmp_path / 'models/v2/evaluation_plan.json').read_bytes() == plan


def test_release_artifact_and_temporal_plan_match():
    import hashlib, json
    root = Path(__file__).resolve().parents[1]
    directory = root / 'models/v2'
    if not (directory / 'final_evaluation.json').exists():
        pytest.skip('V2 release not yet evaluated')
    development = json.loads((directory / 'development_report.json').read_text())
    evaluation = json.loads((directory / 'final_evaluation.json').read_text())
    plan = json.loads((directory / 'evaluation_plan.json').read_text())
    digest = hashlib.sha256((directory / 'detector.joblib').read_bytes()).hexdigest()
    assert digest == development['artifact_sha256'] == evaluation['artifact_sha256']
    roles = [plan['partitions'][r] for r in ['train', 'validation', 'calibration', 'lockbox']]
    assert all(a['end_ms'] < b['start_ms'] for a,b in zip(roles, roles[1:]))
    assert development['training_records'] == roles[0]['records'] > 1_000_000
    assert development['engineered_feature_count'] > 9


def test_release_native_explanations_work_without_shap():
    pytest.importorskip('interpret')
    import joblib
    from unittest.mock import patch
    root = Path(__file__).resolve().parents[1]
    if not (root / 'models/v2/detector.joblib').exists():
        pytest.skip('No v2 artifact')
    bundle = joblib.load(root / 'models/v2/detector.joblib')
    sample = pd.read_csv(root / 'data/rich_flow_sample.csv')
    with patch.dict('sys.modules', {'shap': None}):
        explanation = native_record_explanation(bundle, sample)
    assert np.isfinite(explanation['probability']).all()
    assert explanation['final_contributions'] is not None
    np.testing.assert_allclose(expit(explanation['final_intercept']+explanation['final_contributions'].sum(axis=1)), explanation['probability'], atol=1e-9)
    np.testing.assert_allclose(expit(explanation['intercept']+explanation['contributions'].sum(axis=1)), explanation['base_probability'])


def test_reference_feature_profiles_share_the_full_fit():
    pytest.importorskip('lightgbm')
    import joblib,json
    root = Path(__file__).resolve().parents[1]
    references = root / 'models/v2/reference_models.joblib'
    if not references.exists():
        pytest.skip('References not frozen')
    bundles = joblib.load(references)
    development = json.loads((root / 'models/v2/development_report.json').read_text())
    assert all(b['fitting_records'] == development['training_records'] for b in bundles.values())
    assert len(bundles['nine_input_lightgbm']['features']) == 9
    assert len(bundles['rich_lightgbm']['features']) > 9

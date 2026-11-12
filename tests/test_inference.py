from pathlib import Path
import io
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.data_preprocessing import encode_traffic, preprocess_features
from src.inference import read_traffic_csv, predict_severity, risk_categories, export_csv

ROOT = Path(__file__).resolve().parents[1]
FEATURES = pd.read_csv(ROOT / 'data/feature_columns.csv').feature.tolist()


@pytest.fixture
def sample():
    return pd.read_csv(ROOT / 'data/sample_input.csv')


def test_encoding_preserves_order_and_input(sample):
    original = sample.copy(deep=True)
    encoded = encode_traffic(sample, FEATURES)
    assert encoded.columns.tolist() == FEATURES
    assert encoded.loc[0, 'proto_tcp'] == 1
    assert encoded.loc[0, 'proto_udp'] == 0
    assert encoded.loc[0, 'sbytes'] == 100
    pd.testing.assert_frame_equal(sample, original)
    pd.testing.assert_frame_equal(encoded.iloc[:1], encode_traffic(sample.iloc[:1], FEATURES))


@pytest.mark.parametrize('change', ['missing_column', 'missing_number', 'infinite', 'negative', 'text_number', 'unknown_category', 'missing_category'])
def test_invalid_input_rejected(sample, change):
    if change == 'missing_column':
        sample = sample.drop(columns='duration')
    elif change in ('unknown_category', 'missing_category'):
        sample.loc[0, 'proto'] = 'sctp' if change == 'unknown_category' else None
    else:
        sample['duration'] = sample['duration'].astype(object)
        sample.loc[0, 'duration'] = {'missing_number': None, 'infinite': np.inf, 'negative': -1, 'text_number': 'invalid'}[change]
    with pytest.raises(ValueError):
        encode_traffic(sample, FEATURES)


def test_training_helper_does_not_scale(sample):
    values, columns = preprocess_features(sample.assign(label=0, attack_cat='Normal', severity_score=0))
    assert values[0, columns.get_loc('sbytes')] == 100
    assert 'label' not in columns


@pytest.mark.parametrize('content', [b'', b'a,a\n1,2', b'\xff\xfe', b'a\n', b'a,b\n"unterminated'])
def test_bad_csv_rejected(content):
    with pytest.raises(ValueError):
        read_traffic_csv(content)


def test_limits(monkeypatch):
    monkeypatch.setattr('src.inference.MAX_FILE_SIZE', 3)
    with pytest.raises(ValueError, match='file size'):
        read_traffic_csv(b'a\n12')
    monkeypatch.setattr('src.inference.MAX_FILE_SIZE', 100)
    monkeypatch.setattr('src.inference.MAX_ROWS', 1)
    with pytest.raises(ValueError, match='rows'):
        read_traffic_csv(b'a\n1\n2')


def test_sample_predictions_and_risk_bands(sample):
    model = joblib.load(ROOT / 'models/risk_score_model.pkl')
    encoded = encode_traffic(sample, FEATURES)
    scores = predict_severity(model, encoded)
    assert scores.shape == (5,)
    assert ((scores >= 0) & (scores <= 10)).all()
    np.testing.assert_allclose(scores[:1], predict_severity(model, encoded.iloc[:1]))
    assert risk_categories([0, 2.99, 3, 6.99, 7, 10]).tolist() == ['Low', 'Low', 'Medium', 'Medium', 'High', 'High']


@pytest.mark.parametrize('scores', [[np.nan], [11], [-1], [[3]]])
def test_invalid_predictions_rejected(sample, scores):
    class BrokenModel:
        def predict(self, data):
            return scores
    with pytest.raises(ValueError):
        predict_severity(BrokenModel(), encode_traffic(sample.iloc[:1], FEATURES))


def test_export_escapes_formulas_and_preserves_numbers():
    raw = pd.DataFrame({'note': ['=1+1', '  @SUM(A1)', 'normal'], 'number': [1, 2, 3]})
    exported = pd.read_csv(io.StringIO(export_csv(raw)))
    assert exported.note.tolist() == ["'=1+1", "'  @SUM(A1)", 'normal']
    assert exported.number.tolist() == [1, 2, 3]
    assert raw.note.iloc[0] == '=1+1'


def test_app_starts_outside_repo(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(str(ROOT / 'app/streamlit_visualizer.py')).run(timeout=30)
    assert not app.exception
    assert not app.error
    assert any('Research prototype' in w.value for w in app.warning)


def test_app_upload_success():
    uploaded = io.BytesIO((ROOT / 'data/netflow_sample.csv').read_bytes())
    with patch('streamlit.file_uploader', return_value=uploaded):
        app = AppTest.from_file(str(ROOT / 'app/streamlit_visualizer.py')).run(timeout=30)
    assert not app.exception
    assert not app.error
    assert len(app.metric) == 4
    assert any('Risk_Score' in frame.value.columns for frame in app.dataframe)


def test_app_rejects_incomplete_upload():
    with patch('streamlit.file_uploader', return_value=io.BytesIO(b'duration\n0.1')):
        app = AppTest.from_file(str(ROOT / 'app/streamlit_visualizer.py')).run(timeout=30)
    assert not app.exception
    assert 'Missing required columns' in app.error[0].value
    assert not app.metric


def test_missing_shap_is_nonfatal():
    uploaded = io.BytesIO((ROOT / 'data/netflow_sample.csv').read_bytes())
    with patch('streamlit.file_uploader', return_value=uploaded), patch.dict('sys.modules', {'shap': None}):
        app = AppTest.from_file(str(ROOT / 'app/streamlit_visualizer.py')).run(timeout=30)
        app.checkbox[0].check().run(timeout=30)
    assert not app.exception
    assert not app.error
    assert len(app.metric) == 4
    assert any('SHAP is unavailable' in w.value for w in app.warning)

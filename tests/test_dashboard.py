import io
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.dashboard import analysis_report, filter_traffic, severity_distribution

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / 'app/streamlit_visualizer.py')
APP_SAMPLE = ROOT / ('data/rich_flow_sample.csv' if (ROOT / 'models/v2/final_evaluation.json').exists() else 'data/netflow_sample.csv')
SAMPLE_COUNT = str(len(pd.read_csv(APP_SAMPLE)))


@pytest.fixture
def results():
    return pd.DataFrame({
        'proto': ['tcp', 'udp', 'tcp', 'icmp'],
        'Risk_Score': [0, 3, 7, 10],
        'Risk_Category': ['Low', 'Medium', 'High', 'High'],
    }, index=pd.Index([1, 2, 3, 4], name='Record'))


def test_filters_combine_and_preserve_source_rows(results):
    selected = filter_traffic(results, ['High'], ['tcp', 'icmp'], (7, 9))
    assert selected.index.tolist() == [3]
    assert len(results) == 4
    assert filter_traffic(results, [], ['tcp'], (0, 10)).empty
    assert filter_traffic(results, ['High'], [], (0, 10)).empty


def test_histogram_includes_extremes(results):
    histogram = severity_distribution(results)
    assert histogram.Records.sum() == 4
    assert histogram.loc['0–1', 'Records'] == 1
    assert histogram.loc['9–10', 'Records'] == 1


def test_report_is_consistent_with_batch(results):
    report = json.loads(analysis_report(results, 'TestModel', 'Sample traffic'))
    assert report['records'] == 4
    assert report['model'] == 'TestModel'
    assert report['source'] == 'Sample traffic'
    assert report['risk_counts'] == {'Low': 1, 'Medium': 1, 'High': 2}
    assert report['risk_score'] == {'mean': 5.0, 'min': 0.0, 'max': 10.0}
    assert sum(report['protocol_counts'].values()) == 4
    assert 'Research prototype' in report['limitations']


def button(app, label):
    return next(b for b in app.button if b.label == label)


def load_demo():
    app = AppTest.from_file(APP).run(timeout=30)
    button(app, 'Load sample traffic').click().run(timeout=30)
    assert not app.exception
    assert not app.error
    return app


def test_demo_load_inspect_and_clear():
    app = load_demo()
    assert app.metric[0].value == SAMPLE_COUNT
    assert [tab.label for tab in app.tabs] == ['Overview', 'Traffic explorer', 'Record inspector', 'Model details']
    app.number_input[0].set_value(2).run(timeout=30)
    assert not app.exception
    assert any('Estimated attack probability' in text.value for text in app.markdown)
    assert any('Input features' in text.value for text in app.markdown)
    button(app, 'Clear sample').click().run(timeout=30)
    assert not app.exception
    assert not app.metric


def test_protocol_filter_and_empty_state():
    app = load_demo()
    protocol = app.multiselect[1].options[0]
    app.multiselect[1].set_value([protocol]).run(timeout=30)
    tables = [f.value for f in app.dataframe if 'Risk_Category' in f.value.columns]
    assert set(tables[0]['proto']) == {protocol}
    assert all(i >= 1 for i in tables[0].index)
    # Batch summary remains unchanged by explorer filters.
    assert app.metric[0].value == SAMPLE_COUNT
    app.multiselect[0].set_value([]).run(timeout=30)
    assert not app.exception
    assert any('No records match' in info.value for info in app.info)


def test_upload_takes_priority_over_loaded_demo():
    app = load_demo()
    sample = pd.read_csv(APP_SAMPLE).iloc[:1]
    with patch('streamlit.file_uploader', return_value=io.BytesIO(sample.to_csv(index=False).encode())):
        app.run(timeout=30)
        assert not app.exception
        assert app.metric[0].value == '1'
        assert any('Uploaded CSV' in item.value for item in app.success)
    app.run(timeout=30)
    assert not app.exception
    assert app.metric[0].value == SAMPLE_COUNT


def test_inspector_missing_shap_is_nonfatal():
    app = load_demo()
    with patch.dict('sys.modules', {'shap': None}):
        choices = [b for b in app.button if b.label == 'Explain selected record']
        if choices:
            choices[0].click().run(timeout=30)
            assert any('Record explanation is unavailable' in item.value for item in app.warning)
        else:
            app.number_input[0].set_value(2).run(timeout=30)
            assert any('Why this record received its score' in item.value for item in app.markdown)
    assert not app.exception
    assert not app.error

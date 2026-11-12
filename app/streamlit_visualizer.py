"""Research dashboard backed by a reproducibly trained NetFlow attack detector."""
import hashlib
import json
import logging
from pathlib import Path
import sys
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import read_traffic_csv, risk_categories, export_csv
from src.dashboard import BANDS, filter_traffic, severity_distribution, analysis_report
from src.network_detector import FEATURES, validate_netflow, score_netflow, display_traffic, tree_explanations

logger = logging.getLogger(__name__)
MODEL_PATH = ROOT / 'models/netflow_detector.joblib'
REPORT_PATH = ROOT / 'models/training_report.json'
SAMPLE_PATH = ROOT / 'data/netflow_sample.csv'
st.set_page_config(page_title='FlowLens | Network Intrusion Analysis', page_icon='🔐', layout='wide')


@st.cache_resource
def load_model_and_report():
    report = json.loads(REPORT_PATH.read_text())
    if hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() != report['artifact_sha256']:
        raise ValueError('Model checksum does not match the evaluation report.')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', InconsistentVersionWarning)
        bundle = joblib.load(MODEL_PATH)
    if bundle['features'] != list(FEATURES) or report['features'] != list(FEATURES):
        raise ValueError('Model input schema does not match the app.')
    if bundle['model_name'] != report['selected_model'] or bundle['threshold'] != report['threshold']:
        raise ValueError('Model does not match the evaluation report.')
    mismatch = any(isinstance(w.message, InconsistentVersionWarning) for w in caught)
    return bundle, report, mismatch


def show_explanations(bundle, clean, max_samples):
    try:
        with st.spinner('Computing explanations…'), threadpool_limits(limits=2):
            values = tree_explanations(bundle, clean.iloc[:max_samples])
        import shap
        st.caption(f'First {min(len(clean), max_samples):,} records. SHAP explains the base estimator before any probability calibration; contribution units depend on that estimator, not the displayed risk index.')
        for title, plot in [('Feature effects', shap.plots.beeswarm), ('Feature importance', shap.plots.bar)]:
            st.markdown(f'**{title}**')
            try:
                plot(values, show=False)
                st.pyplot(plt.gcf())
            finally:
                plt.close('all')
    except ImportError:
        st.warning('SHAP is unavailable. Install requirements-shap.txt to enable explanations.')
    except Exception:
        logger.exception('SHAP explanation failed')
        st.warning('Explanations could not be computed. Predictions remain available.')


def render_overview(result, bundle, source):
    st.subheader('Traffic at a glance')
    st.caption('The risk index is estimated attack probability × 10. It does not measure the impact or severity of an attack.')
    left, right = st.columns(2)
    with left:
        st.markdown('**Risk distribution**')
        st.bar_chart(result['Risk_Category'].value_counts().reindex(BANDS, fill_value=0).rename('Records'), color='#2dd4bf')
    with right:
        st.markdown('**Risk index distribution**')
        st.bar_chart(severity_distribution(result), color='#60a5fa')
    left, right = st.columns(2)
    with left:
        st.markdown('**Average risk index by protocol**')
        st.bar_chart(result.groupby('proto')['Risk_Score'].mean().rename('Mean risk index'), color='#fbbf24')
    with right:
        st.markdown('**Destination port activity**')
        st.bar_chart(result['service'].value_counts().head(15).rename('Records'), color='#a78bfa')
    st.markdown('**Highest-scoring records**')
    columns = ['Risk_Score', 'Attack_Probability', 'Predicted_Attack', 'Prediction_Set', 'proto', 'service', 'IN_BYTES', 'OUT_BYTES']
    st.dataframe(result.nlargest(5, 'Risk_Score')[columns], use_container_width=True, column_config={
        'Risk_Score': st.column_config.ProgressColumn('Risk index', min_value=0, max_value=10, format='%.2f'),
        'Attack_Probability': st.column_config.NumberColumn('Estimated attack probability', format='%.4f'),
    })
    st.download_button('Download analysis report', analysis_report(result, bundle['model_name'], source), 'traffic_analysis.json', 'application/json')


def render_traffic(result):
    st.subheader('Traffic explorer')
    st.caption('Combine filters to narrow the batch. Record numbers refer to original CSV data rows, starting at 1. Overview statistics use the full batch.')
    left, middle, right = st.columns([1, 1, 2])
    bands = left.multiselect('Risk bands', list(BANDS), default=list(BANDS))
    protocols = middle.multiselect('Protocols', sorted(result['proto'].unique()), default=sorted(result['proto'].unique()))
    bounds = right.slider('Risk index range', 0.0, 10.0, (0.0, 10.0), step=.1)
    filtered = filter_traffic(result, bands, protocols, bounds)
    st.caption(f'{len(filtered):,} of {len(result):,} records match.')
    if filtered.empty:
        st.info('No records match. Broaden the risk, protocol, or risk index filters.')
    else:
        st.dataframe(filtered, use_container_width=True, column_config={
            'Risk_Score': st.column_config.ProgressColumn('Risk index', min_value=0, max_value=10, format='%.2f'),
        })
    numbered = filtered.copy()
    name = 'Source_Record'
    while name in numbered:
        name = '_' + name
    numbered.insert(0, name, numbered.index)
    st.download_button('Download filtered traffic', export_csv(numbered), 'filtered_traffic.csv', 'text/csv', disabled=filtered.empty)


def render_inspector(result, clean, bundle):
    st.subheader('Record inspector')
    st.caption(f'Choose a data-row number from 1 to {len(result):,}. The inspector uses the full batch.')
    record = st.number_input('Record number', min_value=1, max_value=len(result), value=1, step=1)
    row = result.loc[record]
    left, right = st.columns([1, 2])
    with left:
        st.markdown(f"### {row['Risk_Score']:.2f} / 10")
        st.progress(float(row['Attack_Probability']))
        st.write(f"Estimated attack probability: **{row['Attack_Probability']:.2%}**")
        st.write(f"Binary decision: **{'Attack' if row['Predicted_Attack'] else 'Benign'}**")
        st.caption(f"Decision threshold selected on validation data: {bundle['threshold']:.3f}. Risk display bands use separate 3/7 boundaries.")
        st.write(f"Experimental prediction set: **{row['Prediction_Set']}**")
        st.caption('An empty or multi-label set calls for review. Set coverage is not guaranteed on new networks.')
        st.write(f"Protocol: {row['proto']} · {row['service']}")
    with right:
        st.markdown('**Input features**')
        st.dataframe(clean.loc[record].rename('Value').to_frame(), use_container_width=True)
        st.caption('These are observed measurements. Feature values alone are not explanations or evidence of an attack.')
    if st.button('Explain selected record', key='explain_record'):
        try:
            with threadpool_limits(limits=2):
                values = tree_explanations(bundle, clean.loc[[record]])
            import shap
            try:
                shap.plots.waterfall(values[0], show=False)
                st.pyplot(plt.gcf())
            finally:
                plt.close('all')
            st.caption('SHAP explains the uncalibrated base model. Contributions do not sum directly to the displayed probability after any calibration or prove causality.')
        except Exception:
            logger.exception('Record explanation failed')
            st.warning('Record explanation is unavailable. Install the optional SHAP dependencies to enable it.')


def render_model_details(bundle, report, clean, explain, max_samples):
    st.subheader('Measured model quality')
    st.write(f"Dataset: **{report['dataset']}** · selected model: **{report['selected_model']}** · seed: {report['seed']}")
    st.write('Predicts the real binary attack label. Source/destination IPs, source port, attack family, and labels are excluded from model inputs.')
    comparison = pd.DataFrame(report['validation_comparison']).T
    st.markdown('**Validation comparison — used for model selection**')
    st.dataframe(comparison[['average_precision', 'roc_auc', 'recall', 'false_positive_rate', 'fit_seconds']], use_container_width=True)
    if report['status'] != 'benchmark_targets_met':
        st.warning('Some benchmark targets are unmet. Review per-attack support and recall before interpreting aggregate results.')
    if report.get('validation_policies'):
        with st.expander('Validation policies and decision-threshold selection'):
            st.write(report['selection_criterion'])
            policies = pd.DataFrame(report['validation_policies']).T
            st.dataframe(policies[['recall', 'false_positive_rate', 'average_precision', 'brier', 'threshold']], use_container_width=True)
    st.markdown('**Held-out test set**')
    test = report['test']
    st.write(f"{test['records']:,} records · average precision {test['average_precision']:.5f} · recall {test['recall']:.2%} · false-positive rate {test['false_positive_rate']:.3%}")
    st.caption('Holdouts retain all valid flows, including ambiguous and repeated patterns. Only fitting data are deduplicated. These are fixed host-pair benchmark results, not production detection rates.')
    st.dataframe(pd.DataFrame(report['per_attack_test']).T.rename(columns={'correct_binary_decision_rate': 'Binary decision accuracy / attack recall'}), use_container_width=True)
    absent = [name for name, values in report['per_attack_test'].items() if values['records'] == 0]
    if absent:
        st.warning('No test support for: ' + ', '.join(absent) + '. No detection rate is established for these categories.')
    if 'per_dataset_test' in report:
        st.markdown('**Test results by source dataset**')
        st.dataframe(pd.DataFrame(report['per_dataset_test']).T[['records', 'average_precision', 'recall', 'false_positive_rate']], use_container_width=True)
    if report.get('novel_pattern_test', {}).get('metrics'):
        novel = report['novel_pattern_test']
        st.caption(f"Novel feature patterns: {novel['records']:,} test records absent from development data; recall {novel['metrics']['recall']:.2%}, false-positive rate {novel['metrics']['false_positive_rate']:.2%}.")
    st.markdown('**Uncertainty and calibration checks**')
    conformal = report['conformal']
    st.write(f"Nominal conformal coverage: {1-conformal['alpha']:.0%}; measured test coverage: {conformal['test_coverage']:.2%}.")
    st.write(f"Probability policy: {report.get('probability_policy', 'sigmoid')}. Brier score (lower is better): selected policy {test['brier']:.6f}; raw base model {test['uncalibrated_brier']:.6f}.")
    st.caption('Calibration can worsen under distribution change. Conformal sets are experimental; dependent flows and drift can break exchangeability assumptions.')
    history_path = ROOT / 'reports/initial_external_evaluation.json'
    if history_path.exists():
        history = json.loads(history_path.read_text())
        with st.expander('Earlier external failure — superseded single-source model'):
            st.write(f"The initial model detected {history['metrics']['recall']:.2%} of attacks in NF-BoT-IoT. This motivated combined-source development.")
            st.caption('This is historical evidence for an earlier artifact. NF-BoT-IoT is now part of training; no independent third-source validation is claimed for the current model.')
    st.markdown('**Permutation importance on held-out records**')
    importance = pd.DataFrame(report['permutation_importance']).set_index('feature')
    st.bar_chart(importance['mean_ap_decrease'].sort_values(ascending=False), color='#2dd4bf')
    st.caption('Average-precision decrease after shuffling one raw input. Correlated features can share or obscure importance. This is not causal evidence.')
    with st.expander('Split, cleaning, and limitations'):
        st.json({'cleaning': report['cleaning'], 'split_counts': report['split_counts'], 'limitations': report['limitations']})
    st.download_button('Download training evidence', json.dumps(report, indent=2), 'training_report.json', 'application/json')
    if explain:
        show_explanations(bundle, clean, max_samples)
    else:
        st.info('Enable SHAP in the sidebar for batch explanations, or explain a single record in the inspector.')


def main():
    st.markdown('NETWORK SECURITY / ANALYSIS WORKSPACE')
    st.title('Traffic risk, in focus.')
    st.write('Explore NetFlow records with a trained attack detector, inspect predictions, and review the evaluation evidence.')
    st.warning('Research prototype: exploratory benchmark results are measured; the test partition was inspected during development. Fresh external validation is needed. Risk index estimates attack likelihood, not severity.')
    try:
        bundle, report, mismatch = load_model_and_report()
    except Exception:
        logger.exception('Unable to load detector and report')
        st.error('Unable to load the trained detector and its evaluation report. Check the bundled files or run scripts/train_detector.py.')
        return
    with st.sidebar:
        st.header('FlowLens')
        st.caption('NETFLOW RESEARCH PROTOTYPE')
        st.divider()
        st.write(f"{bundle['model_name']} · {len(FEATURES)} inputs")
        st.caption(f"Trained on {report['dataset']}")
        if mismatch:
            st.warning(f"Model version mismatch. Use scikit-learn {report['scikit_learn']} as specified in requirements.txt.")
        st.markdown('**Analysis workflow**')
        st.write('1. Load traffic\n2. Review the overview\n3. Filter and inspect records\n4. Check model evidence and export')
        st.divider()
        explain = st.checkbox('Enable SHAP explanations', value=False)
        max_samples = st.slider('Rows to explain', 10, 1000, 100, disabled=not explain)
        st.caption('Display bands: Low <3, Medium 3–<7, High ≥7. These are not validated alert thresholds.')
    with st.expander('Required NetFlow CSV format'):
        st.code(', '.join(FEATURES))
        st.write('Use NetFlow v1 fields from NF-ToN-IoT. All nine inputs must be finite, nonnegative, and complete. Ports, IP protocol identifiers, flags, byte counts, and packet counts must be whole numbers.')
        st.caption('Duration is in milliseconds; L7_PROTO is the numeric Layer 7 identifier. UTF-8 CSV · maximum 50 MB / 100,000 rows. Extra columns are retained and ignored by the model.')
    with st.container(border=True):
        left, right = st.columns([2, 1])
        with left:
            st.subheader('Start an analysis')
            uploaded = st.file_uploader('Upload traffic CSV', type=['csv'])
        with right:
            st.markdown('**Explore real sample flows**')
            st.caption('Eight feature-only records drawn from the held-out dataset. They demonstrate inference, not a separate evaluation.')
            if st.button('Load sample traffic', type='primary'):
                st.session_state['use_sample'] = True
            st.download_button('Download sample CSV', SAMPLE_PATH.read_bytes(), 'netflow_sample.csv', 'text/csv')
            if st.session_state.get('use_sample') and st.button('Clear sample'):
                st.session_state['use_sample'] = False
    if uploaded is not None:
        content, source = uploaded.getvalue(), 'Uploaded CSV'
    elif st.session_state.get('use_sample'):
        content, source = SAMPLE_PATH.read_bytes(), 'Combined-dataset holdout sample'
    else:
        st.info('Upload a CSV or choose Load sample traffic to open the analysis workspace.')
        return
    try:
        raw = read_traffic_csv(content)
        raw.index = pd.RangeIndex(1, len(raw) + 1, name='Record')
        clean = validate_netflow(raw)
    except ValueError as exc:
        st.error(str(exc))
        return
    st.success(f'{source} · {len(raw):,} records validated · {len(FEATURES)} model inputs')
    try:
        with st.spinner('Scoring traffic…'), threadpool_limits(limits=2):
            scored = score_netflow(bundle, clean)
    except Exception:
        logger.exception('Prediction failed')
        st.error('The model could not score this data. Check the server logs for details.')
        return
    result = display_traffic(raw, clean)
    for col in scored:
        result[col] = scored[col]
    result['Risk_Category'] = risk_categories(result['Risk_Score'])
    for col, label, value in zip(st.columns(4),
                                 ('Records analyzed', 'Mean risk index', 'Predicted attack records', 'Uncertain prediction sets'),
                                 (f'{len(result):,}', f"{result['Risk_Score'].mean():.3f}", int(result['Predicted_Attack'].sum()), int(result['Prediction_Set'].isin(['Empty set — review', 'Benign or Attack']).sum()))):
        with col:
            with st.container(border=True):
                st.metric(label, value)
    overview, traffic, inspector, details = st.tabs(['Overview', 'Traffic explorer', 'Record inspector', 'Model details'])
    with overview:
        render_overview(result, bundle, source)
    with traffic:
        render_traffic(result)
    with inspector:
        render_inspector(result, clean, bundle)
    with details:
        render_model_details(bundle, report, clean, explain, max_samples)
    st.divider()
    st.download_button('Download predictions', export_csv(result), 'threat_predictions.csv', 'text/csv')
    st.caption('Research prototype · no live network monitoring · uploaded traffic is processed in memory by this app.')


if __name__ == '__main__':
    main()

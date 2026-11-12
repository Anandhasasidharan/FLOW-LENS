"""Model-native explanation views for the FlowLens v2 release."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy.special import expit

from src.interpretable_detector import native_record_explanation, native_effect


def record_terms(bundle, clean, record):
    if bundle['model_name'] == 'explainable_boosting':
        explanation = native_record_explanation(bundle, clean.loc[[record]])
        exact_final = explanation['final_contributions'] is not None
        values = explanation['final_contributions'][0] if exact_final else explanation['contributions'][0]
        names = explanation['terms']
        intercept = explanation['final_intercept'] if exact_final else explanation['intercept']
        base_probability = explanation['probability'][0] if exact_final else explanation['base_probability'][0]
    else:
        exact_final = False
        pipeline = bundle['base_model']
        transformed = pipeline.named_steps['features'].transform(clean.loc[[record]])
        contributions = pipeline.named_steps['estimator'].predict(transformed, pred_contrib=True)[0]
        values, intercept = contributions[:-1], contributions[-1]
        names = list(pipeline.named_steps['features'].get_feature_names_out())
        base_probability = expit(intercept + np.sum(values))
    st.markdown('**Why this record received its score**')
    st.caption('Positive contributions raise attack log-odds; negative contributions lower them. ' + ('These terms reconstruct the final deployed probability, including sigmoid calibration.' if exact_final else 'These terms refer to the base classifier before probability calibration.'))
    table = pd.DataFrame({'Term': names, 'Contribution (log-odds)': values})
    table = table.iloc[np.argsort(-np.abs(values))]
    st.bar_chart(table.head(12).set_index('Term'), horizontal=True, color='#2dd4bf')
    with st.expander('Full contribution breakdown and reconstruction'):
        st.dataframe(table, hide_index=True, use_container_width=True)
        level = 'Final' if exact_final else 'Base'
        st.code(f'{level} log-odds = {intercept:.6f} + {np.sum(values):.6f}\n{level} attack probability = sigmoid(log-odds) = {base_probability:.6%}')
        st.caption('This additive sum is checked against the saved classifier, rather than approximating a different model.')


def model_details(bundle, report, clean):
    development = report['development']
    evaluation = report['evaluation']
    st.subheader('A richer model with built-in interpretation')
    st.write(f"**{development['model_name']}** · {len(bundle['features'])} raw fields · {development['engineered_feature_count']} model features · {development['training_records']:,} fitting records")
    st.caption('The full training fit uses every eligible chronological training row. Smaller stratified samples are used only to screen hyperparameters.')
    st.markdown('**Validation finalists**')
    comparison = pd.DataFrame(development['finalist_validation']).T
    st.dataframe(comparison[['average_precision', 'macro_attack_recall', 'recall', 'false_positive_rate', 'fit_seconds']], use_container_width=True)
    st.markdown('**Final chronological test**')
    measured = evaluation['selected_model']
    st.write(f"{evaluation['test_records']:,} future-flow records · recall {measured['recall']:.2%} · benign false-positive rate {measured['false_positive_rate']:.3%} · average precision {measured['average_precision']:.5f}")
    st.caption('Models and operating thresholds were frozen before this test was opened. Results describe this laboratory dataset and its time split.')
    per_family = pd.DataFrame(measured['per_family']).T.rename(columns={'binary_correctness': 'Binary correctness / attack recall'})
    st.dataframe(per_family, use_container_width=True)
    references = {name: {k:v for k,v in result.items() if k in ['average_precision', 'recall', 'false_positive_rate', 'macro_attack_recall', 'f1']} for name,result in evaluation['references'].items()}
    st.markdown('**References on the same final test**')
    st.dataframe(pd.DataFrame(references).T, use_container_width=True)
    if bundle['model_name'] == 'explainable_boosting':
        estimator = bundle['base_model'].named_steps['estimator']
        importance = pd.Series(estimator.term_importances(), index=estimator.term_names_, name='Mean absolute contribution')
        st.markdown('**Native model-term importance**')
        st.bar_chart(importance.nlargest(15), horizontal=True, color='#2dd4bf')
        mains = [i for i, term in enumerate(estimator.term_features_) if len(term) == 1]
        chosen = st.selectbox('Explore a feature effect', mains, format_func=lambda i: estimator.term_names_[i])
        effect = native_effect(bundle, chosen)
        st.caption('Learned feature effect in attack log-odds. Individual predictions also include the intercept, other features, and any interaction terms.')
        if pd.api.types.is_numeric_dtype(effect.Value):
            st.line_chart(effect.set_index('Value'), color='#60a5fa')
        else:
            st.bar_chart(effect.set_index('Value'), color='#60a5fa')
        scores = np.asarray(estimator.term_scores_[chosen])
        st.caption(f'Missing-value contribution: {scores[0]:.4f}; unknown-value contribution: {scores[-1]:.4f}. These bins are disclosed separately from the plotted observed-value curve.')
    st.markdown('**Probability reliability**')
    st.write(f"Selected policy: {development['probability_policy']} · Brier score: {measured['brier']:.6f} · calibration error: {measured['expected_calibration_error']:.6f}")
    reliability = pd.DataFrame(measured['reliability_bins'])
    st.dataframe(reliability, hide_index=True, use_container_width=True)
    with st.expander('Evaluation design, quality checks, and scope'):
        st.json({'design': evaluation['evaluation_design'], 'checks': evaluation['quality_checks'], 'limitations': evaluation['limitations']})
    st.download_button('Download v2 evaluation evidence', json.dumps(evaluation, indent=2), 'flowlens_v2_evaluation.json', 'application/json')

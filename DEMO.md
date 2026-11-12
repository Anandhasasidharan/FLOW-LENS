# Presenting FlowLens v2

Launch with `python -m streamlit run app/streamlit_visualizer.py` after installing `requirements.txt`.

1. **Load sample traffic.** Ten real development examples cover benign traffic and nine attack families. These examples are separate from the final lockbox. Explain that the index represents attack likelihood rather than severity.
2. **Review the overview.** Show record counts, predicted attacks, risk distribution, and protocol/port summaries. The actual trained EBM scores the examples.
3. **Explore traffic.** Filter by protocol, band, and risk index. Export a filtered CSV with source-row references. The overview continues to describe the full batch.
4. **Inspect a record.** Select a row and review its raw measurements and signed feature contributions. Positive terms raise attack log-odds; negative terms lower them. No SHAP installation or additional explanation button is required.
5. **Verify the explanation.** Open the reconstruction panel. The calibrated intercept and scaled native terms reconstruct the final deployed probability. This explains the same model used for inference.
6. **Explore model effects.** In Model details, inspect term importance and feature-effect curves. Missing and unknown bins are disclosed separately. Discuss why a model's association is not a causal intervention.
7. **Review evidence.** Show the full-data training count, development comparison, and the reserved chronological test. EBM and both LightGBM references detected all 31,107 attack flows; their false-positive counts were 2, 4, and 450 respectively for EBM, rich LightGBM, and the nine-input reference.
8. **Export the evaluation.** Download the frozen evidence and point reviewers to `docs/MODEL_CARD_V2.md` for per-family support, calibration, and scope.

## Interview topics

Explain temporal leakage and boundary purging, why final-fit sampling differs from hyperparameter screening, how an EBM adds learned feature functions, how sigmoid calibration preserves an exact additive log-odds decomposition, and why the test result does not establish performance on every network.

The underlying traffic is a 2015 laboratory capture with a richer v3 extraction. A separate environment is the next independent evaluation. The prototype does not perform live packet capture or classify attack families.

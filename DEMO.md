# Presenting the trained prototype

Run `streamlit run app/streamlit_visualizer.py` from the repository root for the configured dark theme.

1. **Load sample traffic.** Eight records come from actual held-out flows. Explain that the new model predicts attack likelihood, and the 0–10 index is estimated probability × 10, not severity.
2. **Review the overview.** Show risk distributions, protocol summaries, destination ports, highest-scoring records, and prediction-set review counts.
3. **Filter traffic.** In Traffic explorer, narrow protocol, risk bands, or index range. Overview statistics continue to describe the full batch. Filtered CSV exports retain source-row references.
4. **Inspect a record.** Select a row, review the nine model inputs, predicted probability, binary decision threshold, and experimental conformal label set. Empty or multi-label sets indicate uncertainty. Optional SHAP explains the base estimator before calibration.
5. **Show the model evidence.** Open Model details for validation comparisons, final holdout and per-family results, dataset-source metrics, calibration diagnostics, and limitations. Be explicit about unsupported attack categories and small samples.
6. **Explain the failed experiment.** The first model's excellent in-domain result did not transfer: external attack recall was only 7.87%. Show how this motivated combined-source training and a stricter full-holdout evaluation. The failed evidence is preserved.
7. **Export the evidence.** Download predictions, the analysis report, and the full training report. Point reviewers to `docs/MODEL_CARD.md` and the explainability research review.

## Interview discussion

Useful topics include target leakage, why attack likelihood differs from severity, why accuracy is misleading under imbalance, why calibration can worsen under shift, why nominal conformal coverage can fail, and why newer explainability methods cannot rescue a weak detector.

Be precise: this is a reproducible research prototype. The final model has not passed a truly unseen third-source, future-capture, or production-network validation. It does not perform live packet capture or classify attack families.

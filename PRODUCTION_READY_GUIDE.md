# Remaining validation and deployment work

The repository now contains a trained detector with measured benchmark evidence. It is a research prototype, not a production-certified intrusion detector.

## Implemented

- Reproducible training from two official UQ NetFlow dataset downloads with source checksums and academic reuse terms.
- A saved preprocessing/model/calibration bundle, with a matching hashed report checked during app loading.
- Group-separated training, validation, probability calibration, conformal calibration, and test roles; fitting-only deduplication and bounded training size.
- Baseline/model comparison, threshold selection, full holdout metrics, per-source/per-family rates, novel-pattern diagnostics, group-bootstrap intervals, permutation importance, and uncertainty checks.
- Preserved evidence of a failed single-source external evaluation.
- Strict bounded CSV validation, export formula escaping, repository-relative paths, in-memory upload processing, and optional SHAP failure handling.
- A demo flow, interactive exploration and inspection, measured model-details view, and downloadable evidence.

## Priorities

1. **Truly independent validation.** Evaluate the final model on a third dataset or a future capture with compatible extraction. Do not retune on that final test. Measure benign false-positive rates at realistic prevalence. Address zero-support and low-recall attack categories using new development captures, then reserve a fresh final evaluation.
2. **Evaluation realism.** Host-pair grouping still shares individual hosts and lab artifacts across partitions. Add temporal, device-disjoint, and cross-network evaluations. Check proxy features and extraction artifacts. Do not treat repeated flows as independent observations when forming confidence intervals.
3. **Calibration and uncertainty.** Assess probability reliability per source and subgroup. A sigmoid calibrator is not guaranteed to improve every distribution. Investigate failures of conformal coverage before treating its sets as alert-control mechanisms; monitor drift and abstain or request review when assumptions fail.
4. **Explainability evaluation.** SHAP is associative and model-relative. If adding counterfactuals, enforce valid protocol/flag/count relationships, fixed categorical fields where appropriate, plausible ranges, and clear interpretation. A model-score change is not a remediation recommendation. See the research review for newer directions and their evidence limits.
5. **Deployment reproducibility.** Create a complete dependency lock and CI environment using the recorded Python/package versions. The requirements pin scikit-learn but are not a fully resolved lockfile. Publish model and report together as one atomic release. Load trusted serialized artifacts only.
6. **Access and workload controls.** Add authentication, TLS, proxy and app upload limits, per-user/concurrent workload limits, retention rules, access controls for model/log files, and latency/error monitoring without logging uploaded traffic. In-memory parsing and UI limits alone do not bound concurrent server load.
7. **Independent release criteria.** Set operational precision/recall/false-positive requirements with a security practitioner, measure alert burden, document failure handling, and require evidence before promotion. Current risk bands and engineering thresholds are prototype conventions.

For actual measured results, read `docs/MODEL_CARD.md` and `models/training_report.json`. Tests verify the software's behavior; they do not establish detection efficacy or eliminate dataset bias.

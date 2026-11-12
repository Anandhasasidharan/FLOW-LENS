# FlowLens v2: validation and deployment

The active EBM meets the recorded chronological benchmark targets and provides native explanations of the deployed probability. It is a research prototype; production readiness also requires validation in the intended environment and operational controls.

## Implemented

- Richer NetFlow v3 measurements and eight engineered features.
- Full fitting on 1,419,227 eligible records, with smaller samples used only for parameter screening.
- Chronological fitting, validation, calibration, and final-test roles with boundary-flow purging.
- Development-only model/policy/threshold selection, a frozen evaluation plan, and a single prospective final evaluation.
- Two full-data LightGBM references for predictive comparison and feature-profile ablation.
- Native EBM terms, feature-effect curves, missing/unknown bin disclosure, and exact reconstruction including sigmoid calibration.
- Publisher-source checksums, model/report integrity checks, runtime metadata, native explanation tests, and Streamlit interaction tests.
- Separate experiment outputs and explicitly labeled replays that preserve the active scored release.

## Next independent validation

Use a separate capture or environment with compatible feature extraction. Do not retune using its final test labels. Measure false alerts at realistic benign prevalence, support for rare families, probability reliability, and drift. The original dataset's traffic dates to 2015, and laboratory service/TTL patterns may not transfer.

## Operational work

1. Add authentication, TLS, proxy upload limits, concurrent workload controls, and retention rules appropriate to the deployment.
2. Establish latency, failure, calibration, and drift monitoring without logging uploaded traffic contents.
3. Ship model and evidence as an atomic release; load trusted artifacts only and match recorded package versions.
4. Add CI and a fully resolved deployment lockfile. Current requirements pin the model libraries and the recorded environment lists tested versions.
5. Define alert thresholds and review procedures with the users of the system. Display bands are separate from the binary decision threshold.
6. If adding scenario or counterfactual tools, enforce networking feasibility and explain the difference between changing a model input and mitigating an attack.

See `docs/MODEL_CARD_V2.md` for measured results, `models/v2/final_evaluation.json` for raw evidence, and `models/v2/feature_audit.json` for the model's dominant terms. Browser visuals have not been inspected in this workspace; application interactions and native explanation fidelity are tested programmatically.

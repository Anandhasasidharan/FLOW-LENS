# FlowLens

### Interpretable Network Intrusion Detection

A network-traffic analysis workspace powered by an **Explainable Boosting Machine (EBM)**. FlowLens combines a richer NetFlow model, native feature explanations, interactive investigation, and reproducible evaluation.

**The v2 model uses 48 raw measurements and 8 engineered features, fitted on 1,419,227 flows.** Its explanations reconstruct the deployed probability, including sigmoid calibration, without an optional SHAP dependency.

[Quick start](#quick-start) · [Measured results](#measured-results) · [Model card](docs/MODEL_CARD_V2.md) · [Reproduce training](#reproduce-the-experiment) · [Demo walkthrough](DEMO.md)

## What you can do

- Load real development examples or upload NetFlow CSV records.
- Review attack-likelihood distributions and protocol/port activity.
- Filter traffic and inspect individual predictions.
- See which native feature contributions raise or lower a record's risk.
- Explore learned feature-effect curves and missing-value behavior.
- Compare the EBM against two LightGBM references on the same reserved test.
- Export predictions, filtered traffic, and measured evaluation evidence.

The target is **benign versus attack**. The displayed 0–10 risk index is estimated attack probability × 10; it does not estimate attack damage or severity.

## Measured results

The active model is a native additive EBM with sigmoid probability calibration. The final test was reserved before model development and scored after the model, probability policy, references, and thresholds were frozen.

**Final test: 354,814 chronologically later flows — 31,107 attacks and 323,707 benign flows.**

| Model | Detected attacks | False positives | Benign false-positive rate |
| --- | ---: | ---: | ---: |
| Nine-input LightGBM reference | 31,107 / 31,107 | 450 | 0.139015% |
| Rich-feature LightGBM | 31,107 / 31,107 | 4 | 0.001236% |
| **FlowLens EBM** | **31,107 / 31,107** | **2** | **0.000618%** |

All nine attack families were present in the final test. EBM precision was **99.9936%**, and all predeclared release quality checks passed. Both LightGBM references use the same full training partition, selected LightGBM parameters, and eight derived features; their raw-input profiles differ.

These are results from one laboratory dataset's chronological holdout, not a promise of performance on arbitrary networks. Rare categories have limited support, including 37 Worms records. See the [model card](docs/MODEL_CARD_V2.md) for per-family counts, calibration diagnostics, dominant features, and scope.

## Interpretation is built into the model

An EBM learns feature-effect functions that add to a decision score. FlowLens uses the fitted terms directly:

```text
final log-odds = calibrated intercept + sum(calibrated feature contributions)
attack probability = sigmoid(final log-odds)
```

For the selected sigmoid calibration policy, its scale and offset preserve this additive representation. The record inspector checks the reconstruction against the actual deployed classifier.

- **Record contributions:** Signed terms explain the selected record's final attack probability.
- **Feature effects:** Curves show how observed values contribute to the base model's log-odds.
- **Native importance:** Mean absolute term contributions summarize model behavior.
- **Missing and unknown values:** Their learned contributions are disclosed separately from observed-value curves.

These explanations describe the fitted model; they do not establish causality or provide remediation instructions. Optional SHAP support remains in the historical v1 path, but v2 explanations work without it.

## Quick start

Use **Python 3.11 or later**; the recorded training environment used Python 3.12.

```bash
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in Windows PowerShell, then:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app/streamlit_visualizer.py
```

Open the URL printed by Streamlit, normally `http://localhost:8501`. Choose **Load sample traffic** to explore ten feature-only development examples, including benign traffic and nine attack families.

The trained model is included; full datasets are not required to run the app. Launch from the repository root for the configured dark theme. Restart Streamlit after replacing a model release to refresh its cached artifact and evidence.

## Input format

Download the current sample from the app or use [data/rich_flow_sample.csv](data/rich_flow_sample.csv). It contains the exact 48-column schema. The historical nine-field samples are not inputs for the active v2 model.

The richer measurements cover:

- Source/destination ports, protocol identifiers, and TCP flags.
- Bytes, packets, durations, sizes, and directional throughput.
- Retransmissions, packet-size distributions, and TCP windows.
- DNS, FTP, ICMP, TTL, and inter-packet arrival statistics.

Eight derived features add aggregate counts, directional ratios, rates, and bytes per packet. They are computed from each record by the saved feature builder. Absolute timestamps, endpoint IPs, DNS transaction IDs, and labels are excluded from the model.

See the [publisher's feature definitions](data/netflow_v3_features.csv). Upload UTF-8 CSVs with headers, up to **50 MB / 100,000 rows**. Empty numeric measurements are handled natively; nonnumeric text, infinite input values, duplicate headers, and entirely empty records are rejected. Examples encode nonfinite publisher readings as empty cells. No missing measurements are silently replaced with zero.

Results include `Attack_Probability`, `Risk_Score`, `Predicted_Attack`, and `Risk_Category`. Display bands are Low (<3), Medium (3–<7), and High (≥7); they are separate from the binary decision threshold chosen on validation data.

Extra fields are excluded from inference and retained for export. Generated model outputs and dashboard aliases replace uploaded fields with matching names. Filtered exports preserve original source-row references; formula-like text is escaped for spreadsheet opening.

## Training and evaluation design

**Data:** [NF-UNSW-NB15-v3 from the University of Queensland](https://staff.itee.uq.edu.au/marius/NIDS_datasets/), with 2,365,424 flows and richer temporal/packet statistics. The original traffic is from 2015; v3 provides a newer extraction, not contemporary production traffic.

1. Reserve chronological fitting, validation, calibration, and final-test partitions at 60/15/10/15 percent flow-start quantiles.
2. Purge flows crossing partition boundaries using their flow-end timestamps.
3. Screen EBM and LightGBM parameters on smaller development samples, including a TTL-free ablation.
4. Fit finalists on **all 1,419,227 eligible training records**, without the old final-fit sample cap.
5. Compare probability policies and choose validation macro attack-family recall at a benign false-positive target of 1%.
6. Prefer native EBM interpretation when its validation performance is within one percentage point of the predictive challenger.
7. Freeze the artifacts and evaluate the previously unopened final partition once.

The [evaluation plan](models/v2/evaluation_plan.json), [development trials](models/v2/development_trials.json), [development report](models/v2/development_report.json), [final evaluation](models/v2/final_evaluation.json), and [feature audit](models/v2/feature_audit.json) make these choices inspectable. Native explanations are checked against actual model predictions.

## Reproduce the experiment

Install training dependencies and prepare the publisher's data:

```bash
python -m pip install -r requirements-training.txt
python scripts/fetch_rich_dataset.py
python scripts/prepare_rich_data.py
```

Replay the public benchmark in a separate directory, preserving the active release:

```bash
python scripts/train_interpretable_v2.py --output models/experiments/v2-reproduction
python scripts/freeze_reference_models_v2.py --experiment-dir models/experiments/v2-reproduction
python scripts/evaluate_interpretable_v2.py --experiment-dir models/experiments/v2-reproduction
python scripts/write_interpretable_model_card.py --experiment-dir models/experiments/v2-reproduction --output docs/MODEL_CARD_V2_REPRODUCTION.md
```

The default training output is `models/v2` for an initial experiment. A scored release is preserved; use `--output` for subsequent work. A replay of an already published split is explicitly labeled as a reproduction, not a new blind evaluation. Future model development needs a fresh independent test for new generalization claims.

Raw downloads and prepared datasets stay in ignored `data/raw/`. [Dataset provenance](data/rich_dataset_provenance.json) records the official source and file hashes. [Environment metadata](models/v2/environment.json) records the versions used. Prepared data can be reconstructed from a fresh clone without changing the frozen evaluation plan.

## Verification

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Checks cover schema validation, artifact integrity, temporal separation, clone reconstruction, prediction parity, optional legacy explanations, native contribution reconstruction, operation without SHAP, exports, uploads, filters, and the demo lifecycle. The software test suite is separate from measured model performance.

Browser appearance has not been visually verified in this workspace; Streamlit interactions are checked with its application test runner.

## Repository

```text
app/                      Analysis workspace and native explanation views
src/interpretable_detector.py
                          Rich features and exact EBM explanations
scripts/                  Download, preparation, search, freezing, and evaluation
models/v2/                Active model, evaluation plan, references, and evidence
data/                     Feature definitions, development sample, and provenance
notebooks/                Exploration and model inspection
reports/                  Historical artifact audits and experiments
tests/                    Model, inference, and dashboard checks
docs/                     Model cards, research, and historical documentation
```

[DEMO.md](DEMO.md) provides a presentation walkthrough. [The research review](docs/EXPLAINABILITY_RESEARCH_2026.md) covers EBM, counterfactuals, HyConEx, and uncertainty methods. [The deployment guide](PRODUCTION_READY_GUIDE.md) lists operational work beyond this benchmark.

## Scope and next work

The new release has strong measured detection and faithful interpretation on its reserved benchmark. A separate environment or future capture remains the next independent evaluation. Ports, protocol behavior, and TTL patterns can reflect a laboratory's traffic generator, so feature interpretation and testing across environments remain important.

Next additions include a compatible external capture, drift monitoring, constrained scenario exploration, authentication, and deployment workload controls. Historical v1 artifacts remain available for traceability; [the archived v1 README](docs/history/README_V1.md) describes that earlier experiment.

## Dataset attribution

Follow the [publisher's dataset page](https://staff.itee.uq.edu.au/marius/NIDS_datasets/) and [v3 temporal-analysis paper](https://arxiv.org/abs/2503.04404) for citations. Academic research reuse includes a commercial-use restriction requiring agreement with the authors. Dataset terms do not constitute a software license for this repository.

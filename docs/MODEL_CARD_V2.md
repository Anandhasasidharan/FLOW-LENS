# FlowLens v2 model card

**Model:** explainable_boosting with sigmoid probability calibration. **Dataset:** NF-UNSW-NB15-v3. **Evaluated:** 2026-11-12T10:56:49.522115+00:00.

## Intended use

A reproducible, interpretable NetFlow intrusion-detection research prototype. The target is the publisher's binary `Label`: benign or attack. This model does not classify attack families or estimate attack damage. The dashboard risk index is estimated attack probability multiplied by ten.

This release replaces the active v1 model. Historical v1 evidence remains available separately; its feature limit, sampled final fit, threshold policy, and post-hoc-only explanation path do not define v2.

## Architecture and training

- 48 raw measurements; 56 features after engineering.
- 1,419,227 rows supplied to the full fitting procedure, with no final-fit row cap.
- 140,094 development records used only for efficient parameter screening.
- Optuna parameter search compared LightGBM and Explainable Boosting Machines. This is parameter search, not a claim of a newly invented algorithm.
- The selection rule prefers native EBM interpretation when validation average precision and macro attack-family recall are within one percentage point of the predictive challenger.
- Raw, sigmoid, and isotonic probability policies were compared on validation data; calibration methods fit a separate calibration partition.
- The operating threshold, **0.806637467**, maximizes validation macro attack-family recall with benign FPR ≤1%. The final test does not select the threshold.
- Raw source/destination ports, protocol metadata, flow statistics, packet-size distributions, retransmissions, TCP windows/flags, DNS/FTP/ICMP fields, and inter-arrival statistics are available. Engineered terms cover aggregate bytes/packets, directional ratios, rates, and packet size.

Absolute flow timestamps, endpoint IPs, DNS transaction IDs, and both label columns are excluded from model inputs. Timing fields are used only to define the evaluation split; within-flow duration and inter-arrival statistics remain legitimate model measurements.

Negative or nonfinite publisher measurements are represented as missing by the fitted feature builder. Missing and unknown bins are part of the learned model and are disclosed in the interface. CSV examples use empty cells for nonfinite publisher readings; no fabricated zero measurements are inserted.

## Prospective temporal evaluation

Design: Chronological 60/15/10/15 percent by flow-start quantiles, purged at boundaries using flow-end timestamps.

| Partition | Records | Earliest flow start (ms) | Latest flow end (ms) |
| --- | ---: | ---: | ---: |
| train | 1,419,227 | 1421927376907 | 1424229622772 |
| validation | 354,790 | 1424229622862 | 1424242209341 |
| calibration | 236,522 | 1424242209344 | 1424250496106 |
| lockbox | 354,814 | 1424250496133 | 1424262565011 |

Boundary flows purged: **71**. Roles are separated in time, including flow-end purging. The prospective evaluation plan was written before model development. Models, probability policies, reference models, and thresholds were frozen before the final lockbox was first scored. Its predictions were not used to revise the chosen model.

Reproduction runs of this public split are replays, not newly blind evaluations. The recorded original release is preserved rather than overwritten by replay commands.

## Final measured results

| Model | Attack recall | False positives | Benign FPR | Average precision |
| --- | ---: | ---: | ---: | ---: |
| Explainable Boosting Machine (active) | 100.0000% | 2 | 0.000618% | 1.00000000 |
| nine_input_lightgbm | 100.0000% | 450 | 0.139015% | 0.99998282 |
| rich_lightgbm | 100.0000% | 4 | 0.001236% | 1.00000000 |

The final test contains **354,814 flows**: **31,107 attacks** and **323,707 benign flows**. The active EBM has **0 false negatives** and **2 false positives**. Precision is **99.9936%**, F1 is **0.99996785**, and macro attack-family recall is **100.0000%**.

Both LightGBM references use the same full training partition, selected LightGBM parameters, and eight engineered features. Their difference is the raw-input profile: nine fields versus the richer schema. Their thresholds are chosen independently on validation data using the same operating-point rule. The EBM is a different model family; the two-vs-four false-positive difference alone does not establish statistical superiority over rich LightGBM.

| Family | Final-test support | Binary correctness / attack recall |
| --- | ---: | ---: |
| Analysis | 294 | 100.0000% |
| Backdoor | 1,260 | 100.0000% |
| Benign | 323,707 | 99.9994% |
| DoS | 1,234 | 100.0000% |
| Exploits | 10,815 | 100.0000% |
| Fuzzers | 7,387 | 100.0000% |
| Generic | 5,126 | 100.0000% |
| Reconnaissance | 4,548 | 100.0000% |
| Shellcode | 406 | 100.0000% |
| Worms | 37 | 100.0000% |

These are binary detection rates, not multiclass classification accuracies. Worms has only 37 test records; perfect observed recall on a rare category is not a guarantee about future attacks.

## Faithful interpretation

The EBM represents its base decision score as an intercept plus learned feature terms. Native term contributions reconstruct the base log-odds exactly. For this release's single sigmoid calibration link, its fitted scale and offset can also be applied to the additive terms. The dashboard verifies that their sum reconstructs the **final deployed attack probability**, including calibration.

This requires no optional SHAP installation. The record inspector shows signed term contributions and the exact reconstruction. Model details show native term importance, observed-value feature-effect curves, and separate missing/unknown bin contributions. Feature effects refer to the fitted model, not causal mechanisms or recommended remediations.

| Top model term | Mean absolute base log-odds contribution |
| --- | ---: |
| L4_DST_PORT | 3.4218 |
| L7_PROTO | 2.6228 |
| DNS_QUERY_TYPE | 2.0269 |
| DNS_TTL_ANSWER | 1.9898 |
| TCP_FLAGS | 1.9024 |
| MIN_TTL | 1.8459 |
| CLIENT_TCP_FLAGS | 1.4074 |
| PROTOCOL | 1.3025 |
| SRC_TO_DST_AVG_THROUGHPUT | 1.1891 |
| SERVER_TCP_FLAGS | 1.0833 |
| L4_SRC_PORT | 1.0666 |
| BYTES_PER_SECOND | 0.8413 |
| DST_TO_SRC_SECOND_BYTES | 0.8152 |
| TCP_WIN_MAX_IN | 0.7648 |
| ICMP_TYPE | 0.7645 |

Strong terms include service/port and protocol-related information. These patterns can reflect the laboratory's traffic generator and environment. The TTL-free development trial also performed strongly; it is a validation ablation, not a separately evaluated final release.

Probability Brier score is **0.000006799**; ten-bin expected calibration error is **0.000027930**. Reliability-bin counts and measured attack fractions are recorded in the JSON report. Calibration on this test does not guarantee reliability under distribution shift.

## Release checks

Status: **benchmark_targets_met**.

- PASS: attack_recall_at_least_95_percent
- PASS: benign_false_positive_rate_at_most_1_percent
- PASS: macro_attack_recall_at_least_90_percent
- PASS: every_attack_family_at_least_90_percent
- PASS: all_development_families_have_test_support
- PASS: native_additive_explanation_verified

## Scope and next evaluation

The result is strong on this previously unopened chronological portion of one laboratory dataset. Its underlying traffic dates to 2015; newer feature extraction and model software do not make it contemporary production traffic. All nine attack families are represented here, but some have limited support. Correlated flows are not independent trials.

No claim is made of universal attack detection, causal explanations, zero-day identification, or cross-network performance. A separate capture/environment is the next independent evaluation. High within-benchmark scores are not deployment certification.

## Artifacts and reproduction

- `models/v2/detector.joblib`: active fitted pipeline, calibration policy, and threshold.
- `models/v2/development_report.json`: model selection and development-only comparisons.
- `models/v2/final_evaluation.json`: final frozen-model results and reference comparisons.
- `models/v2/evaluation_plan.json`: source checksum and temporal partition definition.
- `models/v2/feature_audit.json`: excluded identifiers and dominant native terms.
- `models/v2/environment.json`: recorded runtime package versions.
- `data/rich_dataset_provenance.json`: official source and file hashes.

Artifact SHA-256: `b2964e6c5203eaebae90b0451d22cfefdbc2ea87e0defed7600fcc34034c253c`. The app checks this against both development and evaluation evidence before loading the model.

Use the reproduction instructions in README.md. A replay writes a separate experiment directory and is explicitly marked as reuse of a previously evaluated split. Software checks also verify native contribution reconstruction, operation without SHAP, group/time separation, and a fresh clone's ability to rebuild prepared inputs from its frozen plan.

## Sources and reuse

Dataset: [UQ NetFlow collection](https://staff.itee.uq.edu.au/marius/NIDS_datasets/), [official delivery record](https://rdm.uq.edu.au/files/abd2f5d8-e268-4ff0-84fb-f2f7b3ca3e8f), and [Temporal Analysis of NetFlow Datasets for Network Intrusion Detection Systems](https://arxiv.org/abs/2503.04404). Follow the publisher's attribution requirements; academic reuse permission includes a commercial-use restriction.

Model: [InterpretML Explainable Boosting Machine](https://interpret.ml/docs/ebm.html) and its [classifier API](https://interpret.ml/docs/python/api/ExplainableBoostingClassifier.html). Challenger: [LightGBM classifier](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html). These are established model families with maintained implementations, not methods newly invented for this project.

# NetFlow detector model card

Generated from `models/training_report.json` (2026-11-12T10:21:54.187649+00:00). Artifact SHA-256: `b652d406ee718f8a9dab649f655167cb9f7d3ab39d0812dbd6cff023c62e8d18`.

## Intended use and target

A portfolio/research prototype for exploring NetFlow-based binary attack detection. Dataset: **NF-ToN-IoT v1 + NF-BoT-IoT v1**. Model: **hist_gradient_boosting**, fitted on **250,000** records. Binary target is the publisher's `Label`. The displayed risk index is estimated attack probability × 10; it is not a measured severity score. Do not treat probability estimates or display bands as operational guarantees.

Training and inference share a saved pipeline with nine raw inputs. IP addresses, source port, binary labels, attack-family labels, and dataset-source identifiers are excluded from model inputs. Addresses are used only to construct evaluation groups.

## Experimental history

The legacy artifact was structurally inconsistent with its documented raw/one-hot preprocessing; see `reports/legacy_model_audit.json`.

An initial model trained on NF-ToN-IoT performed well on a cleaned in-domain test but detected only **7.87%** of attacks on the full, independent NF-BoT-IoT challenge. That failed experiment is retained under `models/experiments/nf_ton_iot/` and `reports/initial_external_evaluation.json`.

The final experiment trains across both sources. Those sources were already inspected during the initial challenge, so this is an iterative combined-source benchmark, not a pristine independent external validation. Earlier cleaned-subset results are archived under `models/experiments/combined_clean_subset/`; the final evaluation retains all valid holdout flows, including repeated and ambiguous patterns.

## Data and evaluation

- Input rows: 1,979,374; invalid rows excluded: 0.
- Split: Disjoint (dataset source, unordered IPv4 host pair) groups; holdouts retain all valid records, including repeated and ambiguous feature patterns. Only fitting records are deduplicated on (inputs, binary target).
- Fitting-only duplicates removed: 725,816.
- Model-selection rule: Maximum validation attack recall subject to each source benign FPR <=5%; ties broken by validation average precision then Brier score. Raw/sigmoid/isotonic policies compared; test data excluded from selection. Raw, sigmoid, and isotonic probability policies are compared on validation data. Calibration methods fit their own holdout; conformal scores use another. Test labels do not fit estimators, calibrators, or thresholds.
- Binary threshold: **0.804**, selected for maximum validation recall subject to a 5% benign false-positive target for each represented source. This is a prototype engineering target, not a security SLA.
- Test rows: **541,272**; test attack prevalence: **84.20%**.
- Different groups can share individual hosts; this is not an unseen-host or temporal benchmark.

| Validation candidate | Average precision | ROC AUC | Fit seconds |
| --- | ---: | ---: | ---: |
| dummy_prior | 0.836640 | 0.500000 | 0.4 |
| logistic_regression | 0.995876 | 0.986088 | 4.7 |
| extra_trees | 0.999354 | 0.996994 | 30.5 |
| hist_gradient_boosting | 0.999241 | 0.996166 | 24.4 |

## Final holdout results

| Metric | Selected model | Prior-only baseline |
| --- | ---: | ---: |
| Average precision | 0.999912 | 0.842022 |
| ROC AUC | 0.999559 | 0.500000 |
| Attack recall | 99.95% | 100.00% |
| Benign false-positive rate | 1.76% | 100.00% |
| Precision | 99.67% | 84.20% |
| F1 | 0.998112 | 0.914237 |

Confusion matrix `[[TN, FP], [FN, TP]]`: `[[84000, 1509], [214, 455549]]`.

| Held-out source | Records | Attack recall | Benign false-positive rate |
| --- | ---: | ---: | ---: |
| NF-BoT-IoT | 84,581 | 99.84% | 32.56% |
| NF-ToN-IoT | 456,691 | 99.98% | 1.34% |

| Attack family / source | Test records | Binary correctness (attack recall for attack rows) |
| --- | ---: | ---: |
| NF-BoT-IoT / Benign | 1,167 | 67.44% |
| NF-BoT-IoT / DDoS | 8,693 | 100.00% |
| NF-BoT-IoT / DoS | 8,693 | 100.00% |
| NF-BoT-IoT / Reconnaissance | 66,028 | 99.79% |
| NF-BoT-IoT / Theft | 0 | Not evaluated |
| NF-ToN-IoT / Benign | 84,342 | 98.66% |
| NF-ToN-IoT / backdoor | 60 | 20.00% |
| NF-ToN-IoT / ddos | 163,436 | 100.00% |
| NF-ToN-IoT / dos | 8 | 100.00% |
| NF-ToN-IoT / injection | 104,724 | 99.99% |
| NF-ToN-IoT / mitm | 79 | 82.28% |
| NF-ToN-IoT / password | 87,495 | 100.00% |
| NF-ToN-IoT / ransomware | 0 | Not evaluated |
| NF-ToN-IoT / scanning | 12,713 | 100.00% |
| NF-ToN-IoT / xss | 3,834 | 100.00% |

Zero-support categories are not evaluated. Small samples do not establish robust detection performance. The model does not classify attack families.

### Novel feature patterns

Test patterns absent from all development partitions (defined without test labels):

```json
{
  "records": 217233,
  "fraction_of_test": 0.4013379594732408,
  "metrics": {
    "records": 217233,
    "attack_prevalence": 0.7127416184465528,
    "average_precision": 0.999950757402017,
    "roc_auc": 0.9998830376634382,
    "precision": 0.9938378697920682,
    "recall": 0.9989536979028747,
    "f1": 0.9963892172557407,
    "false_positive_rate": 0.015368097176372553,
    "brier": 0.004792834084027191,
    "log_loss": 0.022669597007153198,
    "confusion_matrix": [
      [
        61443,
        959
      ],
      [
        162,
        154669
      ]
    ]
  }
}
```

Natural repeated feature patterns remain in the full test. These are descriptive diagnostics; test comparisons are not used to choose another release model.

### Descriptive uncertainty

95% host-pair group-bootstrap intervals, fixed model and 500 resamples:

```json
{
  "recall": [
    0.9986082910130797,
    0.9999418512367361
  ],
  "false_positive_rate": [
    0.004009702664812646,
    0.13777914960472476
  ],
  "precision": [
    0.9888879121888038,
    0.9990393623847367
  ]
}
```

These intervals omit model-fitting variability and do not guarantee generalization to another network.

## Calibration and uncertainty

Brier score (lower is better): selected **sigmoid** policy **0.003200**, raw base model **0.003172**. Calibration may worsen under a changed distribution; both measurements are disclosed.

Experimental conformal sets target 90% nominal coverage. Observed test coverage is **96.17%**; mean set size is **0.962**. This is measured row-level coverage, not a deployment promise. Flow dependence, group distribution differences, and drift undermine simple exchangeability assumptions. Empty or multi-label sets should be reviewed.

## Quality checks

Status: **benchmark_targets_not_met**. Passing aggregate metrics alone does not certify a great or production-ready detector.

- PASS: test_average_precision_above_prevalence
- PASS: test_recall_at_least_90_percent
- PASS: test_false_positive_rate_at_most_5_percent
- PASS: host_pairs_disjoint
- FAIL: every_attack_family_has_at_least_30_test_records
- FAIL: every_supported_attack_recall_at_least_90_percent
- FAIL: every_source_test_false_positive_rate_at_most_5_percent

## Limitations

- Lab dataset with attack prevalence unlike many real networks; no real-world performance guarantee.
- The combined-data experiment was designed after a failed NF-BoT-IoT external challenge. Its dataset sources were previously inspected; this is not a pristine external validation.
- Host pairs are disjoint, but individual hosts can occur in multiple splits; no claim of unseen-host, temporal, or cross-dataset generalization.
- Holdouts retain ambiguous and repeated patterns. Training-only deduplication changes fitting weights; probability calibration uses a separate raw holdout.
- Binary labels are ground truth; risk score is 10 times estimated attack probability, not attack severity.
- Raw, sigmoid and isotonic probability policies were compared on validation data. Their probability reliability and decision thresholds may fail under distribution shift.
- Per-attack detection rates are binary decisions, not attack-family classification accuracy.
- This host-pair test partition was inspected across pipeline revisions. Results are exploratory and need confirmation on a fresh locked test source or capture.

No truly unseen third-source dataset, future capture, or live network has validated the final combined-source model. The prior independent source is now part of development.

## Reproduce

```bash
python3 scripts/fetch_dataset.py
python3 scripts/fetch_dataset.py --dataset nf-bot-iot
python3 scripts/train_detector.py --include-bot
python3 scripts/write_model_card.py
python3 -m pytest -q
```

Raw datasets are excluded from Git. Download provenance and checksums are in `data/dataset_provenance.json` and `data/external_dataset_provenance.json`. Download endpoints are official University of Queensland records. Academic reuse is allowed with citation; commercial reuse requires agreement with the authors. Cite Sarhan, Layeghy, Moustafa and Portmann (2021), *NetFlow Datasets for Machine Learning-Based Network Intrusion Detection Systems*. [Publisher's dataset collection](https://staff.itee.uq.edu.au/marius/NIDS_datasets/).

Runtime: Python 3.12.3, scikit-learn 1.9.0, pandas 2.3.3, NumPy 2.4.6. Seeds, partition counts, timings, metrics and permutation importance are recorded in the JSON report. Thread pools and forest parallelism are limited to two workers during training. Timing is machine-specific.

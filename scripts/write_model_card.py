"""Render measured training evidence into a model card."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    report = json.loads((ROOT / 'models/training_report.json').read_text())
    test = report['test']
    baseline = report['test_dummy']
    conformal = report['conformal']
    rows = '\n'.join(
        f"| {name} | {values['records']:,} | {values['correct_binary_decision_rate']:.2%} |"
        if values['correct_binary_decision_rate'] is not None else f"| {name} | 0 | Not evaluated |"
        for name, values in sorted(report['per_attack_test'].items())
    )
    comparison = '\n'.join(f"| {name} | {values['average_precision']:.6f} | {values['roc_auc']:.6f} | {values['fit_seconds']:.1f} |" for name, values in report['validation_comparison'].items())
    domains = '\n'.join(f"| {name} | {values['records']:,} | {values['recall']:.2%} | {values['false_positive_rate']:.2%} |" for name, values in report.get('per_dataset_test', {}).items())
    checks = '\n'.join(f"- {'PASS' if value else 'FAIL'}: {name}" for name, value in report['engineering_checks'].items())
    caveats = '\n'.join('- '+x for x in report['limitations'])
    novelty = report.get('novel_pattern_test', {})
    novelty_text = json.dumps(novelty, indent=2)
    intervals = json.dumps(report.get('group_bootstrap_95_intervals', {}), indent=2)
    text = f'''# NetFlow detector model card

Generated from `models/training_report.json` ({report['created_at']}). Artifact SHA-256: `{report['artifact_sha256']}`.

## Intended use and target

A portfolio/research prototype for exploring NetFlow-based binary attack detection. Dataset: **{report['dataset']}**. Model: **{report['selected_model']}**, fitted on **{report['fitted_records']:,}** records. Binary target is the publisher's `Label`. The displayed risk index is estimated attack probability × 10; it is not a measured severity score. Do not treat probability estimates or display bands as operational guarantees.

Training and inference share a saved pipeline with nine raw inputs. IP addresses, source port, binary labels, attack-family labels, and dataset-source identifiers are excluded from model inputs. Addresses are used only to construct evaluation groups.

## Experimental history

The legacy artifact was structurally inconsistent with its documented raw/one-hot preprocessing; see `reports/legacy_model_audit.json`.

An initial model trained on NF-ToN-IoT performed well on a cleaned in-domain test but detected only **7.87%** of attacks on the full, independent NF-BoT-IoT challenge. That failed experiment is retained under `models/experiments/nf_ton_iot/` and `reports/initial_external_evaluation.json`.

The final experiment trains across both sources. Those sources were already inspected during the initial challenge, so this is an iterative combined-source benchmark, not a pristine independent external validation. Earlier cleaned-subset results are archived under `models/experiments/combined_clean_subset/`; the final evaluation retains all valid holdout flows, including repeated and ambiguous patterns.

## Data and evaluation

- Input rows: {report['cleaning']['original_records']:,}; invalid rows excluded: {report['cleaning']['invalid_records_removed']:,}.
- Split: {report['split_method']}
- Fitting-only duplicates removed: {report.get('fit_duplicates_removed', 0):,}.
- Model-selection rule: {report['selection_criterion']} Raw, sigmoid, and isotonic probability policies are compared on validation data. Calibration methods fit their own holdout; conformal scores use another. Test labels do not fit estimators, calibrators, or thresholds.
- Binary threshold: **{report['threshold']:.3f}**, selected for maximum validation recall subject to a 5% benign false-positive target for each represented source. This is a prototype engineering target, not a security SLA.
- Test rows: **{test['records']:,}**; test attack prevalence: **{test['attack_prevalence']:.2%}**.
- Different groups can share individual hosts; this is not an unseen-host or temporal benchmark.

| Validation candidate | Average precision | ROC AUC | Fit seconds |
| --- | ---: | ---: | ---: |
{comparison}

## Final holdout results

| Metric | Selected model | Prior-only baseline |
| --- | ---: | ---: |
| Average precision | {test['average_precision']:.6f} | {baseline['average_precision']:.6f} |
| ROC AUC | {test['roc_auc']:.6f} | {baseline['roc_auc']:.6f} |
| Attack recall | {test['recall']:.2%} | {baseline['recall']:.2%} |
| Benign false-positive rate | {test['false_positive_rate']:.2%} | {baseline['false_positive_rate']:.2%} |
| Precision | {test['precision']:.2%} | {baseline['precision']:.2%} |
| F1 | {test['f1']:.6f} | {baseline['f1']:.6f} |

Confusion matrix `[[TN, FP], [FN, TP]]`: `{test['confusion_matrix']}`.

| Held-out source | Records | Attack recall | Benign false-positive rate |
| --- | ---: | ---: | ---: |
{domains}

| Attack family / source | Test records | Binary correctness (attack recall for attack rows) |
| --- | ---: | ---: |
{rows}

Zero-support categories are not evaluated. Small samples do not establish robust detection performance. The model does not classify attack families.

### Novel feature patterns

Test patterns absent from all development partitions (defined without test labels):

```json
{novelty_text}
```

Natural repeated feature patterns remain in the full test. These are descriptive diagnostics; test comparisons are not used to choose another release model.

### Descriptive uncertainty

95% host-pair group-bootstrap intervals, fixed model and 500 resamples:

```json
{intervals}
```

These intervals omit model-fitting variability and do not guarantee generalization to another network.

## Calibration and uncertainty

Brier score (lower is better): selected **{report.get('probability_policy', 'sigmoid')}** policy **{test['brier']:.6f}**, raw base model **{test['uncalibrated_brier']:.6f}**. Calibration may worsen under a changed distribution; both measurements are disclosed.

Experimental conformal sets target 90% nominal coverage. Observed test coverage is **{conformal['test_coverage']:.2%}**; mean set size is **{conformal['test_mean_set_size']:.3f}**. This is measured row-level coverage, not a deployment promise. Flow dependence, group distribution differences, and drift undermine simple exchangeability assumptions. Empty or multi-label sets should be reviewed.

## Quality checks

Status: **{report['status']}**. Passing aggregate metrics alone does not certify a great or production-ready detector.

{checks}

## Limitations

{caveats}

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

Runtime: Python {report['python']}, scikit-learn {report['scikit_learn']}, pandas {report['pandas']}, NumPy {report['numpy']}. Seeds, partition counts, timings, metrics and permutation importance are recorded in the JSON report. Thread pools and forest parallelism are limited to two workers during training. Timing is machine-specific.
'''
    (ROOT / 'docs/MODEL_CARD.md').write_text(text)
    print('Generated docs/MODEL_CARD.md')


if __name__ == '__main__':
    main()

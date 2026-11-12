"""Generate the v2 model card from frozen development and final-test evidence."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir', type=Path, default=ROOT / 'models/v2')
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/MODEL_CARD_V2.md')
    args = parser.parse_args()
    directory = args.experiment_dir
    development = json.loads((directory / 'development_report.json').read_text())
    evaluation = json.loads((directory / 'final_evaluation.json').read_text())
    plan = json.loads((directory / 'evaluation_plan.json').read_text())
    measured = evaluation['selected_model']
    matrix = measured['confusion_matrix']
    rows = {'Explainable Boosting Machine (active)': measured, **evaluation['references']}
    comparison = '\n'.join(f"| {name} | {value['recall']:.4%} | {value['confusion_matrix'][0][1]:,} | {value['false_positive_rate']:.6%} | {value['average_precision']:.8f} |" for name,value in rows.items())
    families = '\n'.join(f"| {name} | {value['records']:,} | {value['binary_correctness']:.4%} |" for name,value in measured['per_family'].items())
    partitions = '\n'.join(f"| {role} | {value['records']:,} | {value['start_ms']} | {value['end_ms']} |" for role,value in plan['partitions'].items())
    checks = '\n'.join(f"- {'PASS' if passed else 'FAIL'}: {name}" for name,passed in evaluation['quality_checks'].items())
    audit_path = directory / 'feature_audit.json'
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    terms = '\n'.join(f"| {name} | {value:.4f} |" for name,value in audit.get('top_terms', {}).items())
    text = f'''# FlowLens v2 model card

**Model:** {development['model_name']} with {development['probability_policy']} probability calibration. **Dataset:** {development['dataset']}. **Evaluated:** {evaluation['evaluated_at']}.

## Intended use

A reproducible, interpretable NetFlow intrusion-detection research prototype. The target is the publisher's binary `Label`: benign or attack. This model does not classify attack families or estimate attack damage. The dashboard risk index is estimated attack probability multiplied by ten.

This release replaces the active v1 model. Historical v1 evidence remains available separately; its feature limit, sampled final fit, threshold policy, and post-hoc-only explanation path do not define v2.

## Architecture and training

- {len(development['features'])} raw measurements; {development['engineered_feature_count']} features after engineering.
- {development['training_records']:,} rows supplied to the full fitting procedure, with no final-fit row cap.
- {development['screening_records']:,} development records used only for efficient parameter screening.
- Optuna parameter search compared LightGBM and Explainable Boosting Machines. This is parameter search, not a claim of a newly invented algorithm.
- The selection rule prefers native EBM interpretation when validation average precision and macro attack-family recall are within one percentage point of the predictive challenger.
- Raw, sigmoid, and isotonic probability policies were compared on validation data; calibration methods fit a separate calibration partition.
- The operating threshold, **{development['threshold']:.9f}**, maximizes validation macro attack-family recall with benign FPR ≤1%. The final test does not select the threshold.
- Raw source/destination ports, protocol metadata, flow statistics, packet-size distributions, retransmissions, TCP windows/flags, DNS/FTP/ICMP fields, and inter-arrival statistics are available. Engineered terms cover aggregate bytes/packets, directional ratios, rates, and packet size.

Absolute flow timestamps, endpoint IPs, DNS transaction IDs, and both label columns are excluded from model inputs. Timing fields are used only to define the evaluation split; within-flow duration and inter-arrival statistics remain legitimate model measurements.

Negative or nonfinite publisher measurements are represented as missing by the fitted feature builder. Missing and unknown bins are part of the learned model and are disclosed in the interface. CSV examples use empty cells for nonfinite publisher readings; no fabricated zero measurements are inserted.

## Prospective temporal evaluation

Design: {plan['split']}

| Partition | Records | Earliest flow start (ms) | Latest flow end (ms) |
| --- | ---: | ---: | ---: |
{partitions}

Boundary flows purged: **{plan['boundary_flows_purged']:,}**. Roles are separated in time, including flow-end purging. The prospective evaluation plan was written before model development. Models, probability policies, reference models, and thresholds were frozen before the final lockbox was first scored. Its predictions were not used to revise the chosen model.

Reproduction runs of this public split are replays, not newly blind evaluations. The recorded original release is preserved rather than overwritten by replay commands.

## Final measured results

| Model | Attack recall | False positives | Benign FPR | Average precision |
| --- | ---: | ---: | ---: | ---: |
{comparison}

The final test contains **{evaluation['test_records']:,} flows**: **{matrix[1][0]+matrix[1][1]:,} attacks** and **{matrix[0][0]+matrix[0][1]:,} benign flows**. The active EBM has **{matrix[1][0]:,} false negatives** and **{matrix[0][1]:,} false positives**. Precision is **{measured['precision']:.4%}**, F1 is **{measured['f1']:.8f}**, and macro attack-family recall is **{measured['macro_attack_recall']:.4%}**.

Both LightGBM references use the same full training partition, selected LightGBM parameters, and eight engineered features. Their difference is the raw-input profile: nine fields versus the richer schema. Their thresholds are chosen independently on validation data using the same operating-point rule. The EBM is a different model family; the two-vs-four false-positive difference alone does not establish statistical superiority over rich LightGBM.

| Family | Final-test support | Binary correctness / attack recall |
| --- | ---: | ---: |
{families}

These are binary detection rates, not multiclass classification accuracies. Worms has only 37 test records; perfect observed recall on a rare category is not a guarantee about future attacks.

## Faithful interpretation

The EBM represents its base decision score as an intercept plus learned feature terms. Native term contributions reconstruct the base log-odds exactly. For this release's single sigmoid calibration link, its fitted scale and offset can also be applied to the additive terms. The dashboard verifies that their sum reconstructs the **final deployed attack probability**, including calibration.

This requires no optional SHAP installation. The record inspector shows signed term contributions and the exact reconstruction. Model details show native term importance, observed-value feature-effect curves, and separate missing/unknown bin contributions. Feature effects refer to the fitted model, not causal mechanisms or recommended remediations.

| Top model term | Mean absolute base log-odds contribution |
| --- | ---: |
{terms}

Strong terms include service/port and protocol-related information. These patterns can reflect the laboratory's traffic generator and environment. The TTL-free development trial also performed strongly; it is a validation ablation, not a separately evaluated final release.

Probability Brier score is **{measured['brier']:.9f}**; ten-bin expected calibration error is **{measured['expected_calibration_error']:.9f}**. Reliability-bin counts and measured attack fractions are recorded in the JSON report. Calibration on this test does not guarantee reliability under distribution shift.

## Release checks

Status: **{evaluation['status']}**.

{checks}

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

Artifact SHA-256: `{evaluation['artifact_sha256']}`. The app checks this against both development and evaluation evidence before loading the model.

Use the reproduction instructions in README.md. A replay writes a separate experiment directory and is explicitly marked as reuse of a previously evaluated split. Software checks also verify native contribution reconstruction, operation without SHAP, group/time separation, and a fresh clone's ability to rebuild prepared inputs from its frozen plan.

## Sources and reuse

Dataset: [UQ NetFlow collection](https://staff.itee.uq.edu.au/marius/NIDS_datasets/), [official delivery record](https://rdm.uq.edu.au/files/abd2f5d8-e268-4ff0-84fb-f2f7b3ca3e8f), and [Temporal Analysis of NetFlow Datasets for Network Intrusion Detection Systems](https://arxiv.org/abs/2503.04404). Follow the publisher's attribution requirements; academic reuse permission includes a commercial-use restriction.

Model: [InterpretML Explainable Boosting Machine](https://interpret.ml/docs/ebm.html) and its [classifier API](https://interpret.ml/docs/python/api/ExplainableBoostingClassifier.html). Challenger: [LightGBM classifier](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html). These are established model families with maintained implementations, not methods newly invented for this project.
'''
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(f'Generated {args.output}')


if __name__ == '__main__':
    main()

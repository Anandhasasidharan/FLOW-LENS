# Explainability and model research

Research checked on 12 November 2026 using original papers, maintainer documentation, and dataset publishers. **Implementation update:** FlowLens v2 now uses a native Explainable Boosting Machine; [its model card](MODEL_CARD_V2.md) records the richer-data experiment and reserved temporal evaluation. This is a targeted literature review, not an exhaustive survey or a claim that a particular method is universally best.

## Is there something more advanced than SHAP?

There are methods that answer different questions. SHAP attributes a model output to features relative to a baseline. It does not establish causality, measure detection accuracy, estimate calibrated uncertainty, or show that changing a feature will prevent an attack. Keep attribution where it helps; compare explanation quality rather than replacing it for novelty alone.

| Approach | What it adds | Evidence and date | Fit for this project |
| --- | --- | --- | --- |
| Constrained diverse counterfactuals | Shows alternative inputs that change the model output | [DiCE, authors' implementation](https://github.com/interpretml/DiCE), original FAT* 2020 paper | Practical next addition, with fixed protocol/category fields, valid ranges, and networking constraints. A score-changing input is not a mitigation instruction. |
| Learned-manifold counterfactuals | Tries to keep alternatives closer to the learned tabular distribution | [TABCF](https://arxiv.org/abs/2410.10463), ICAIF 2024 | Research comparison. Its reported experiments use financial datasets; transfer to network flows must be evaluated. Adds VAE training and runtime complexity. |
| Joint prediction and counterfactual generation | Builds local counterfactual explanations into the classifier | [HyConEx](https://arxiv.org/abs/2503.12525), March 2026 revision; paper identifies Neurocomputing 2026 publication | A verifiable 2026 direction. It is a different classification architecture, not a plugin replacement for explaining an existing forest. No superiority on this project's traffic is established. |
| Intrinsically interpretable models | Exposes model terms directly, including selected interactions | [Explainable Boosting Machine, InterpretML](https://interpret.ml/docs/ebm.html) | Implemented in v2 and compared against LightGBM. Native terms reconstruct the deployed probability, including its sigmoid calibration. EBM is established technology rather than a new 2026 method. |
| Conformal uncertainty | Returns plausible label sets or regression intervals, with assumptions | [MAPIE theory](https://mapie.readthedocs.io/en/stable/content/conformal-prediction/theory/) | A useful complement to attribution. Standard marginal coverage needs exchangeability; dependent network flows and drift can violate it. Report empirical coverage, set size, and subgroup behavior. |
| Counterfactuals for prediction sets | Explains changes to the whole conformal label set | [Maalej et al., PMLR 266](https://proceedings.mlr.press/v266/maalej25a.html), August 2025 | A research extension after classifier and uncertainty calibration are independently checked. Changes the question from one class label to uncertainty-aware decisions. |

A 2026 industrial cybersecurity review discusses how operational usefulness and reliability remain open issues across explanation families; its arXiv version is a preprint, not proof of a new method outperforming all others. [Mohamed et al., August 2026](https://arxiv.org/abs/2609.00171)

## Stronger predictive models are a separate question

Gradient boosting and tree ensembles are sensible tabular baselines. A foundation model is an experiment, not an automatic upgrade. The January 2025 TabPFN paper reports strong small-data results on datasets up to 10,000 samples; that specific result does not justify assuming it wins on millions of correlated network flows. [Hollmann et al., Nature](https://www.nature.com/articles/s41586-024-08328-6)

V2 now compares a native EBM against rich-feature LightGBM and a nine-input LightGBM reference. Hyperparameters are screened on development samples; finalists use all 1,419,227 training records. A chronological final partition was reserved before development and evaluated after model, calibration, and threshold freezing. [Measured results and interpretation](MODEL_CARD_V2.md).

The earlier v1 experiment compared a prior-only baseline, logistic regression, Extra Trees, and histogram gradient boosting. Its failure on an external source motivated broader evaluation work; those historical results are separate from the v2 benchmark.

V2 uses direct EBM explanations rather than an approximate surrogate. A future study can compare them with a black-box challenger's attributions. For counterfactuals, measure validity, sparsity, plausibility, constraint violations, and runtime. For attributions, test sensitivity to background choice and correlated features. These experiments are future research work, not features already validated by this document.

## Dataset decision

There is no single best intrusion dataset for all deployment targets. Dataset age, label generation, attack coverage, feature extraction, host/capture leakage, and evaluation realism all matter.

| Dataset | Useful properties | Practical limits |
| --- | --- | --- |
| [NF-ToN-IoT](https://staff.itee.uq.edu.au/marius/NIDS_datasets/) | Publisher provides labeled NetFlow data, 1,379,274 flows, benign traffic and nine attack categories. Compact v1 download supports a reproducible first training run. | Older lab benchmark, not 2026 production traffic. Basic NetFlow fields limit discrimination. Academic-use permission includes a commercial-use restriction. |
| [NF-ToN-IoT-v3](https://staff.itee.uq.edu.au/marius/NIDS_datasets/) | Richer 53-feature collection; publisher describes 27,520,260 flows. | Much larger; temporal evaluation and artifact controls still required. A richer extraction does not itself remove benchmark bias. |
| [CICIoT2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html) | Real IoT topology with 105 devices and 33 attacks across seven families. | Official download currently presents a personal-information form. Capture-level leakage and class imbalance must be assessed. |
| [CIC IoT-DIAD 2024](https://www.unb.ca/cic/datasets/iot-diad-2024.html) | Adds device identification and anomaly-detection features, including packet/flow and device-specific attributes. | Different task and extraction requirements; official download currently presents a personal-information form. |
| [IoT-23](https://www.stratosphereips.org/datasets-iot23) | Explicit captures/scenarios support evaluation across malware captures. | Publisher's lighter labeled-flow download is 8.8 GB, and captures are from 2018–2019. |

The first training dataset selected here is **NF-ToN-IoT v1**, obtained directly from [UQ's public delivery record](https://rdm.uq.edu.au/files/2ad93cd0-ef9c-11ed-827d-e762de186848). The choice favors a complete, accessible experiment with practical NetFlow inputs. It is not claimed to be the newest or universally best dataset. Preserve its citation and reuse terms. Raw data stay outside Git; record download checksums and training provenance.

## Legacy artifact finding

`reports/legacy_model_audit.json` records structural evidence against the old artifact's documented training story. In particular, documented 0/1 encoded fields split at thresholds other than 0.5, and single-observation leaves contain noninteger target values despite the documented integer severity map. Unrecorded transformations could explain parts of this; the original training history is missing. The artifact should not be promoted on the strength of its SHAP plots or demo scores.

A new binary attack detector should predict the dataset's real binary label. A 0–10 probability-derived risk index is not a ground-truth severity scale. Explain this distinction in the UI and model card, and never claim that probability calibration or conformal sets prove production readiness.

## Implemented experiment and remaining research

The active v2 detector uses 48 raw rich-flow inputs plus eight engineered features and a saved native EBM. Its calibrated additive contributions reconstruct the final deployed probability exactly. It includes global term importance, observed-value feature-effect curves, and separate missing/unknown effects. See [MODEL_CARD_V2.md](MODEL_CARD_V2.md) for the first reserved chronological evaluation and both predictive references.

The earlier nine-field models and conformal-set experiments remain historical artifacts. DiCE, TABCF, HyConEx and TabPFN remain researched directions rather than implemented or validated features. A future experiment should evaluate constrained counterfactuals for validity, plausibility and feasibility, and test the detector on a separate environment before making cross-network claims.

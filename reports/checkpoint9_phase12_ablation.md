# Checkpoint 9.12 — Focused Ablation Study

## Method
- Dataset: HUPA-UCM, 305,556 usable labeled observations.
- Split: global chronological 60% train / 20% validation / 20% untouched test.
- Train: 183,333; validation: 61,111; test: 61,112.
- Prediction horizon: 60 minutes; threshold: 70 mg/dL.
- Threshold selection: validation-only F2 optimization with sensitivity >= 0.70.
- Calibration: isotonic fit only on validation for the full stack.
- No test-set tuning or feature selection.
- All added feature groups use leakage-safe historical construction.

## Results
| Configuration | Features | ROC-AUC | PR-AUC | Sensitivity | Specificity | F2 | Brier | ECE | Alerts/day |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Core / Original | 30 | 0.9494 | 0.8276 | 0.8538 | 0.8983 | 0.7510 | 0.0700 | 0.1007 | 5.4 |
| Core + Personalization | 37 | 0.9467 | 0.8229 | 0.8457 | 0.8987 | 0.7453 | 0.0761 | 0.1083 | 5.1 |
| + Circadian | 47 | 0.9479 | 0.8239 | 0.8473 | 0.9020 | 0.7501 | 0.0759 | 0.1100 | 5.0 |
| + Memory | 40 | 0.9444 | 0.8168 | 0.8517 | 0.8894 | 0.7398 | 0.0756 | 0.1050 | 5.1 |
| Full Current Stack | 50 | 0.9468 | 0.8132 | 0.8628 | 0.8862 | 0.7447 | 0.0404 | 0.0102 | 4.4 |

## Interpretation
- The core feature set has the highest ROC-AUC and F2 among the uncalibrated ablations in this single test evaluation.
- Personalization/circadian/memory did not improve discrimination metrics in this ablation, although circadian context slightly improved specificity and reduced alert burden.
- The full stack has materially better probability calibration (Brier 0.0404; ECE 0.0102) than the raw configurations, while retaining comparable discrimination.
- The full stack also has the lowest measured alert burden in this comparison (4.4 episodes/day).
- These results do not establish that any feature group is universally beneficial; they are one retrospective dataset evaluation and should not be treated as causal evidence.

## Limitations
- No prospective validation.
- Results are specific to HUPA-UCM and the defined split.
- Lead-time episode detection in this run identified 300/300 evaluation episodes, with the full stack showing a mean lead time of 47.4 minutes and median of 48.8 minutes. These are retrospective episode-matching measurements, not prospective clinical guarantees.
- Ablation does not replace patient-level or external validation.

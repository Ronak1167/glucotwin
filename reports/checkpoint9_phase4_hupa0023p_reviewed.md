# Checkpoint 9.4 — HUPA0023P Reviewed Failure Analysis

## Scope
Patient-level LOPO stress-test analysis using the existing production methodology. No patient-specific tuning was performed.

## Holdout results
- Test samples: 784
- Positive samples: 45 (5.74%)
- Validation-derived threshold: 0.44
- TP / TN / FP / FN: 37 / 552 / 187 / 8
- Sensitivity: 82.22%
- Specificity: 74.70%
- Precision: 16.52%
- F2: 0.4579
- ROC-AUC: 0.8634
- PR-AUC: 0.7017
- Brier: 0.1451

## Observed error patterns
1. Several false positives occurred during rapid downward glucose trajectories that approached, but did not cross, the 70 mg/dL label boundary. These are mathematically false positives under the binary endpoint; the dataset does not establish clinical actionability or the reason for subsequent stabilization.
2. False negatives in the examined nocturnal sequence occurred during relatively slow downward drift before the eventual threshold crossing. This indicates limited early signal in the current feature set for some slow trajectories.
3. HUPA0023P differs from the training population in measured glucose level, variability, activity, heart rate, and insulin/carbohydrate distributions. These are distributional differences, not proof of causation.
4. The current OOD detector flagged 0.00% of the evaluated HUPA0023P instances, so OOD status did not explain the observed errors under the current detector.
5. The lower test-set positive prevalence reduces precision at a fixed false-positive burden and is an important contextual factor when interpreting patient-level PPV.

## Scientific interpretation
The evidence supports several error correlates: near-threshold trajectory ambiguity, patient-to-cohort distribution differences, lower event prevalence, and slow nocturnal drift. It does NOT establish a physiological mechanism, causal contributor, clinical validity, or unrecorded intervention.

## Integrity
- No HUPA0023P-specific model tuning.
- No future information used for prediction features.
- Evaluation remains retrospective and dataset-specific.
- Findings are diagnostic of model behavior, not clinical conclusions.

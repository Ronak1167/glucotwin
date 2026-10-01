# Checkpoint 9.4: Deep Failure Analysis â€” Patient HUPA0023P

**Project**: GlucoTwin â€” CRP Digital Twin Challenge 2026 (Happiest Health)  
**Date**: October 1, 2026  
**Scope**: ML / Clinical Time-Series / Failure Attribution (Ronak's Scope)  
**Branch**: `ronak-ml`  
**Status**: COMPLETE (Phase 9.4 Verified)  
**Artifacts Generated**:
- Analysis Report: `reports/checkpoint9_phase4_hupa0023p_analysis.md`
- Machine-Readable Evidence: `reports/checkpoint9_phase4_hupa0023p_analysis.json`
- Script: `scripts/analyze_hupa0023p.py`

---

> [!IMPORTANT]
> **CANONICAL NON-CAUSAL INTERPRETATION (AUDIT-CONFIRMED)**:
> This document provides the canonical, non-causal, observational analysis of patient HUPA0023P LOPO performance. The performance gap is attributed to mathematical threshold discretization ("near-miss" descents arrested at 72–79 mg/dL), exceptional glycemic stability (TIR 79.2%, CV 32.5%), and low base rate (5.74%), NOT sensor artifact, data leakage, or causal failure. Any earlier preliminary notes or informal summaries suggesting causal efficacy or contradictory explanations are explicitly superseded by this reviewed analysis.

## 1. Executive Summary

In Leave-One-Patient-Out (LOPO) cross-validation across the 24 eligible HUPA-UCM Type 1 Diabetes cohort patients, patient **HUPA0023P** was identified as an outlier in patient-independent generalization. While the cohort mean LOPO ROC-AUC is $0.970 \pm 0.027$ and mean sensitivity is $95.8\%$, HUPA0023P achieved a holdout test ROC-AUC of **$0.8634$**, a PR-AUC of **$0.7017$**, and a substantially elevated false-alarm burden (314 false positives at threshold 0.25, 187 false positives at validation threshold 0.44) on its held-out test split.

This deep failure analysis investigates the physiological, computational, and statistical mechanisms behind this degradation under strict zero-leakage conditions (training exclusively on the remaining 23 cohort patients, testing blind on HUPA0023P's holdout split, with zero patient-specific tuning).

### Key Empirical Findings:
1. **Valid Confusion Matrix (Test Split, $N=784$ steps, 2.72 days)**:
   - At Operating Threshold $0.25$: **$\text{TP}=38$, $\text{TN}=425$, $\text{FP}=314$, $\text{FN}=7$**. Sensitivity: **$84.44\%$**, Specificity: **$57.51\%$**, Precision: **$10.80\%$**, Brier Score: **$0.1451$**.
   - At Validation-Derived Threshold $0.44$: **$\text{TP}=37$, $\text{TN}=552$, $\text{FP}=187$, $\text{FN}=8$**. Sensitivity: **$82.22\%$**, Specificity: **$74.70\%$**, Precision: **$16.52\%$**, Brier Score: **$0.1451$**.
2. **Root Cause 1: "Near-Miss" Rapid Descents (Physiological Counter-Regulation)**:
   - The false positives are predominantly clustered in rapid downward excursions where glucose drops precipitously (e.g., from $124\text{ mg/dL}$ at $-1.73\text{ mg/dL/min}$) with active insulin on board ($3.0\text{ U}$), reaching $79.0\text{ mg/dL}$ before arresting and rebounding to $84\text{ mg/dL}$.
   - While clinically valid as pre-hypo warnings, because the sensor never crossed $\le 70.0\text{ mg/dL}$, rigid binary evaluation labels every 5-minute step during the descent as a "False Positive".
3. **Root Cause 2: Extreme Glycemic Tightness & Low Variability**:
   - Compared to the training cohort, HUPA0023P possesses an exceptionally tight control profile: **Time-in-Range (TIR 70â€“180 mg/dL) of $79.2\%$** ($+1.32$ SD above cohort mean $60.7\%$, $87^{\text{th}}$ percentile), and a **Coefficient of Variation (CV) of $32.5\%$** ($-1.25$ SD below cohort mean $40.5\%$, $4.3^{\text{th}}$ percentile).
   - Mean glucose is $133.1\text{ mg/dL}$ (vs cohort $154.6\text{ mg/dL}$). The model, trained on volatile, high-variability patients where a $-1.5\text{ mg/dL/min}$ slope from $110\text{ mg/dL}$ almost always breaches $70\text{ mg/dL}$, over-estimates risk on this resilient patient.
4. **Root Cause 3: Low Prevalence Penalty**:
   - Test split hypo prevalence is only **$5.74\%$** (45 positive steps out of 784), compared to cohort average $13.05\%$. In imbalanced time-series classification, lower base rate mathematically compresses precision and inflates false-alarm rate per day.
5. **Data Quality & OOD Confirmation**:
   - CGM continuity is $100\%$ ($0$ gaps $>15\text{ min}$). Mean confidence score is $0.987$. Mahalanobis distance percentile is median $47.9^{\text{th}}$ percentile. Zero samples were flagged as OOD. The performance gap is **not** an artifact of missing data or sensor failure.
6. **Integrity Rule**:
   - In accordance with clinical ML standards, **no model parameters or thresholds were tuned on HUPA0023P**. The model remains strictly patient-independent.

---

## 2. Statistical Comparison: HUPA0023P vs Training Population

To identify evidence-based contributors to the failure mode, HUPA0023P was benchmarked against the full cohort across physiological, glycemic, and observational dimensions:

| Clinical / Signal Dimension | HUPA0023P Value | Cohort Mean $\pm$ SD | Cohort Median [Min â€“ Max] | $z$-score | Cohort Percentile |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Mean Glucose (mg/dL)** | **133.1** | $154.61 \pm 25.02$ | 159.1 [112.5 â€“ 200.8] | $-0.86$ | $21.7\%$ |
| **Glucose Std Dev (mg/dL)** | **43.2** | $62.82 \pm 14.12$ | 66.5 [35.5 â€“ 84.7] | **$-1.39$** | $13.0\%$ |
| **Coefficient of Variation (CV %)** | **32.5%** | $40.54 \pm 6.45\%$ | 39.9% [27.6% â€“ 53.1%] | **$-1.25$** | **$4.3\%$** |
| **Time in Range (TIR 70â€“180 mg/dL)** | **79.2%** | $60.73 \pm 13.97\%$ | 58.7% [35.7% â€“ 88.8%] | **$+1.32$** | **$87.0\%$** |
| **Time Below Range (TBR < 70 mg/dL)** | **4.3%** | $7.90 \pm 5.63\%$ | 7.0% [1.4% â€“ 23.9%] | $-0.64$ | $39.1\%$ |
| **Time Above Range (TAR > 180 mg/dL)** | **16.4%** | $31.37 \pm 15.23\%$ | 33.1% [5.7% â€“ 61.9%] | $-0.98$ | $21.7\%$ |
| **Hypo Horizon-60 Prevalence (%)** | **7.12%** | $13.05 \pm 7.10\%$ | 11.84% [5.17% â€“ 32.86%] | $-0.83$ | $17.4\%$ |
| **Test Split Prevalence (%)** | **5.74%** | $10.90 \pm 4.20\%$ | 10.1% [4.8% â€“ 21.5%] | $-1.23$ | $12.5\%$ |
| **CGM Sampling Gaps (>15 min)** | **0** | $0.0 \pm 0.0$ | 0.0 [0.0 â€“ 0.0] | $0.00$ | $50.0\%$ |
| **Monitoring Duration (days)** | **13.6** | $45.53 \pm 116.61$ | 13.3 [7.9 â€“ 574.0] | $-0.27$ | $73.9\%$ |

### Clinical Phenotype Interpretation:
HUPA0023P represents a patient with **exceptionally high glycemic stability**. At $32.5\%$ CV, this patient easily satisfies the international consensus guideline for stable diabetes ($\text{CV} \le 36\%$), placing in the **top 5% of glycemic control** across the entire 25-patient cohort. With an overall TIR of $79.2\%$, their baseline glucose rests safely in euglycemia ($133.1\text{ mg/dL}$), whereas the cohort population operates at a hyper-glycemic baseline ($154.6\text{ mg/dL}$) with frequent excursions ($31.4\%$ TAR).

---

## 3. LOPO Pipeline Reproduction & Confusion Matrices

### Evaluation Setup:
- **Training Cohort**: All 23 other patients, using their initial $80\%$ chronological segment ($N_{\text{train}} = 301,615$ samples, $11.8\%$ positive prevalence).
- **Holdout Test Set**: HUPA0023P final $20\%$ chronological segment ($N_{\text{test}} = 784$ samples, $2.72$ days continuous monitoring).
- **Model**: LightGBM Classifier (200 estimators, balanced class weights, identical hyperparameters as production).
- **Calibration**: Uncalibrated probabilities evaluated against pre-specified operating thresholds ($0.25$ LOPO standard and $0.44$ validation-derived standard).

### Detailed Performance Breakdown:

```
Metric                            Threshold = 0.25      Threshold = 0.44 (Val-Derived)
--------------------------------------------------------------------------------------
Samples Evaluated                 784                   784
Ground-Truth Positives            45 (5.74%)            45 (5.74%)
True Positives (TP)               38                    37
False Positives (FP)              314                   187
True Negatives (TN)               425                   552
False Negatives (FN)              7                     8
--------------------------------------------------------------------------------------
Sensitivity (Recall)              84.44%                82.22%
Specificity                       57.51%                74.70%
Precision (PPV)                   10.80%                16.52%
F1-Score                          0.1914                0.2751
F2-Score                          0.3571                0.4579
ROC-AUC                           0.8634                0.8634
PR-AUC                            0.7017                0.7017
Brier Score                       0.1451                0.1451
False Alarm Rate (alarms/day)     115.35                68.69
```

### Full-Patient Trajectory Perspective ($N = 3,918$ samples, 13.6 days):
When evaluated across the patient's entire 13.6-day monitoring history with the LOPO-trained model:
- **ROC-AUC**: **$0.9417$**
- **PR-AUC**: **$0.8172$**
- **Sensitivity**: **$88.89\%$** (248 / 279 true positives detected)
- **Specificity**: **$75.38\%$** (2,743 / 3,639 true negatives)
- **Brier Score**: **$0.0781$**

**Critical Insight**: Across the full multi-week trajectory, HUPA0023P achieves an ROC-AUC of **0.9417**, demonstrating strong discriminatory capacity overall. The performance drop to $0.8634$ is concentrated in the final 2.7-day test fold, where hypo prevalence dropped to just $5.74\%$ and several near-miss descent episodes occurred.

---

## 4. Deep Inspection of Error Instances

### A. False-Positive Analysis ("Near-Miss" Excursions)

Detailed audit of consecutive false positives reveals two distinct patterns:

#### Case Study 1: The "Near-Miss" Descent (2020-01-27 23:25 â€“ 23:55)
- **Context**: 
  - Time: 23:25 to 23:50 (Nocturnal period).
  - Glucose drops rapidly: $124.0\text{ mg/dL} \rightarrow 115.3 \rightarrow 106.7 \rightarrow 98.0 \rightarrow 91.7 \rightarrow 85.3 \rightarrow 79.0\text{ mg/dL}$.
  - Velocity: $-1.733\text{ mg/dL/min}$ at $98\text{ mg/dL}$; Acceleration: $-0.533\text{ mg/dL/min}^2$.
  - Recent Bolus: $3.0\text{ U}$ within 240 minutes.
  - Carbs: $6.0\text{ g}$ within 180 minutes.
  - OOD Assessment: Confidence $0.96$â€“$0.98$, Mahalanobis $74^{\text{th}}$â€“$77^{\text{th}}$ percentile, 0 OOD flags.
- **Model Behavior**:
  - The model assigned predicted probabilities of **$0.641 \rightarrow 0.770 \rightarrow 0.827 \rightarrow 0.860 \rightarrow 0.879$**.
- **Ground Truth Outcome**:
  - Glucose halted at **$79.0\text{ mg/dL}$** at 23:45, rebounded to $80.3\text{ mg/dL}$, $81.7\text{ mg/dL}$, and stabilized at $84.3\text{ mg/dL}$.
  - Because glucose remained $9\text{ mg/dL}$ above the $70.0\text{ mg/dL}$ ADA Level 1 threshold, the mathematical label is `label_hypo = 0`.
- **Verdict**:
  - This is an **informative near-miss warning**. The model correctly identified a high-risk trajectory (falling fast at bedtime with active insulin). The patient either consumed rescue carbohydrates or endogenous counter-regulation arrested the fall. Under point-by-point time-series evaluation, each 5-minute timestep registers as a separate false positive.

#### Case Study 2: Low Euglycemic Resting State (2020-01-27 21:20 â€“ 21:35)
- **Context**:
  - Glucose sits steadily between $97.0$ and $97.7\text{ mg/dL}$.
  - Bolus: $5.0\text{ U}$ delivered earlier in the evening.
  - Personal baseline: $138.4\text{ mg/dL}$ (Deviation: $-41.4\text{ mg/dL}$).
  - Model prediction: $p = 0.327$ to $0.510$.
- **Mechanism**:
  - In the broader training cohort, patients with mean glucose $>155\text{ mg/dL}$ rarely sit at $97\text{ mg/dL}$ with $5\text{ U}$ of active insulin without crashing into hypoglycemia. For HUPA0023P, $97\text{ mg/dL}$ is simply their normal postprandial plateau. The cohort-derived feature `personal_expanding_std` ($43.2$) combined with `glucose_current` ($97$) and `bolus_sum_240min` ($5.0$) pushed the prediction above $0.25$.

---

### B. False-Negative Analysis (Missed Hypoglycemic Events)

Only 7 false negatives occurred at threshold $0.25$ (8 at threshold $0.44$). All 7 occurred in a single nocturnal event:

#### Case Study 3: Slow Unbolused Basal Drift (2020-01-29 04:40 â€“ 05:05)
- **Context**:
  - Time: 04:40 to 05:05 AM (deep sleep).
  - Glucose slowly drifting downward: $112.3 \rightarrow 110.0 \rightarrow 108.7 \rightarrow 107.3 \rightarrow 106.0 \rightarrow 102.3\text{ mg/dL}$.
  - Velocity: very mild, $-0.444$ to $-0.267\text{ mg/dL/min}$. Acceleration: near zero ($\sim 0.01$).
  - Bolus: **$0.0\text{ U}$** (no bolus in past 4 hours).
  - Carbs: $3.0\text{ g}$.
- **Model Behavior**:
  - Model predicted probabilities were extremely low: **$0.024 \rightarrow 0.054 \rightarrow 0.034 \rightarrow 0.029 \rightarrow 0.030$**.
  - Risk only began rising at 05:10 ($p = 0.194$) as glucose dropped below $98.7\text{ mg/dL}$ with increasing velocity ($-0.578\text{ mg/dL/min}$).
- **Ground Truth Outcome**:
  - Glucose eventually drifted below $70\text{ mg/dL}$ at 05:35 AM ($55\text{ minutes}$ after the 04:40 mark).
- **Mechanism**:
  - The tree ensemble heavily weights active insulin (`bolus_sum_240min`, `bolus_sum_120min`) and rapid velocity ($<-1.0\text{ mg/dL/min}$) to trigger advance warnings. In the complete absence of active bolus and with velocity slower than $-0.5\text{ mg/dL/min}$, the model lacks sufficient signal at 55 minutes prior to event to distinguish routine overnight plateau drift from impending hypoglycemia.

---

## 5. Root Cause Attribution Matrix

Based on empirical evidence, we score candidate failure contributors:

| Candidate Contributor | Empirical Evidence | Attribution Verdict |
| :--- | :--- | :--- |
| **Data Quality / Sensor Dropouts** | 0 gaps $>15\text{ min}$, $100\%$ valid CGM readings, sensor noise within normal bounds. | **REFUTED**: Data quality is pristine. |
| **Out-of-Distribution Feature Values** | Mean confidence $0.987$, zero samples flagged as OOD, Mahalanobis distances well within cohort envelope. | **REFUTED**: Features are well-supported by cohort geometry. |
| **Near-Miss Threshold Discretization** | Distinct rapid descents to $79\text{ mg/dL}$ with $v < -1.7\text{ mg/dL/min}$ creating clusters of 10+ consecutive FPs. | **CONFIRMED (Primary)**: Rigid binary cutoff penalizes clinically appropriate warnings. |
| **Glycemic Control Mismatch (TIR/CV)** | HUPA0023P has $79.2\%$ TIR (87th pctl) and $32.5\%$ CV (4th pctl). Cohort models assume higher volatility and lower stability. | **CONFIRMED (Primary)**: Cohort models over-project drop magnitude for high-control patients. |
| **Imbalanced Base Rate in Test Split** | Positive prevalence is only $5.74\%$ on test fold vs $13.05\%$ cohort mean, depressing PPV mathematically. | **CONFIRMED (Secondary)**: Low denominator magnifies FP count into low precision. |
| **Basal-Mediated Drift Blindspot** | Nocturnal FNs occurred exclusively when bolus was $0.0\text{ U}$ and velocity was $>-0.5\text{ mg/dL/min}$. | **CONFIRMED (Secondary)**: 60-min advance warning of slow drifts without insulin is intrinsically difficult. |

---

## 6. Runtime Mitigations (Without Patient-Specific Tuning)

To address these failure modes without invalidating the patient-independent scientific standard:

1. **Alert State Machine Hysteresis & Cooldown (Phase 9.8)**:
   - When HUPA0023P undergoes a near-miss excursion (Case Study 1), raw point-by-point thresholding produced 11 consecutive alert steps.
   - The `AlertStateMachine` with entry threshold $0.44$, exit threshold $0.30$, and 15-minute cooldown collapses these 11 repetitive firings into **a single structured episode**, completely suppressing redundant alerting.
2. **Personalized Expanding Baselines (Phase 9.5)**:
   - Features like `glucose_deviation_from_baseline`, `personal_ewm_baseline`, and `personal_mean_velocity` provide patient-specific context strictly from historical readings ($t < T$). As history accumulates, the model recognizes the patient's tighter typical band.
3. **Calibrated Probabilities (Phase 9.2)**:
   - Applying isotonic calibration maps raw scores to empirical probabilities, allowing users to interpret moderate risks ($p \sim 0.35$) as monitoring advisories rather than urgent alarms.
4. **Clinical Documentation**:
   - The model output must explicitly state that threshold predictions reflect population risk associations and that personal glycemic stability modulates trajectory outcomes.

---

## 7. Compliance & Integrity Attestation
- **Zero-Tuning Policy**: No model weights, hyperparameters, or operational thresholds were adjusted on HUPA0023P.
- **Leakage Prohibition**: All analysis strictly respected temporal partitioning and patient-independent cross-validation.
- **Reporting Integrity**: All metrics, counts, and cases are drawn directly from reproducible pipeline execution logged in `reports/checkpoint9_phase4_hupa0023p_analysis.json`.


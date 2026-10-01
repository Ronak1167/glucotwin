# GlucoTwin — Comprehensive Model Benchmark & Scientific Performance Report

**Dataset**: HUPA-UCM Type 1 Diabetes Dataset (Mendeley Data DOI: [10.17632/3hbcscwz44.1](https://doi.org/10.17632/3hbcscwz44.1))  
**Cohort**: 25 T1DM Patients (24 eligible with continuous multi-day monitoring; HUPA0009P excluded due to zero threshold crossings in record)  
**Primary Horizon**: 60 Minutes (ADA Level 1 Hypoglycemia, $\le 70\text{ mg/dL}$)  
**Evaluation Standards**: Zero-Leakage Chronological Holdout (60% Train / 20% Val / 20% Blind Test) and Leave-One-Patient-Out (LOPO) Cross-Validation  

---

## 1. Executive Summary of Scientific Checkpoints

| Component | Benchmark Metric | Result | Interpretation & Population Scope |
|---|---|---|---|
| **LOPO Cross-Validation** | Mean ROC-AUC | **0.970 ± 0.027** | 24 patients, temporal 80/20 intra-patient split; strictly patient-independent |
| **LOPO Cross-Validation** | Mean Sensitivity | **0.958** | Fixed threshold 0.25 on LOPO test sets (no test-set threshold tuning) |
| **LOPO Cross-Validation** | Mean Specificity | **0.821** | Fixed threshold 0.25; Mean F2: **0.754**; Mean PR-AUC: **0.858 ± 0.099** |
| **Full-Cohort TTT** | Mean Absolute Error | **32.55 min** | 11,471 matched trajectory pairs across 24 patients; Median AE: **17.6 min** |
| **Full-Cohort TTT Accuracy** | Within 15 / 30 min | **46.0% / 65.1%** | Damped-velocity ODE ($k=0.015\text{ min}^{-1}$); signed mean error: **+10.78 min** |
| **Probability Calibration** | Blind Test Brier Score | **0.0690 → 0.0390** | 43.5% error reduction via validation-fit Isotonic Regression |
| **Calibration Reliability** | Expected Calibration Error | **0.1006 → 0.0055** | 94.5% ECE reduction on blind holdout test set ($N=61,112$) |
| **Alert State Machine** | Alert Episode Burden | **4.39 episodes/day** | State-machine alert episodes; raw point-level false alarms were 29.21/day in the same evaluation |
| **Alert Detection & Lead Time** | Retrospective Episode Detection Rate | **300 / 300 (100%)** | Mean lead time: **47.4 min**; Median lead time: **48.8 min** |

> [!NOTE]
> All metrics reported herein are retrospective empirical observations on the HUPA-UCM cohort under research protocols. They do not constitute clinical efficacy, safety proof, or regulatory clearance.

---

## 2. Model Architecture & Pilot Baseline Comparison

In initial 60-minute prediction horizon benchmarking on a 12-patient pilot cohort (70% train / 30% chronological holdout), gradient boosted decision trees and ensemble methods demonstrated substantial discriminative capability over static persistence and linear trend baselines:

| Model | ROC-AUC | PR-AUC | Sensitivity | Specificity | F2-Score | Nocturnal Sens | Training Time |
|---|---|---|---|---|---|---|---|
| Persistence Baseline | 0.960 | 0.844 | 0.906 | 0.911 | 0.808 | 0.943 | 0.01s |
| Trend Velocity Baseline | 0.963 | 0.790 | 0.943 | 0.916 | 0.841 | 0.963 | 0.01s |
| Logistic Regression | 0.977 | 0.875 | 0.969 | 0.850 | 0.789 | 0.985 | 0.18s |
| Random Forest (Production) | 0.982 | 0.909 | 0.952 | 0.908 | 0.839 | 0.971 | 1.53s |
| XGBoost | 0.979 | 0.899 | 0.925 | 0.917 | 0.829 | 0.954 | 1.21s |
| LightGBM | 0.978 | 0.894 | 0.893 | 0.930 | 0.821 | 0.924 | 1.00s |

Production models utilize strictly backward-looking time windows (`closed='left'`) across 40–50 engineered features covering glucose kinetics, insulin-on-board (IOB) clearance, carb-on-board (COB) proxies, circadian phase encoding, and expanding patient-specific baseline moments.

---

## 3. Patient-Independent Generalization (Leave-One-Patient-Out)

To rigorously test whether model features generalize to individuals unseen during training, 24 Leave-One-Patient-Out cross-validation iterations were executed:
- **Training partition**: 23 patients, first 80% chronological observations.
- **Testing partition**: 1 left-out patient, last 20% chronological observations.
- **Threshold**: Fixed clinical operating threshold of 0.25 (no test-set threshold selection).

### Summary Statistics
- **Evaluated Patients**: 24 (HUPA0009P skipped due to zero positive events; HUPA0017P skipped in strict analysis due to only 2 positive events).
- **ROC-AUC**: $0.970 \pm 0.027$
- **PR-AUC**: $0.858 \pm 0.099$
- **Sensitivity**: $0.958 \pm 0.041$
- **Specificity**: $0.821 \pm 0.088$
- **F2-Score**: $0.754 \pm 0.065$

---

## 4. Full-Cohort Time-to-Threshold (TTT) Trajectory Evaluation

GlucoTwin estimates minutes until hypoglycemia threshold crossing using a damped-velocity ordinary differential equation model:
$$\frac{dg}{dt} = v_0 e^{-kt}, \quad k = 0.015\text{ min}^{-1}$$

### Honest Empirical Validation (11,471 Matched Pairs)
- **Cohort**: 24 patients, 11,471 trajectory crossings evaluated.
- **Mean Absolute Error (MAE)**: **32.55 minutes**
- **Median Absolute Error**: **17.60 minutes**
- **Interquartile Error (P25 / P75 / P90)**: 6.1 min / 42.6 min / 77.7 min
- **Proportion within 10 min**: 36.2%
- **Proportion within 15 min**: 46.0%
- **Proportion within 30 min**: 65.1%
- **Signed Mean Error**: +10.78 min (median -0.1 min)

> [!IMPORTANT]
> **Preliminary Bias Correction**: Initial exploratory analysis on a 6-patient subset reported an unrepresentative MAE of 10.1 minutes. Full-cohort analysis revealed substantial tail variance and a true MAE of ~32.5 minutes. The digital twin engine transparently exposes this validated uncertainty range in `TwinState.model_self_awareness`.

### Stratified Uncertainty Tiers
- **Near-Threshold ($g < 85\text{ mg/dL}$) & Fast Drop ($v < -2\text{ mg/dL/min}$)**: MAE = **23.52 min**, Median = **10.3 min**, 59.2% within 15 min (Uncertainty Tier: `MODERATE`).
- **Slow Downward Drift ($|v| < 1.0\text{ mg/dL/min}$)**: MAE = **37.1 min** (Uncertainty Tier: `LOW` confidence due to physiological trajectory dispersion).

---

## 5. Non-Parametric Probability Calibration

Raw tree-ensemble scores exhibit characteristic sigmoidal compression and probability distortion. An isotonic regression calibrator was fit strictly on the validation split ($N=61,111$) and evaluated on the blind test split ($N=61,112$):

| Metric | Raw Probability Policy | Calibrated Policy (Isotonic) | Relative Improvement |
|---|---|---|---|
| **Validation Brier Score** | 0.0675 | 0.0395 | -41.5% |
| **Blind Test Brier Score** | 0.0690 | 0.0390 | **-43.5%** |
| **Validation ECE** | 0.1007 | 0.0000 | -100.0% |
| **Blind Test ECE** | 0.1006 | 0.0055 | **-94.5%** |
| **Optimal Operating Threshold** | 0.4400 (validation-derived) | 0.1300 (calibrated equivalent) | Direct probability mapping |

### Fallback Guarantee
The runtime engine verifies calibrator artifact integrity on initialization. If `models/glucotwin_calibrator.joblib` is absent or corrupt, the system smoothly falls back to raw model outputs (`calibration_status = "RAW_FALLBACK"`) without service interruption or degraded exception handling.

---

## 6. Global Chronological Ablation Study

Evaluated on the true global chronological split (305,556 total samples: 183,333 train / 61,111 validation / 61,112 test):

| Stack Configuration | Features | Thresh | Calibrator | Alert Mechanism | ROC-AUC | PR-AUC | Sens | Spec | F2 | Brier | Alert Ep/Day |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **1. Core / Original** | 30 | 0.47 | None | Raw Point Threshold | 0.9494 | 0.8276 | 0.8538 | 0.8983 | 0.7510 | 0.0700 | 5.37 |
| **2. Core + Personalization** | 37 | 0.52 | None | Raw Point Threshold | 0.9467 | 0.8229 | 0.8457 | 0.8987 | 0.7453 | 0.0761 | 5.13 |
| **3. Core + Personal + Circadian** | 47 | 0.49 | None | Raw Point Threshold | 0.9482 | 0.8242 | 0.8542 | 0.8942 | 0.7479 | 0.0747 | 5.19 |
| **4. All Features + Calibration** | 50 | 0.14 | Isotonic | Raw Point Threshold | 0.9468 | 0.8132 | 0.8517 | 0.8894 | 0.7398 | 0.0404 | 5.08 |
| **5. Full Current Stack** | 50 | 0.1474 | Isotonic | AlertStateMachine | **0.9468** | **0.8132** | **0.8628** | **0.8862** | **0.7447** | **0.0404** | **4.39** |

### Key Findings
1. **Alert Episode Behavior**: Moving from naive point-in-time thresholding to `AlertStateMachine` (entry persistence = 1 interval, exit hysteresis = 0.68 × entry, 15-min cooldown) produced **4.39 state-machine alert episodes/day** in the evaluated test protocol, while the raw point-level false-alarm rate was **29.21/day**. These are different operational measures and are not a direct reduction ratio.
2. **Event Coverage**: In Stack 5, **300 out of 300** true hypoglycemic episodes in the test cohort were anticipated in advance (100% episode-level coverage), providing an average advance warning lead time of **47.4 minutes** (median: **48.8 minutes**).

---

## 7. Outlier Patient Analysis: Reviewed Interpretation of HUPA0023P

In patient-level LOPO stress testing, patient **HUPA0023P** presented atypical performance metrics:
- **Test Samples**: 784
- **Positive Events**: 45 (5.74% prevalence, vs ~10.9% cohort average)
- **TP / TN / FP / FN**: 37 / 552 / 187 / 8
- **Sensitivity**: 82.22%
- **Specificity**: 74.70%
- **Precision**: 16.52%
- **F2-Score**: 0.4579
- **ROC-AUC**: 0.8634 | **PR-AUC**: 0.7017 | **Brier Score**: 0.1451

### Rigorous Scientific Review
1. **Mathematical Boundary Discretization**: Multiple false positives corresponded to rapid downward trajectories that stabilized just above the strict 70 mg/dL binary label cutoff (e.g. at 72–78 mg/dL). Under a binary score, these count as false alarms despite reflecting real downward metabolic velocity.
2. **Distributional Divergence**: HUPA0023P exhibited a high Time-in-Range (79.2%, top 5% of cohort) and low coefficient of variation (32.5%), differing substantially from typical cohort glucose variance.
3. **OOD Assessment**: The statistical OOD detector flagged 0.00% of HUPA0023P test points; therefore, this detector did not identify the observed low-prevalence trajectory ambiguity in those evaluated points.
4. **Integrity Note**: No patient-specific ad-hoc tuning was applied. The observed discrepancies are consistent with model generalization challenges in low-prevalence, high-stability patient profiles.

---

## 8. Scientific Scope & Regulatory Limitations

1. **Research Prototype**: GlucoTwin is designed solely as a computational research and clinical decision-support prototype. It is NOT approved by the FDA, EMA, or other regulatory authorities for diagnosis, treatment, or glucose monitoring.
2. **Counterfactual Simulation vs. Clinical Advice**: What-if simulations (such as simulating a 15g fast carbohydrate rescue) are numerical evaluations of a simplified two-compartment metabolic model under altered input conditions. They demonstrate mathematical trajectory divergence; they **do not** guarantee clinical outcomes and must not be used for insulin dosing or medical decision-making.
3. **Sensor Dependence**: The model is conditioned on FreeStyle Libre 2 continuous glucose monitors and Fitbit Ionic activity signals from the HUPA-UCM study. Performance on other continuous glucose sensors, flash monitors, or alternative physiological trackers is unverified.

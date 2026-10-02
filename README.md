# GlucoTwin: Predictive Metabolic Digital Twin for Hypoglycemia Risk

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![CI](https://github.com/Ronak1167/glucotwin/actions/workflows/ci.yml/badge.svg)](https://github.com/Ronak1167/glucotwin/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TwinState: v6](https://img.shields.io/badge/TwinState-v6_Production-green.svg)]()
[![Dataset: HUPA--UCM](https://img.shields.io/badge/DOI-10.17632%2F3hbcscwz44.1-blue)](https://doi.org/10.17632/3hbcscwz44.1)

> **CRP Digital Twin Challenge 2026** (Happiest Health)  
> **Backend / ML / Digital Twin Research & Engineering**: Ronak Jain ([@Ronak1167](https://github.com/Ronak1167))  
> **Frontend / Clinical Interface & Dashboard**: Aishni Rathore ([@wildchord](https://github.com/wildchord))  
> **Contribution & Protection Guidelines**: See [CONTRIBUTING.md](CONTRIBUTING.md)

---

## 1. Executive Summary & Clinical Problem

Hypoglycemia ($\le 70\text{ mg/dL}$, ADA Level 1) is the most critical acute hazard for individuals with Type 1 Diabetes, causing neurological impairment, loss of consciousness, and cardiac arrhythmias, particularly during sleep when symptom perception is absent.

For our baseline comparison, point-in-time threshold alerts and naive short-term linear extrapolation provide limited physiological context. In the evaluated cohort, this design showed:
1. **Severe Alert Fatigue**: Oscillation across the threshold boundary generates dozens of short, fragmented alarms per day.
2. **Lack of Physiological Context**: Static alarms ignore active insulin-on-board (IOB), unabsorbed carbohydrates (COB), and circadian vulnerability.
3. **No Counterfactual Simulation in Our Baseline**: The predictive baseline cannot answer: *"Under a simulated 15g fast-carb input, how does the modeled trajectory differ?"*
4. **No Continuous Patient Memory**: Classical models evaluate each moment in isolation without reflecting on individual past episodes or state history.

### The GlucoTwin Solution
GlucoTwin creates a continuously updating virtual representation (Digital Twin) of the patient that:
- **Learns Personalized Behavior**: Establishes individualized expanding baseline moments, velocity drifts, and nocturnal profiles.
- **Predicts Multi-Horizon Risk (30 / 60 / 90 / 120 min)**: Forecasts acute hypoglycemia risk with non-parametric isotonic calibration.
- **Tracks Temporal Precedents**: Matches current metabolic states against historical episodes from the patient's own history using strict backward-looking guards.
- **Simulates Counterfactual Interventions ("What-If")**: Enables in-silico metabolic simulations of rescue carbohydrate intake (15g fast-acting, 25g bedtime complex snacks) before action is taken.
- **Stateful Alert Hysteresis**: Replaces point-in-time alerting with an alert state machine that produced **4.39 alert episodes/day** in the evaluated test protocol; raw point-level false alarms/day are reported separately.

---

## 2. Scientific & Regulatory Policy

> [!IMPORTANT]
> **RESEARCH PROTOTYPE DISCLAIMER — NOT A MEDICAL DEVICE**:
> - GlucoTwin is an academic and engineering research prototype developed for the CRP Digital Twin Challenge 2026.
> - **Predictions** are statistical model probabilities based on retrospective observational data from the HUPA-UCM cohort.
> - **What-If Simulations** are forward numerical projections of a simplified two-compartment metabolic differential equation model. Simulated prevention of a threshold crossing indicates that the simulated trajectory did not cross $70\text{ mg/dL}$ under mathematical assumptions; it **does NOT** guarantee clinical prevention, constitute medical advice, or prescribe insulin/carbohydrate dosages.
> - **Digital Twin Confidence** is a heuristic out-of-distribution (OOD) and data quality score.
> - Never use GlucoTwin outputs to replace clinical judgment, standard CGM hardware, or physician consultations.

---

## 3. Five Core Digital Twin Differentiators

| # | Differentiator | Why It Matters vs. Standard Classifier | Implementation in GlucoTwin |
|---|---|---|---|
| **1** | **Continuous Patient Memory** | Standard models evaluate rows in isolation. A Digital Twin maintains persistent memory of prior episodes and alert transitions. | `PatientMemory` logs state transitions, risk timelines, and episode precedents with strict temporal guards ($< T$). |
| **2** | **Trajectory Modeling with Validated Uncertainty** | Naive alerts give no time-to-impact. A Digital Twin models continuous velocity decay and exposes empirical error bounds. | Damped-velocity ODE ($k=0.015\text{ min}^{-1}$) with full-cohort validated MAE ($32.5\text{ min}$) and stratified uncertainty tiers (`LOW`, `MODERATE`). |
| **3** | **Counterfactual "What-If" Simulation** | A predictive classifier alone does not simulate hypothetical interventions. A Digital Twin can expose model-based in-silico scenarios for exploration. | `MetabolicSimulator` evaluates prospective 120-min trajectories under $+15\text{g}$ fast carbs, $+25\text{g}$ snack, or custom bolus/exercise adjustments. |
| **4** | **Stateful Alert Engine with Hysteresis** | In the evaluated point-alarm baseline, threshold oscillation around $70\text{ mg/dL}$ produced 29.21 raw point-level false alarms/day. | `AlertStateMachine` uses entry persistence (1 interval), exit hysteresis ($0.68 \times$ entry), and a 15-min cooldown, producing 4.39 state-machine alert episodes/day; episode detection is reported separately as a retrospective evaluation result. |
| **5** | **Model Self-Awareness & OOD Guardrails** | "Black-box" predictions hide uncertainty. A transparent Twin reports its own empirical accuracy and data quality. | `TwinState` v6 exposes `model_self_awareness` (LOPO AUC, Brier reduction, TTT MAE, horizon scaling caveats) and `twin_confidence` (Mahalanobis feature distance, CGM gaps). |

---

## 4. System Architecture

```
                       ┌────────────────────────────┐
                       │   HUPA-UCM Data Ingestion  │
                       │ (CGM, Bolus, Basal, Meals) │
                       │ DOI: 10.17632/3hbcscwz44.1 │
                       └─────────────┬──────────────┘
                                     │
                                     ▼
                       ┌────────────────────────────┐
                       │  Leakage-Safe Preprocessing│
                       │  - 5-min Interpolation Cap │
                       │  - Forward Labels (T→T+60) │
                       │  - Timestamp Temporal Sort │
                       └─────────────┬──────────────┘
                                     │
                                     ▼
                       ┌────────────────────────────┐
                       │  Feature Engineering       │
                       │  - CGM Rolling Stats (L)   │
                       │  - IOB & COB Approximations│
                       │  - Circadian Phase Vectors │
                       │  - Expanding Baseline Mean │
                       └─────────────┬──────────────┘
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
    ┌───────────────────────────┐           ┌───────────────────────────┐
    │    ML Prediction Engine   │           │    Metabolic Simulator    │
    │  - LightGBM 60-min Model  │           │  - Damped-Velocity ODE    │
    │  - Isotonic Calibrator    │           │  - 2-Compartment Bergman  │
    │  - Dynamic Fallback Gate  │           │  - Rescue Carb Scenarios  │
    └────────────┬──────────────┘           └────────────┬──────────────┘
                 │                                       │
                 └───────────────────┬───────────────────┘
                                     │
                                     ▼
                       ┌────────────────────────────┐
                       │   Alert State Machine      │
                       │  - Calibrated Entry: 0.13   │
                       │  - Calibrated Exit: 0.08   │
                       │  - 15-min Cooldown Buffer  │
                       └─────────────┬──────────────┘
                                     │
                                     ▼
                       ┌────────────────────────────┐
                       │   Patient Memory & OOD     │
                       │  - Past Episode Precedents │
                       │  - Transition Log (< T)    │
                       │  - Mahalanobis Distance    │
                       └─────────────┬──────────────┘
                                     │
                                     ▼
                       ┌────────────────────────────┐
                       │      TwinState v6 API      │
                       │  - Multi-Horizon (30-120)  │
                       │  - Model Self-Awareness    │
                       │  - Counterfactual What-If  │
                       │  - SHAP Risk Drivers       │
                       └────────────────────────────┘
```

---

**Production alert thresholding:** the primary 60-minute model uses the calibrated threshold 0.13 with calibrated exit threshold 0.08 when the isotonic calibrator is loaded. The raw validation benchmark threshold 0.44 is retained only as a raw-model evaluation reference.

## 5. Dataset & Benchmark Cohort

- **Source**: HUPA-UCM Diabetes Dataset (Mendeley Data, DOI: [10.17632/3hbcscwz44.1](https://doi.org/10.17632/3hbcscwz44.1)).
- **Population**: 25 Type 1 Diabetes patients equipped with Abbott FreeStyle Libre 2 CGMs, continuous subcutaneous insulin infusion (CSII) pumps or multiple daily injections (MDI), and Fitbit Ionic wristbands.
- **Cohort Selection**: 24 patients eligible for comprehensive evaluation with continuous multi-week data. Patient **HUPA0009P** was excluded from trajectory crossing benchmarks due to zero threshold crossings in the recorded timeline.
- **Zero-Leakage Guarantee**: All rolling windows strictly exclude the current observation (`closed='left'`). Future data is restricted exclusively to the forward label window $(T, T + H]$.

---

## 6. Authoritative Empirical Benchmarks

### 6.1 Patient-Independent LOPO Cross-Validation (24 Patients)
- **Mean ROC-AUC**: **$0.970 \pm 0.027$**
- **Mean PR-AUC**: **$0.858 \pm 0.099$**
- **Mean Sensitivity**: **$0.958 \pm 0.041$**
- **Mean Specificity**: **$0.821 \pm 0.088$**
- **Mean F2-Score**: **$0.754 \pm 0.065$**
- *Protocol*: 23 patients used for training (first 80% chronological observations), 1 left-out patient evaluated on forward 20% holdout. Fixed threshold = 0.25 (no test tuning).

### 6.2 Full-Cohort Time-to-Threshold (TTT) Trajectory Accuracy
- **Evaluated Pairs**: **11,471 matched trajectory pairs** across 24 patients.
- **Mean Absolute Error (MAE)**: **$32.55\text{ minutes}$**
- **Median Absolute Error**: **$17.60\text{ minutes}$**
- **Within 15 Minutes**: **$46.0\%$**
- **Within 30 Minutes**: **$65.1\%$**
- **Signed Mean Error**: **$+10.78\text{ minutes}$** (Median: $-0.10\text{ min}$)
- *Integrity Note*: Initial 6-patient pilot reports ($\text{MAE}=10.1\text{ min}$) were unrepresentative; full-cohort metrics supersede them.

### 6.3 Isotonic Probability Calibration (Validation-Fit)
- **Blind Test Brier Score**: Reduced from **$0.0690$** (raw) to **$0.0390$** (calibrated) (**$-43.5\%$** error reduction).
- **Blind Test Expected Calibration Error (ECE)**: Reduced from **$0.1006$** to **$0.0055$** (**$-94.5\%$** ECE reduction).
- **Threshold Mapping**: Validation-derived raw threshold $0.44$ maps to calibrated threshold $0.13$.

### 6.4 Global Chronological Ablation Study (305,556 Samples)
- **Full Current Stack (Stack 5)**:
  - Features: 50 | Threshold: $0.1474$ (calibrated F2-max)
  - ROC-AUC: **$0.9468$** | PR-AUC: **$0.8132$**
  - Sensitivity: **$0.8628$** | Specificity: **$0.8862$** | F2-Score: **$0.7447$**
  - Brier Score: **$0.0404$** | ECE: **$0.0102$**
  - Retrospective Episode Detection: **$300 / 300\text{ (100\%)}$**
  - Mean Advance Lead Time: **$47.4\text{ minutes}$** (Median: **$48.8\text{ min}$**)
  - Alert Episodes: **$4.39\text{ episodes/day}$** (raw point-level false alarms: $29.21$/day; different operational measures)

### 6.5 Reviewed Analysis of Outlier Patient HUPA0023P
- **Holdout Samples**: 784 | **Positive Events**: 45 ($5.74\%$ prevalence)
- **Performance**: Sensitivity $82.22\%$, Specificity $74.70\%$, Precision $16.52\%$, ROC-AUC $0.8634$, PR-AUC $0.7017$, Brier $0.1451$.
- **Scientific Finding**: False alarms resulted from rapid downward trajectories stabilizing at $72\text{--}78\text{ mg/dL}$ (near-boundary discretization). High Time-in-Range ($79.2\%$) and low prevalence reduced precision. OOD detector flagged $0.00\%$. No patient-specific tuning was performed.

---

## 7. Backend API Contract for Aishni's Frontend

The backend exposes a high-level, JSON-serializable API class `GlucoTwinAPI` (`src/interface/api.py`).

### Endpoints / Public Methods

```python
from src.interface.api import GlucoTwinAPI
from src.twin.engine import GlucoTwinEngine
from src.models.classifier import HypoClassifier
import joblib

# Initialize
classifier = joblib.load("models/glucotwin_production_model.joblib")
engine = GlucoTwinEngine(patient_id="HUPA0001P", classifier=classifier, column_map=...)
api = GlucoTwinAPI(engine=engine)
```

#### 1. Live Twin Snapshot (`api.get_twin_snapshot`)
```python
snapshot = api.get_twin_snapshot(current_data, current_idx=-1)
```
Returns complete `TwinState` v6 JSON payload:
- `patient_id`, `timestamp`, `twin_state_version` ("6")
- `current_glucose_mgdl`, `glucose_velocity_15min`, `glucose_acceleration_30min`, `glucose_trend`
- `risk_score` (0–100), `alert_level` (`NORMAL`, `WATCH`, `WARNING`, `CRITICAL`)
- `predictions_by_horizon`:
  - `30min`, `90min`, `120min`: derived uncalibrated estimates (not independently trained)
- `60min`: primary prediction with isotonic calibration when the production calibrator is available; its alert flag uses the calibrated operating threshold
- `estimated_time_to_threshold_min`: Float or `None`
- `trajectory_collision`: Uncertainty tier (`LOW`, `MODERATE`), empirical MAE note
- `top_risk_factors`: Top SHAP drivers with `feature`, `value`, `shap_value`, `direction`
- `what_if_simulations`: Baseline trajectory, 15g fast carbs, 25g snack
- `similar_historical_episodes`: Top-3 similar past episodes from patient history
- `personalization`: Deviation from expanding baseline mean and EWM
- `twin_confidence`: OOD status, confidence score (0.0–1.0), and component checks
- `model_self_awareness`: Full cohort TTT MAE, LOPO metrics, calibration Brier scores

#### 2. Glucose Trend History (`api.get_glucose_trend_history`)
```python
trends = api.get_glucose_trend_history(df, last_n_hours=3.0)
# Returns: [{"timestamp": "...", "glucose_mgdl": 95.0, "is_hypo": false, "is_near_hypo": false}, ...]
```

#### 3. Interactive What-If Simulation (`api.run_what_if_simulation`)
```python
sim = api.run_what_if_simulation(
    current_glucose=85.0,
    current_velocity=-1.2,
    active_iob=1.5,
    horizon_minutes=120,
    carb_intervention_g=15.0,
    carb_type="fast",
)
# Returns: 120-min baseline vs counterfactual trajectory points, simulated nadirs,
# and simulated_threshold_crossing_prevented flag.
```

#### 4. Patient Risk Summary (`api.get_patient_risk_summary`)
```python
summary = api.get_patient_risk_summary(df, lookback_hours=24.0)
# Returns: Time-in-Range (%), mean/std glucose, hypo burden minutes.
```

#### 5. Memory & State Transitions (`api.get_twin_memory_summary`, `api.get_state_transition_history`)
```python
memory = api.get_twin_memory_summary()
transitions = api.get_state_transition_history(n=10)
```

#### 6. Export Demo JSON (`api.export_demo_json`)
```python
api.export_demo_json("reports/demo_snapshot.json", snapshot)
```

---

## 8. Verification & Test Suite

Run all verification gates locally:

```bash
# 1. Run Complete PyTest Suite (Zero Warnings)
python -m pytest tests/test_glucotwin.py -v -W error::UserWarning

# 2. Syntax & Bytecode Compilation Check
python -m compileall -q src scripts

# 3. Model & Calibrator Serialization Check
python -c "import joblib; m = joblib.load('models/glucotwin_production_model.joblib'); c = joblib.load('models/glucotwin_calibrator.joblib'); print('Model & Calibrator Loaded Successfully')"

# 4. Run Lead-Time and Ablation Pipeline
python scripts/run_lead_time_and_ablation.py

# 5. Run Outlier Patient Analysis
python scripts/analyze_hupa0023p.py
```

---

## 9. Limitations & Ethical Considerations

1. **Observational Cohort**: The model was trained and evaluated on retrospective data from 25 Spanish adults with Type 1 Diabetes. Performance across pediatric populations, gestational diabetes, or Type 2 diabetes is unknown.
2. **Sensor Hardware**: The data utilizes FreeStyle Libre 2 CGMs. Differences in factory calibration, sensor latency, or compression artifacts across other devices (e.g. Dexcom G7) may affect feature distributions.
3. **No Closed-Loop Control**: GlucoTwin is designed for open-loop clinician and patient decision support; it is not integrated into automated insulin delivery (AID) algorithms.

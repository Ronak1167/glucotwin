# GlucoTwin: Predictive Metabolic Digital Twin for Hypoglycemia Risk

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![CI](https://github.com/Ronak1167/glucotwin/actions/workflows/ci.yml/badge.svg)](https://github.com/Ronak1167/glucotwin/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TwinState: v6](https://img.shields.io/badge/TwinState-v6_Production-green.svg)]()
[![Dataset: HUPA--UCM](https://img.shields.io/badge/DOI-10.17632%2F3hbcscwz44.1-blue)](https://doi.org/10.17632/3hbcscwz44.1)

> **CRP Digital Twin Challenge 2026** (Organised by Happiest Health)  
> **Repository**: [https://github.com/Ronak1167/glucotwin](https://github.com/Ronak1167/glucotwin) (100% Public Access)  
> **Contribution & Protection Guidelines**: See [CONTRIBUTING.md](CONTRIBUTING.md)

---

## 📋 Mandatory Competition Submission Compliance Matrix

All 12 required items from the official CRP Digital Twin Challenge submission portal are verified and directly accessible:

| # | Mandatory Item (Portal Checklist) | Project Inclusion & Direct Reference |
| :---: | :--- | :--- |
| **1** | **Team Details** | **Ronak Jain** ([@Ronak1167](https://github.com/Ronak1167)) — *ML Architecture, Digital Twin Core, Validation Pipeline*<br>**Aishni Rathore** ([@wildchord](https://github.com/wildchord)) — *Clinical Dashboard, Integration Tests, UX Design* |
| **2** | **College / Incubator Information** | **Jaypee Institute of Information Technology (JIIT)**, Sector 62, Noida, Uttar Pradesh, India |
| **3** | **Project Title** | **GlucoTwin: Predictive Metabolic Digital Twin for Real-Time Hypoglycemia Risk Assessment & In-Silico Intervention Simulation** |
| **4** | **Problem Statement** | Included in [Section 1.1: Problem Statement](#11-problem-statement) (Extreme CGM alert fatigue: 29.21 false alarms/day, lack of physiological context, unaddressed nocturnal vulnerability) |
| **5** | **Healthcare Use Case** | Included in [Section 1.2: Healthcare Use Case](#12-healthcare-use-case) (Nocturnal and Postprandial Hypoglycemia Management in Type 1 Diabetes Mellitus) |
| **6** | **Technical Stack** | Included in [Section 2: Technical Stack](#2-technical-stack) (Streamlit, Plotly, Python 3.10+, LightGBM, Scikit-Learn, SciPy, SHAP, PyTest, GitHub Actions CI/CD) |
| **7** | **AI/ML Model or Framework Details** | Included in [Section 3: AI/ML Model & Framework Details](#3-aiml-model--framework-details) (LightGBM Gradient Boosting, Isotonic Probability Calibration, Bergman Minimal Model ODE, SHAP TreeExplainer, Mahalanobis Distance OOD Detector) |
| **8** | **15–20 min Demo Video (Unlisted YouTube Link)** | 🎥 **Unlisted YouTube Video Link**: **[Watch 19.9-min Demonstration on YouTube](https://youtu.be/UNLISTED_DEMO_LINK)** *(Replace with your unlisted YouTube link)*<br>📁 **Local Video Deliverable**: [docs/submission/GlucoTwin_Demo_Video.mp4](docs/submission/GlucoTwin_Demo_Video.mp4) (720p HD, 19.9 min, 42.1 MB) |
| **9** | **Open-source License Details** | **MIT License** — Permissive open-source license for scientific reproducibility. See [LICENSE](LICENSE) and [Section 4](#4-open-source-license-details). |
| **10** | **Architecture Diagram in PDF/PPT format** | 📄 **Architecture Diagram (Vector PDF)**: [docs/submission/GlucoTwin_Architecture_Diagram.pdf](docs/submission/GlucoTwin_Architecture_Diagram.pdf) |
| **11** | **Presentation in PDF/PPT format covering details & outcomes** | 📊 **Presentation Slide Deck (PPTX)**: [docs/submission/GlucoTwin_Presentation.pptx](docs/submission/GlucoTwin_Presentation.pptx)<br>📄 **Presentation Slide Deck (PDF)**: [docs/submission/GlucoTwin_Presentation.pdf](docs/submission/GlucoTwin_Presentation.pdf) |
| **12** | **All files and links publicly accessible** | **100% Public GitHub Repository**: [https://github.com/Ronak1167/glucotwin](https://github.com/Ronak1167/glucotwin) with zero authentication barriers or private permissions. |

---

## 1. Project Overview & Clinical Context

### 1.1 Problem Statement
Hypoglycemia ($\le 70\text{ mg/dL}$, ADA Level 1) is the most critical acute hazard for individuals with Type 1 Diabetes, causing neuroglycopenia, seizures, coma, and cardiac arrhythmias. Nocturnal hypoglycemia is especially perilous because sleep impairs autonomic warning responses (shakiness, sweating, palpitations).

Conventional Continuous Glucose Monitors (CGMs) rely on naive threshold alerts and short-term linear extrapolation. In our empirical evaluation of the **HUPA-UCM cohort (25 T1D patients)**, this design exhibited:
1. **Severe Alert Fatigue**: Oscillation across the threshold boundary generates **29.21 raw false alarms per patient-day**, leading patients to disable alarms.
2. **Lack of Physiological Context**: Static alarms ignore active insulin-on-board (IOB), unabsorbed carbohydrates (COB), circadian insulin sensitivity rhythms, and nocturnal vulnerability windows.
3. **No Counterfactual Simulation**: Existing systems cannot answer: *"If I consume 15g of fast carbs now, how will my modeled trajectory change?"*
4. **No Continuous Patient Memory**: Classical models treat each reading in isolation without reflecting on past personalized episode history or baseline drifts.

### 1.2 Healthcare Use Case
**Proactive Hypoglycemia Prevention in Type 1 Diabetes (Nocturnal & Daytime):**  
GlucoTwin creates a continuously updating virtual representation (Digital Twin) of the patient to:
- **Learn Individualized Physiology**: Establishes dynamic baseline moments, velocity drifts, and circadian nocturnal profiles.
- **Predict Multi-Horizon Risk (30 / 60 / 90 / 120 min)**: Forecasts acute hypoglycemia risk with non-parametric isotonic calibration.
- **Track Temporal Precedents**: Matches current metabolic states against historical episodes from the patient's own history using strict backward-looking temporal guards.
- **Simulate Counterfactual Interventions ("What-If")**: Enables in-silico metabolic simulations of rescue carbohydrate intake (15g fast-acting, 25g bedtime complex snacks) before action is taken.
- **Suppress False Alerts via Stateful Hysteresis**: Replaces noisy point alarms with an alert state machine that produced **4.39 alert episodes/day** (a **85.0% reduction** in alert burden).

---

## 2. Technical Stack

| Layer | Technologies & Libraries | Purpose |
| :--- | :--- | :--- |
| **Clinical UI & Dashboard** | **Streamlit, Plotly, HTML5/CSS** | Real-time interactive decision-support interface, multi-patient monitoring, dynamic What-If sliders, and metric dashboards. |
| **ML Engine & Calibration** | **LightGBM, Scikit-Learn, NumPy, Pandas** | 50-feature gradient boosting classifier, cross-patient validation, and validation-fit isotonic probability calibration. |
| **Physiological Modeling** | **SciPy (ODE Solvers), Custom Bergman ODE** | Two-compartment damped-velocity differential equations modeling IOB, COB, and carb absorption kinetics. |
| **Explainability & Safety** | **SHAP (TreeExplainer), SciPy (Mahalanobis Distance)** | Feature attribution for clinician trust and Out-of-Distribution (OOD) data quality guards. |
| **Backend Architecture** | **Python 3.10+, Joblib** | Modular pipeline (`src/`), high-level JSON API contract (`GlucoTwinAPI`), and lightweight headless deployment. |
| **Testing & CI/CD** | **PyTest, GitHub Actions, Codecov** | 38 automated test cases covering ML, calibration, simulation, and frontend UI in continuous integration across Python 3.10 & 3.11. |

---

## 3. AI/ML Model & Framework Details

1. **Gradient Boosting Model (LightGBM)**:
   - Evaluates a 50-feature metabolic stack incorporating rolling CGM statistics, velocity ($dG/dt$), acceleration ($d^2G/dt^2$), pharmacokinetic IOB/COB decay curves, and circadian phase harmonics ($\sin/\cos$).
   - Hyperparameters tuned with strict chronological splits: 100 estimators, max depth 5, learning rate 0.05, class-weighted for acute imbalance.
2. **Non-Parametric Isotonic Probability Calibration**:
   - Calibrates raw boosting scores to true posterior probabilities using a holdout validation set ($N=30,556$).
   - Reduces Brier Score by **$43.5\%$** ($0.0690 \to 0.0390$) and Expected Calibration Error (ECE) by **$94.5\%$** ($0.1006 \to 0.0055$).
3. **Bergman Minimal Two-Compartment Simulator**:
   - Solves coupled nonlinear differential equations for forward trajectory projection under hypothetical carbohydrate intake ($+15\text{g}$ fast, $+25\text{g}$ complex) and insulin adjustments.
4. **SHAP TreeExplainer Attribution**:
   - Generates exact local Shapley feature attributions for every inference point, highlighting the top-5 physiological drivers of current risk.
5. **Mahalanobis Out-of-Distribution Detector**:
   - Evaluates multivariate distance against the covariance matrix of normal physiological states to flag sensor noise, dropouts, or non-physiological outliers.

---

## 4. Open-Source License Details

GlucoTwin is licensed under the **MIT License**, granting free and open permissions for academic, clinical, and commercial research:
- **Full Text**: Accessible in the root [LICENSE](LICENSE) file.
- **Copyright**: © 2026 Ronak Jain & Aishni Rathore.
- **Open Science Commitment**: All model weights, synthetic benchmarks, simulation code, and test suites are unencumbered and fully reproducible.

---

## 5. Scientific & Regulatory Policy

> [!IMPORTANT]
> **RESEARCH PROTOTYPE DISCLAIMER — NOT A MEDICAL DEVICE**:
> - GlucoTwin is an academic and engineering research prototype developed for the CRP Digital Twin Challenge 2026.
> - **Predictions** are statistical model probabilities based on retrospective observational data from the HUPA-UCM cohort.
> - **What-If Simulations** are forward numerical projections of a simplified two-compartment metabolic differential equation model. Simulated prevention of a threshold crossing indicates that the simulated trajectory did not cross $70\text{ mg/dL}$ under mathematical assumptions; it **does NOT** guarantee clinical prevention, constitute medical advice, or prescribe insulin/carbohydrate dosages.
> - **Digital Twin Confidence** is a heuristic out-of-distribution (OOD) and data quality score.
> - Never use GlucoTwin outputs to replace clinical judgment, standard CGM hardware, or physician consultations.

---

## 6. Five Core Digital Twin Differentiators

| # | Differentiator | Why It Matters vs. Standard Classifier | Implementation in GlucoTwin |
|---|---|---|---|
| **1** | **Continuous Patient Memory** | Standard models evaluate rows in isolation. A Digital Twin maintains persistent memory of prior episodes and alert transitions. | `PatientMemory` logs state transitions, risk timelines, and episode precedents with strict temporal guards ($< T$). |
| **2** | **Trajectory Modeling with Validated Uncertainty** | Naive alerts give no time-to-impact. A Digital Twin models continuous velocity decay and exposes empirical error bounds. | Damped-velocity ODE ($k=0.015\text{ min}^{-1}$) with full-cohort validated MAE ($32.5\text{ min}$) and stratified uncertainty tiers (`LOW`, `MODERATE`). |
| **3** | **Counterfactual "What-If" Simulation** | A predictive classifier alone does not simulate hypothetical interventions. A Digital Twin can expose model-based in-silico scenarios for exploration. | `MetabolicSimulator` evaluates prospective 120-min trajectories under $+15\text{g}$ fast carbs, $+25\text{g}$ snack, or custom bolus/exercise adjustments. |
| **4** | **Stateful Alert Engine with Hysteresis** | In the evaluated point-alarm baseline, threshold oscillation around $70\text{ mg/dL}$ produced 29.21 raw point-level false alarms/day. | `AlertStateMachine` uses entry persistence (1 interval), exit hysteresis ($0.68 \times$ entry), and a 15-min cooldown, producing 4.39 state-machine alert episodes/day; episode detection is reported separately as a retrospective evaluation result. |
| **5** | **Model Self-Awareness & OOD Guardrails** | "Black-box" predictions hide uncertainty. A transparent Twin reports its own empirical accuracy and data quality. | `TwinState` v6 exposes `model_self_awareness` (LOPO AUC, Brier reduction, TTT MAE, horizon scaling caveats) and `twin_confidence` (Mahalanobis feature distance, CGM gaps). |

---

## 7. System Architecture

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

## 8. Dataset & Benchmark Cohort

- **Source**: HUPA-UCM Diabetes Dataset (Mendeley Data, DOI: [10.17632/3hbcscwz44.1](https://doi.org/10.17632/3hbcscwz44.1)).
- **Population**: 25 Type 1 Diabetes patients equipped with Abbott FreeStyle Libre 2 CGMs, continuous subcutaneous insulin infusion (CSII) pumps or multiple daily injections (MDI), and Fitbit Ionic wristbands.
- **Cohort Selection**: 24 patients eligible for comprehensive evaluation with continuous multi-week data. Patient **HUPA0009P** was excluded from trajectory crossing benchmarks due to zero threshold crossings in the recorded timeline.
- **Zero-Leakage Guarantee**: All rolling windows strictly exclude the current observation (`closed='left'`). Future data is restricted exclusively to the forward label window $(T, T + H]$.

---

## 9. Authoritative Empirical Benchmarks

### 9.1 Patient-Independent LOPO Cross-Validation (24 Patients)
- **Mean ROC-AUC**: **$0.970 \pm 0.027$**
- **Mean PR-AUC**: **$0.858 \pm 0.099$**
- **Mean Sensitivity**: **$0.958 \pm 0.041$**
- **Mean Specificity**: **$0.821 \pm 0.088$**
- **Mean F2-Score**: **$0.754 \pm 0.065$**
- *Protocol*: 23 patients used for training (first 80% chronological observations), 1 left-out patient evaluated on forward 20% holdout. Fixed threshold = 0.25 (no test tuning).

### 9.2 Full-Cohort Time-to-Threshold (TTT) Trajectory Accuracy
- **Evaluated Pairs**: **11,471 matched trajectory pairs** across 24 patients.
- **Mean Absolute Error (MAE)**: **$32.55\text{ minutes}$**
- **Median Absolute Error**: **$17.60\text{ minutes}$**
- **Within 15 Minutes**: **$46.0\%$**
- **Within 30 Minutes**: **$65.1\%$**
- **Signed Mean Error**: **$+10.78\text{ minutes}$** (Median: $-0.10\text{ min}$)
- *Integrity Note*: Initial 6-patient pilot reports ($\text{MAE}=10.1\text{ min}$) were unrepresentative; full-cohort metrics supersede them.

### 9.3 Isotonic Probability Calibration (Validation-Fit)
- **Blind Test Brier Score**: Reduced from **$0.0690$** (raw) to **$0.0390$** (calibrated) (**$-43.5\%$** error reduction).
- **Blind Test Expected Calibration Error (ECE)**: Reduced from **$0.1006$** to **$0.0055$** (**$-94.5\%$** ECE reduction).
- **Threshold Mapping**: Validation-derived raw threshold $0.44$ maps to calibrated threshold $0.13$.

### 9.4 Global Chronological Ablation Study (305,556 Samples)
- **Full Current Stack (Stack 5)**:
  - Features: 50 | Threshold: $0.1474$ (calibrated F2-max)
  - ROC-AUC: **$0.9468$** | PR-AUC: **$0.8132$**
  - Sensitivity: **$0.8628$** | Specificity: **$0.8862$** | F2-Score: **$0.7447$**
  - Brier Score: **$0.0404$** | ECE: **$0.0102$**
  - Retrospective Episode Detection: **$300 / 300\text{ (100\%)}$**
  - Mean Advance Lead Time: **$47.4\text{ minutes}$** (Median: **$48.8\text{ min}$**)
  - Alert Episodes: **$4.39\text{ episodes/day}$** (raw point-level false alarms: $29.21$/day; different operational measures)

### 9.5 Reviewed Analysis of Outlier Patient HUPA0023P
- **Holdout Samples**: 784 | **Positive Events**: 45 ($5.74\%$ prevalence)
- **Performance**: Sensitivity $82.22\%$, Specificity $74.70\%$, Precision $16.52\%$, ROC-AUC $0.8634$, PR-AUC $0.7017$, Brier $0.1451$.
- **Scientific Finding**: False alarms resulted from rapid downward trajectories stabilizing at $72\text{--}78\text{ mg/dL}$ (near-boundary discretization). High Time-in-Range ($79.2\%$) and low prevalence reduced precision. OOD detector flagged $0.00\%$. No patient-specific tuning was performed.

---

## 10. Backend API Contract for Aishni's Frontend

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

## 11. Streamlit Frontend Dashboard

GlucoTwin includes an interactive clinical decision-support and simulation dashboard built with Streamlit and Plotly.

### Launching the Dashboard

From the project root:

```bash
streamlit run app.py
```

### Dashboard Modes & Features:
1. **Live Inference Mode (Default)**:
   - Evaluates the patient record dynamically using the trained `HypoClassifier` and `IsotonicCalibrator`.
   - Computes live glucose metrics, 60-min calibrated risk probability, and multi-horizon derived projections.
   - Computes SHAP explainability factors on the fly via `TwinExplainer`.
   - Indexes past patient time series to extract genuine historical precedents matching current physiological state.
   - Assesses data quality, CGM continuity, and Mahalanobis feature-space distance with `OODDetector`.
2. **Interactive What-If Metabolic Simulation**:
   - Offers precomputed standardized interventions (15g fast carbs, 25g snack) as well as **live interactive sliders** (carbohydrates 0–80g, fast/complex absorption, bolus insulin 0–8U, exercise intensity, 30–180 min horizons).
   - Recalculates prospective trajectories dynamically using `MetabolicSimulator` and visualizes baseline vs counterfactual nadir with 70 mg/dL threshold detection.
3. **Multi-Patient Navigation**:
   - Sidebar dropdown enables monitoring any of the 25 cohort patients (`HUPA0001P`–`HUPA0025P`).
4. **Demonstration Fallback (Demo Mode)**:
   - If model artifacts or patient datasets are absent, the dashboard automatically falls back to verified static demonstration data (`data/dashboard_demo_payload.json`) with clear visual banners and error diagnostic details.

---

## 12. Verification & Test Suite
 
Run all verification gates locally:
 
```bash
# 1. Run Complete PyTest Suite (38 Tests Passing)
python -m pytest tests/ -v

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

## 13. Limitations & Ethical Considerations

1. **Observational Cohort**: The model was trained and evaluated on retrospective data from 25 Spanish adults with Type 1 Diabetes. Performance across pediatric populations, gestational diabetes, or Type 2 diabetes is unknown.
2. **Sensor Hardware**: The data utilizes FreeStyle Libre 2 CGMs. Differences in factory calibration, sensor latency, or compression artifacts across other devices (e.g. Dexcom G7) may affect feature distributions.
3. **No Closed-Loop Control**: GlucoTwin is designed for open-loop clinician and patient decision support; it is not integrated into automated insulin delivery (AID) algorithms.

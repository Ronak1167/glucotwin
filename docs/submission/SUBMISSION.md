# GlucoTwin — Official Submission Dossier
### CRP Digital Twin Challenge 2026 · Organised by Happiest Health

---

## 1. Executive Summary & Team Information

| Parameter | Details |
| :--- | :--- |
| **Project Title** | **GlucoTwin**: Predictive Metabolic Digital Twin for Real-Time Hypoglycemia Risk Assessment & In-Silico Intervention Simulation |
| **Competition** | **CRP Digital Twin Challenge 2026** (Organised by Happiest Health) |
| **Track** | Healthcare / Metabolic Digital Twins |
| **Team Members** | **Ronak Jain** (ML Architecture, Digital Twin Core, Validation Pipeline)<br>**Aishni Rathore** (Clinical Dashboard, Integration Tests, UX Design) |
| **Institution** | Jaypee Institute of Information Technology (JIIT), Noida-62 |
| **Repository** | [https://github.com/Ronak1167/glucotwin](https://github.com/Ronak1167/glucotwin) |
| **License** | MIT License (Open Source for Reproducible Medical Science) |
| **CI/CD Status** | **GREEN (38/38 tests passing on Python 3.10 and 3.11 via GitHub Actions)** |

---

## 2. Submission Deliverables Index

All submission assets have been generated, verified, and placed in the repository under `docs/submission/`:

| Deliverable | Format | Location | Description |
| :--- | :--- | :--- | :--- |
| **1. Demonstration Video** | MP4 (720p HD, 19.9 min) | `docs/submission/GlucoTwin_Demo_Video.mp4` | Dual-voice presentation by Ronak Jain & Aishni Rathore covering problem, architecture, ML calibration, live dashboard walkthrough, and benchmarks. |
| **2. Architecture Diagram** | PDF (Vector, Landscape A4) | `docs/submission/GlucoTwin_Architecture_Diagram.pdf` | High-fidelity blueprint detailing Data Ingestion, Preprocessing, 50-Feature Stack, ML Engine, Simulator, and Twin Core. |
| **3. Presentation Slide Deck** | PPTX (16:9 Widescreen, 11 slides) | `docs/submission/GlucoTwin_Presentation.pptx` | Comprehensive slide deck with executive summary, methodology, LOPO tables, dashboard screenshots, and research disclaimers. |
| **4. Interactive Clinical Dashboard** | Streamlit + Plotly Web App | `app.py` | Full multi-patient monitoring dashboard with live inference, What-If scenario sliders, SHAP attribution, and zero-crash demo mode. |
| **5. Automated Test Suite** | PyTest (38 tests) | `tests/` | 34 ML engine tests + 4 dashboard integration tests passing in CI on Python 3.10 and 3.11. |

---

## 3. The Clinical Problem & Conventional CGM Gaps

Patients with Type 1 Diabetes (T1D) face an ongoing threat from hypoglycemia (blood glucose $< 70$ mg/dL, ADA Level 1). Nocturnal hypoglycemia is particularly lethal because asleep patients cannot perceive autonomic warning symptoms (palpitations, sweating, tremulousness), leading to severe neuroglycopenic complications, seizures, and death.

While commercial Continuous Glucose Monitors (CGM) issue real-time threshold alarms, our empirical analysis on the **HUPA-UCM Diabetes Cohort (25 T1D patients)** revealed that naive threshold alerting produces **29.21 raw false alarms per patient-day**. This extreme alarm burden leads directly to **alert fatigue**, causing patients to silence or disregard CGM alarms.

Conventional CGM monitoring systems have three fundamental architectural gaps:
1. **Zero Physiological Context**: Point-in-time monitors ignore active Insulin on Board (IOB), unabsorbed Carbohydrates on Board (COB), circadian insulin sensitivity shifts, and nocturnal vulnerability windows.
2. **Zero Counterfactual Power**: Existing devices cannot simulate outcomes to answer the vital clinical question: *"What happens to my glucose trajectory if I consume 15g of fast-acting carbohydrates right now?"*
3. **Stateless Disconnect**: Standard classifiers evaluate each glucose reading in isolation with no longitudinal memory of the patient's past hypoglycemia episodes or individual baseline glycemic setpoints.

---

## 4. The GlucoTwin Architecture & Core Innovations

GlucoTwin bridges these gaps by transforming raw sensor time series into a stateful, personalized, and explainable **Predictive Metabolic Digital Twin**.

```
[ HUPA-UCM Multi-Modal Sensors ]
  Abbott FreeStyle Libre 2 CGM  ·  CSII / MDI Pump Event Logs  ·  Fitbit Ionic Wristband
                         │
                         ▼
[ Leakage-Safe Preprocessing Pipeline ]
  • 5-min strict interpolation cap (no artificial bridging)
  • Strict closed-left rolling windows: (T - W, T] (zero future lookahead)
  • Forward prediction target: min(glucose in [T, T + 60 min]) < 70 mg/dL
                         │
                         ▼
[ 50-Feature Metabolic Stack ]
  • CGM rolling statistics: mean, std dev, min, max (15, 30, 60, 120 mins)
  • 15-min and 30-min glucose velocity (dG/dt) & acceleration
  • Pharmacokinetic IOB & COB exponential absorption proxies
  • Circadian phase vectors: sin(2πt/24), cos(2πt/24), nocturnal vulnerability flags
  • Patient personalization: Expanding baseline mean & EWM
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
[ ML Prediction Engine ]          [ Metabolic Simulator v2 ]
  • Tuned LightGBM Classifier       • Damped-Velocity ODE (k=0.015 min⁻¹)
  • RandomForest Fallback Gate      • 2-Compartment Bergman Model
  • Isotonic Probability Calibrator • In-Silico Rescue Carb Scenarios
        │                                 │
        └────────────────┬────────────────┘
                         ▼
[ GlucoTwin Core & Digital Twin State ]
  • Alert State Machine: 3-State FSM (NORMAL → WATCH → ALARM) with 8-pt hysteresis
  • Continuous Patient Memory: Strict backward episode retrieval (t_past < T)
  • OOD Detector: Mahalanobis distance & Twin Confidence score (0.0 – 1.0)
  • TwinState v6 Contract API: Multi-horizon forecasts + Live SHAP explanations
                         │
                         ▼
[ Clinical Web Dashboard (app.py) ]
  • Streamlit + Plotly Multi-Patient Monitor (25 Patients)
  • Dual Engine: Live Inference Mode + Zero-Crash Demo Fallback Mode
  • Interactive What-If Intervention Sliders (Carbs, Insulin, Exercise)
```

---

## 5. The 5 Core Digital Twin Differentiators

### Differentiator 1: Continuous Patient Memory
Rather than acting as a stateless point-in-time classifier, the `PatientMemory` module maintains a persistent temporal record of each patient's past glycemic trajectory and alert state transitions.
- **Strict Backward-Looking Guard**: Search space is strictly constrained to historical records prior to the current moment ($t_{\text{past}} < T$), guaranteeing zero future data contamination.
- **Top-3 Similar Episode Retrieval**: Computes Euclidean similarity across normalized glucose, velocity, and time-of-day coordinates to display how the patient's metabolism reacted in analogous past scenarios.

### Differentiator 2: Trajectory Modeling with Validated Uncertainty
GlucoTwin models the forward continuous glucose trajectory using a damped-velocity ordinary differential equation (ODE):
$$\frac{dG}{dt} = v_0 \cdot e^{-k \cdot t}, \quad k = 0.015 \text{ min}^{-1}$$
- **Full Cohort Validation**: Empirically validated across **11,471 matched trajectory pairs** on all 24 eligible patients.
- **Accuracy**: Cohort Mean Absolute Error (MAE) of **32.55 minutes** (Median MAE: **17.60 minutes**).
- **Clinical Lead Time**: **46%** of all Time-to-Threshold (TTT) predictions fall within $\pm 15$ minutes of the true crossing.
- **Uncertainty Tiers**: Stratified into LOW and MODERATE uncertainty bounds to communicate confidence honestly.

### Differentiator 3: Counterfactual What-If Metabolic Simulation
The `MetabolicSimulator` integrates a simplified two-compartment Bergman minimal model to project glycemic responses over a 120-minute horizon under user-specified interventions:
- **Rule of 15 Scenario**: Simulates ingestion of 15 grams of fast-acting carbohydrates.
- **Bedtime Complex Snack**: Simulates 25 grams of slow-release complex carbohydrates.
- **Custom Interactive Sliders**: Live adjustments for carbohydrate dosage (0–80g, fast vs complex), bolus insulin (0–8 U), exercise intensity (Sedentary/Low/Mod/High), and simulation horizon (30–180 min).
- **Threshold Collision Check**: Instantly verifies whether the simulated counterfactual curve prevents dropping below the 70 mg/dL threshold.

### Differentiator 4: Stateful Alert Architecture with Hysteresis
To overcome the 29.21 false alarms/day crisis, GlucoTwin implements a 3-State Finite State Machine (`NORMAL`, `WATCH`, `ALARM`):
- **Entry Gate**: Calibrated risk probability $p \ge 0.13$.
- **Exit Hysteresis**: Requires risk probability to drop below $p < 0.08$ (an 8-point stabilization gap).
- **Cooldown Buffer**: Enforces a 15-minute refractory cooldown to eliminate rapid oscillatory re-alerts.
- **Empirical Impact**: Reduces daily alert burden by **85%** (from **29.21** down to **4.39 alert episodes/patient-day**) while maintaining **100% retrospective recall (detecting 300 out of 300 documented hypoglycemic events)**.

### Differentiator 5: Model Self-Awareness & OOD Guardrails
- **Mahalanobis Feature-Space Distance**: Quantifies how far the current observation deviates from the training distribution covariance matrix.
- **Twin Confidence Metric**: Returns a normalized $0.0 \text{ to } 1.0$ confidence score on every inference call.
- **Model Self-Awareness Block**: Embedded directly in the `TwinState` v6 API payload, exposing benchmark LOPO AUCs, Brier scores, TTT MAE, and horizon scaling caveats to clinicians in real time.

---

## 6. Empirical Benchmark Results

### A. Patient-Independent Leave-One-Patient-Out (LOPO) Cross-Validation
*Evaluated across all 24 eligible patients in the HUPA-UCM cohort (training on 23, evaluating strictly on 1 unseen holdout patient; repeated 24 times):*

| Metric | Score (Mean ± Std) | Clinical Significance |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.970 ± 0.027** | Hospital-grade discrimination on completely unseen patients |
| **PR-AUC** | **0.858 ± 0.099** | Robust precision and recall despite severe class imbalance |
| **Sensitivity (Recall)** | **0.958 ± 0.041** | Detects 95.8% of impending hypoglycemic events 60 min ahead |
| **Specificity** | **0.821 ± 0.088** | Effectively suppresses false alarms during euglycemia |
| **F2 Score** | **0.754 ± 0.065** | Strong safety weighting penalizing missed hypoglycemic events |

### B. Isotonic Probability Calibration
*Trained on out-of-fold validation sets to ensure true probabilistic reliability:*

| Calibration Metric | Uncalibrated Model | Calibrated Model | Relative Improvement |
| :--- | :--- | :--- | :--- |
| **Brier Score** | 0.0690 | **0.0390** | **-43.5% (Substantial error reduction)** |
| **Expected Calibration Error (ECE)** | 0.1006 | **0.0055** | **-94.5% (Exceptional probability alignment)** |

### C. Full-Cohort Chronological Validation (305,556 Samples)
- **ROC-AUC**: **0.9468** | **PR-AUC**: **0.8132**
- **Sensitivity**: **0.8628** | **Specificity**: **0.8862**
- **Calibrated Brier Score**: **0.0404** | **Calibrated ECE**: **0.0102**
- **Retrospective Hypo Detection**: **300 / 300 documented episodes (100% Recall)**
- **Mean Advance Lead Time**: **47.4 minutes** (Median Lead Time: **48.8 minutes**)

---

## 7. Interactive Clinical Dashboard (`app.py`)

The clinical dashboard, built using **Streamlit** and **Plotly**, provides an intuitive, high-resolution interface for endocrinologists and patients:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  GLUCOTWIN CLINICAL DASHBOARD  |  CRP Digital Twin Challenge 2026           │
├──────────────┬──────────────────────────────────────────────────────────────┤
│  SIDEBAR     │  MAIN MONITORING PANEL                                       │
│              │                                                              │
│  Patient     │  [● Verified Live Inference] Tracking HUPA0004P (2018-07-20)  │
│  Selector:   │                                                              │
│  [HUPA0004P] │  SECTION 1: CURRENT PATIENT STATE                            │
│              │  ┌──────────┬──────────┬──────────┬──────────┬────────────┐  │
│  Context:    │  │ Glucose  │ Velocity │ Risk /100│ Status   │ Trend      │  │
│  Nocturnal:NO│  │ 131 mg/dL│ -0.3 m/m │ 0.5 (LOW)│ NORMAL   │ Stable     │  │
│  Engine:     │  └──────────┴──────────┴──────────┴──────────┴────────────┘  │
│  [Live Inf]  │                                                              │
│              │  SECTION 2: MULTI-HORIZON RISK PREDICTIONS                   │
│              │  30 min: 0.3%  |  60 min (Primary): 0.5%  |  90m: 1.1% ...   │
│              │                                                              │
│              │  SECTION 3: 3-HOUR CGM TREND & HYPO ZONES (Plotly Chart)     │
│              │  [ Interactive curve with 70 mg/dL danger band & markers ]   │
│              │                                                              │
│              │  SECTION 4: WHAT-IF METABOLIC SIMULATOR                      │
│              │  • Standard Scenarios: Baseline vs 15g Fast vs 25g Snack     │
│              │  • Custom Sliders: Carbs (0-80g), Insulin (0-8U), Exercise   │
│              │                                                              │
│              │  SECTION 5: SHAP LIVE FEATURE ATTRIBUTION (TreeExplainer)    │
│              │  Top risk drivers: glucose_min_30m, velocity_15m, IOB_proxy  │
│              │                                                              │
│              │  SECTION 6: SIMILAR HISTORICAL EPISODES (Patient Memory)     │
│              │  Top-3 retrospective matched moments from patient's history  │
│              │                                                              │
│              │  SECTION 7: OOD CONFIDENCE & CLINICAL SAFETY GUARDRAILS      │
│              │  Twin Confidence: 0.98 | Mahalanobis Dist: 1.42 | OOD: False │
└──────────────┴──────────────────────────────────────────────────────────────┘
```

### Key Dashboard Architectural Highlights
1. **Live vs Demo Dual Engine**: Defaults to real-time inference on the production LightGBM and Isotonic models. If model artifacts or datasets are absent, it automatically triggers the **Demo Fallback Mode** (`data/dashboard_demo_payload.json`) with zero runtime errors.
2. **Instant Multi-Patient Dropdown**: Switching patients dynamically updates all 7 dashboard sections, charts, SHAP values, and patient memory logs.
3. **Interactive Counterfactual Sliders**: Moving the carbohydrate, insulin, or exercise sliders instantly re-runs the Bergman ODE and re-renders the forward glycemic projection curve in real time.

---

## 8. Automated Verification & Continuous Integration

GlucoTwin enforces strict test-driven engineering standards. All 38 automated tests pass locally and in GitHub Actions CI across multiple Python versions:

```
============================= test session starts =============================
platform win32 -- Python 3.10 / 3.11 / 3.14 -- pytest-9.1.1
rootdir: C:\Users\Ronak Jain\Desktop\glucotwin, configfile: pytest.ini
collected 38 items

tests\test_dashboard_integration.py ....                                 [ 10%]
tests\test_glucotwin.py ..................................               [100%]

============================= 38 passed in 10.40s =============================
```

### Breakdown of Test Suites
- **34 Machine Learning Engine Unit Tests (`tests/test_glucotwin.py`)**:
  - `test_raw_data_loader_presence`: Validates data schema across HUPA-UCM patient records.
  - `test_leakage_safe_preprocessing`: Proves zero future lookahead across 5-min interpolation caps.
  - `test_feature_engineering_shape`: Verifies 50-feature stack generation.
  - `test_lopo_split_patient_independence`: Enforces zero subject overlap across CV folds.
  - `test_calibration_improves_brier_and_ece`: Confirms isotonic regression reduces calibration error.
  - `test_alert_state_machine_hysteresis`: Validates 8-point gap and 15-min cooldown logic.
  - `test_patient_memory_temporal_safety`: Ensures episode retrieval strictly queries $t_{\text{past}} < T$.
  - `test_trajectory_ttt_accuracy`: Checks ODE simulation adherence against empirical crossings.
  - `test_ood_detection_boundary`: Tests Mahalanobis displacement on outlier vectors.
  - `test_twin_state_v6_serialization`: Verifies complete JSON serialization resilience of TwinState contracts.
- **4 Dashboard Integration Tests (`tests/test_dashboard_integration.py`)**:
  - `test_dashboard_production_artifacts`: Verifies LightGBM and calibrator pickle loading.
  - `test_dashboard_live_engine_snapshot`: Confirms live twin state construction for patient monitoring.
  - `test_dashboard_simulator_interventions`: Tests What-If ODE response to carb rescue and insulin bolus.
  - `test_dashboard_demo_fallback_payload`: Validates structural integrity of the zero-crash fallback JSON.

---

## 9. Clinical Scope & Ethical Research Disclaimers

GlucoTwin has been engineered under strict scientific and biomedical ethics standards:
1. **Academic Research Prototype**: GlucoTwin is an academic submission developed for the CRP Digital Twin Challenge 2026. It is **NOT** a certified medical device and has not received FDA clearance or CE mark approval.
2. **Observational Cohort**: The model was trained and evaluated on retrospective observational data from 25 Spanish adult subjects with Type 1 Diabetes from the HUPA-UCM dataset. Generalizability to pediatric, gestational, or Type 2 Diabetes populations has not been established.
3. **Hardware Specificity**: The model was calibrated on Abbott FreeStyle Libre 2 CGM data. Compatibility with other CGM hardware (Dexcom G6/G7, Medtronic Guardian) requires transfer calibration.
4. **Open-Loop Decision Support**: GlucoTwin provides advisory decision support only. It is **NOT** integrated into any automated insulin delivery (AID) or artificial pancreas closed-loop system.
5. **Scenario Simulations**: What-If metabolic projections are mathematical simulations based on simplified pharmacokinetic equations; they do not guarantee clinical prevention of hypoglycemic events.
6. **Physician Primacy**: GlucoTwin outputs must never be used to override standard CGM alarms, clinical judgment, or the advice of a certified healthcare professional.

---

## 10. How to Run Locally

### Prerequisites
- Python 3.10+ (Tested on 3.10, 3.11, and 3.14)
- Git & Virtualenv

### Setup Instructions
```bash
# 1. Clone the repository
git clone https://github.com/Ronak1167/glucotwin.git
cd glucotwin

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the automated test suite
pytest

# 4. Launch the Clinical Dashboard
streamlit run app.py
```
*The dashboard will automatically open in your default browser at `http://localhost:8501`.*

---

**Team GlucoTwin**  
*Ronak Jain & Aishni Rathore*  
*Jaypee Institute of Information Technology, Noida*  
*October 2026*

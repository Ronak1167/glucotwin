"""
GlucoTwin Demo Video Generator
Voices: Microsoft David (Ronak, Male) + Microsoft Zira (Aishni, Female)
Engine: Offline Windows SAPI5 via pyttsx3 + FFmpeg via imageio_ffmpeg
Output: docs/submission/GlucoTwin_Demo_Video.mp4
"""
import os, sys, wave, subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg
import pyttsx3

ROOT = Path(r"C:\Users\Ronak Jain\Desktop\glucotwin")
OUT_DIR = ROOT / "docs" / "submission"
AUDIO_TMP = OUT_DIR / "audio_segments"
SLIDES_TMP = OUT_DIR / "slide_images"
CLIPS_TMP = OUT_DIR / "video_clips"

AUDIO_TMP.mkdir(parents=True, exist_ok=True)
SLIDES_TMP.mkdir(parents=True, exist_ok=True)
CLIPS_TMP.mkdir(parents=True, exist_ok=True)

VOICE_RONAK = r"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_DAVID_11.0"
VOICE_AISHNI = r"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0"

W_VID, H_VID = 1280, 720
FPS = 24

# Font setup with fallback
def get_fonts():
    try:
        title_font = ImageFont.truetype(r"C:\Windows\Fonts\segoeuib.ttf", 32)
        heading_font = ImageFont.truetype(r"C:\Windows\Fonts\segoeuib.ttf", 22)
        body_font = ImageFont.truetype(r"C:\Windows\Fonts\segoeui.ttf", 17)
        badge_font = ImageFont.truetype(r"C:\Windows\Fonts\segoeuib.ttf", 13)
        small_font = ImageFont.truetype(r"C:\Windows\Fonts\segoeui.ttf", 12)
        metric_font = ImageFont.truetype(r"C:\Windows\Fonts\segoeuib.ttf", 26)
    except Exception:
        title_font = ImageFont.load_default(32)
        heading_font = ImageFont.load_default(22)
        body_font = ImageFont.load_default(17)
        badge_font = ImageFont.load_default(13)
        small_font = ImageFont.load_default(12)
        metric_font = ImageFont.load_default(26)
    return title_font, heading_font, body_font, badge_font, small_font, metric_font

# ──────────────────────────────────────────────────────────
# 20-SEGMENT SCRIPT & SLIDE DEFINITIONS
# ──────────────────────────────────────────────────────────
SEGMENTS = [
    # 00
    {
        "speaker": "R",
        "audio_text": """
Hello everyone, and welcome to our submission for the CRP Digital Twin Challenge 2026, organised by Happiest Health.
My name is Ronak Jain, and I am a student at the Jaypee Institute of Information Technology, Noida.
Today, my teammate Aishni Rathore and I will walk you through GlucoTwin — a Predictive Metabolic Digital Twin
for real-time hypoglycemia risk assessment in Type 1 Diabetes patients.
""",
        "category": "PROJECT INTRODUCTION",
        "title": "GlucoTwin — Predictive Metabolic Digital Twin",
        "subtitle": "Real-Time Hypoglycemia Risk Assessment & In-Silico Intervention Simulation",
        "badge_speaker": "Ronak Jain  |  ML Architecture & Lead Engineer",
        "speaker_color": (37, 99, 235),
        "cards": [
            {
                "title": "Core System Capabilities",
                "accent": (37, 99, 235),
                "items": [
                    "Patient-Independent ML Prediction: 60-min advance hypoglycemia risk forecasting",
                    "Isotonic Probability Calibration: Brier score reduced by 43.5%, ECE by 94.5%",
                    "Stateful Alert Architecture: 3-state hysteresis reducing daily false alarms by 85%",
                    "Metabolic Digital Twin: 2-compartment Bergman ODE counterfactual simulator",
                    "Continuous Patient Memory: Strict temporal backward-only case retrieval (< T)",
                    "Explainable & Safe: Live SHAP TreeExplainer attribution & Mahalanobis OOD guard"
                ]
            },
            {
                "title": "Clinical & Hackathon Foundation",
                "accent": (16, 185, 129),
                "items": [
                    "Event: CRP Digital Twin Challenge 2026 (Organised by Happiest Health)",
                    "Institution: Jaypee Institute of Information Technology (JIIT), Noida",
                    "Dataset: HUPA-UCM Diabetes Cohort (25 T1D patients, CGM + CSII/MDI + Fitbit)",
                    "Verification: 38/38 CI tests passing on Python 3.10 & 3.11 with green GitHub build",
                    "Clinical UI: Live interactive Streamlit dashboard with What-If scenario sliders"
                ]
            }
        ]
    },

    # 01
    {
        "speaker": "A",
        "audio_text": """
Hi, I am Aishni Rathore, also from Jaypee Institute of Information Technology, Noida.
GlucoTwin is the result of weeks of collaborative engineering — Ronak built the entire machine learning backend,
digital twin engine, and scientific validation pipeline, while I designed and implemented the clinical dashboard
and integration test suite.
Let us start with the problem we are solving.
""",
        "category": "TEAM COLLABORATION & DIVISION OF ROLES",
        "title": "Collaborative Engineering Architecture",
        "subtitle": "Systematic End-to-End Division of Labor Across ML Core & Clinical Dashboard",
        "badge_speaker": "Aishni Rathore  |  Dashboard Design & Integration Lead",
        "speaker_color": (147, 51, 234),
        "cards": [
            {
                "title": "Ronak Jain — Machine Learning & Twin Backend",
                "accent": (37, 99, 235),
                "items": [
                    "End-to-End Leakage-Safe Data Ingestion & Preprocessing (HUPA-UCM cohort)",
                    "50-Feature Metabolic Engineering (CGM dynamics, IOB/COB, circadian phase)",
                    "LightGBM Classifier + RandomForest Fallback + Isotonic Probability Calibrator",
                    "Damped-Velocity Trajectory Modeling (TTT) validated over 11,471 episode pairs",
                    "Alert State Machine FSM with 8-point hysteresis & 15-min cooldown",
                    "TwinState v6 Contract API with embedded Model Self-Awareness metadata"
                ]
            },
            {
                "title": "Aishni Rathore — Dashboard, UI/UX & Quality Assurance",
                "accent": (147, 51, 234),
                "items": [
                    "Interactive Clinical Dashboard built with Streamlit and Plotly visualizations",
                    "Section 1-3: Live Patient Status, Multi-Horizon Risk (30-120m), 3h CGM Trends",
                    "Section 4: What-If Metabolic Simulator (Standard Presets + Custom Sliders)",
                    "Section 5-7: Live SHAP Impact Panel, Patient Memory Timeline, OOD Guardrails",
                    "Zero-Crash Demo Fallback Mode with dynamic diagnostic inspection banner",
                    "Dashboard Integration Test Suite (4 tests) & Comprehensive PR Review Verification"
                ]
            }
        ]
    },

    # 02
    {
        "speaker": "R",
        "audio_text": """
Type 1 Diabetes patients face an invisible danger — hypoglycemia, which is when blood glucose drops
below 70 milligrams per deciliter, or the ADA Level 1 threshold.
This is particularly dangerous during sleep, when the patient cannot perceive warning symptoms.

Standard continuous glucose monitors alert when the glucose crosses a threshold in real time.
But our baseline analysis on the HUPA-UCM cohort revealed a severe problem:
point-in-time threshold alerts generated 29 point 21 raw false alarms per day.
That is alert fatigue so extreme that patients start ignoring the device.

Beyond false alarms, conventional systems have three fundamental gaps.
First, they have no physiological context. They ignore active insulin on board,
unabsorbed carbohydrates, and circadian sleep vulnerability.
Second, they cannot answer the key clinical question: what if the patient takes 15 grams of fast-acting carbohydrates?
Third, every moment is evaluated in isolation — there is no memory of past similar episodes.
GlucoTwin was built to solve all three of these gaps simultaneously.
""",
        "category": "PROBLEM STATEMENT & CLINICAL CONTEXT",
        "title": "The Invisible Danger: Hypoglycemia & Alert Fatigue",
        "subtitle": "Clinical Gap Analysis on 25 Real-World Type 1 Diabetes Patients",
        "badge_speaker": "Ronak Jain  |  ML Architecture & Lead Engineer",
        "speaker_color": (37, 99, 235),
        "cards": [
            {
                "title": "The Clinical Crisis: Nocturnal Hypo & Alert Burden",
                "accent": (239, 68, 68),
                "items": [
                    "Severe Hypoglycemia Danger: Glucose < 70 mg/dL causes seizures, coma, neuroglycopenia",
                    "Nocturnal Vulnerability: Patients are asleep and cannot perceive autonomic warning signs",
                    "Baseline Alert Fatigue: 29.21 raw threshold crossings per patient-day in HUPA-UCM",
                    "Critical Consequence: Alarm fatigue leads patients to mute or discard CGM devices"
                ]
            },
            {
                "title": "3 Fundamental Deficiencies of Existing CGM Solutions",
                "accent": (245, 158, 11),
                "items": [
                    "Gap 1 — Zero Physiological Context: Ignores active IOB, COB, circadian sensitivity",
                    "Gap 2 — Zero Counterfactual Power: Cannot project 'What if I eat 15g carbs now?'",
                    "Gap 3 — Stateless Evaluation: Evaluates every reading in a vacuum without patient history",
                    "The GlucoTwin Solution: Context-aware, stateful, simulated, and personalized digital twin"
                ]
            }
        ]
    },

    # 03
    {
        "speaker": "R",
        "audio_text": """
Let me now explain the system architecture of GlucoTwin from the ground up.

The data foundation is the HUPA-UCM Diabetes Dataset, published on Mendeley Data under DOI 10 point 17632 slash 3hbcscwz44 point 1.
This dataset covers 25 Type 1 Diabetes patients, each equipped with an Abbott FreeStyle Libre 2 continuous glucose monitor,
a subcutaneous insulin infusion pump or multiple daily injections regimen, and a Fitbit Ionic wristband.

The raw data flows through a leakage-safe preprocessing pipeline.
We enforce strict 5-minute interpolation caps, timestamp-sorted forward label generation
where labels for a time-point T use only data in the window T to T plus 60 minutes,
and all rolling windows use closed-left guards to guarantee zero data leakage from future observations.
""",
        "category": "DATASET & PREPROCESSING PIPELINE",
        "title": "HUPA-UCM Data Foundation & Zero-Leakage Pipeline",
        "subtitle": "Strict Mathematical Isolation of Past vs Future Observations",
        "badge_speaker": "Ronak Jain  |  ML Architecture & Lead Engineer",
        "speaker_color": (37, 99, 235),
        "cards": [
            {
                "title": "HUPA-UCM Cohort Specifications",
                "accent": (6, 182, 212),
                "items": [
                    "Citation & DOI: Mendeley Data, DOI: 10.17632/3hbcscwz44.1",
                    "Population: 25 adult patients diagnosed with Type 1 Diabetes",
                    "CGM Device: Abbott FreeStyle Libre 2 flash/continuous glucose sensor",
                    "Insulin Regimen: CSII continuous subcutaneous infusion pumps & MDI regimens",
                    "Wearable Sensor: Fitbit Ionic recording heart rate, step count, and sleep epochs"
                ]
            },
            {
                "title": "Rigorous Leakage Prevention Invariants",
                "accent": (16, 185, 129),
                "items": [
                    "Strict 5-Min Interpolation Cap: Missing CGM gaps are never artificially bridged",
                    "Closed-Left Rolling Windows: Feature computations strictly evaluate (T - W, T]",
                    "Forward-Only Target Window: Ground-truth label y(T) = 1 if min(glucose in [T, T+60]) < 70",
                    "Temporal Chronological Sorting: Monotonic time order prevents cross-contamination",
                    "Zero Future Peeking: Mathematically provable separation verified by unit test suite"
                ]
            }
        ]
    },

    # 04
    {
        "speaker": "R",
        "audio_text": """
From the cleaned time series, we engineer a 50-feature stack.
This includes CGM rolling statistics — means, standard deviations, and minimums over multiple windows.
We compute 15-minute and 30-minute glucose velocity and acceleration.
We approximate insulin on board and carbohydrate on board from the pump event logs.
We encode circadian phase using sine and cosine projections of the time of day.
And critically, we compute an expanding baseline mean personalised to each patient —
because what is a dangerous glucose level for one patient may be normal for another.

This feature stack feeds two parallel components: the ML prediction engine and the metabolic simulator.
""",
        "category": "FEATURE ENGINEERING",
        "title": "50-Feature Metabolic Feature Stack",
        "subtitle": "Combining Glycemic Dynamics, Pharmacokinetics, and Circadian Rhythms",
        "badge_speaker": "Ronak Jain  |  ML Architecture & Lead Engineer",
        "speaker_color": (37, 99, 235),
        "cards": [
            {
                "title": "Glycemic Dynamics & Rate of Change",
                "accent": (37, 99, 235),
                "items": [
                    "Multi-Window Statistics: Rolling mean, std dev, min, max over 15, 30, 60, 120 mins",
                    "Glucose Velocity: 15-min and 30-min first derivative (dG/dt in mg/dL/min)",
                    "Glucose Acceleration: Second derivative indicating rate of glycemic plunge",
                    "Low Glucose Index (LBGI): Non-linear clinical risk transformation of raw values"
                ]
            },
            {
                "title": "Physiological & Circadian Context",
                "accent": (16, 185, 129),
                "items": [
                    "Insulin on Board (IOB): Exponential pharmacokinetic clearance proxy from bolus events",
                    "Carbohydrates on Board (COB): Two-phase gastrointestinal absorption decay curve",
                    "Circadian Cycle Vectors: Continuous sin(2pi*t/24) and cos(2pi*t/24) encoding",
                    "Nocturnal Risk Flag: Dedicated binary indicators for 00:00 to 06:00 vulnerability window",
                    "Personalized Baseline: Expanding cumulative mean and EWM tracking individual set-points"
                ]
            }
        ]
    },

    # 05
    {
        "speaker": "R",
        "audio_text": """
The machine learning engine is built on LightGBM as the primary classifier,
with a RandomForest ensemble providing a dynamic fallback gate.
After training, we fit an isotonic probability calibrator on a validation set held out from training.

The calibration results are remarkable.
The Brier score — a measure of probabilistic forecast accuracy — dropped from 0.069 to 0.039,
a reduction of 43 point 5 percent.
The Expected Calibration Error, or ECE, dropped from 0.1006 to just 0.0055 —
a reduction of 94 point 5 percent.
This means the model's predicted probabilities now closely match the true empirical frequencies.

To validate patient-independence, we ran Leave-One-Patient-Out cross-validation across all 24 eligible patients.
The mean ROC-AUC was 0.970 plus or minus 0.027.
The mean sensitivity was 0.958 and mean specificity was 0.821.
These are hospital-grade classification metrics achieved without any patient-specific tuning.
""",
        "category": "MACHINE LEARNING & CALIBRATION",
        "title": "ML Engine & Isotonic Probability Calibration",
        "subtitle": "Leave-One-Patient-Out Cross-Validation & Hospital-Grade Reliability",
        "badge_speaker": "Ronak Jain  |  ML Architecture & Lead Engineer",
        "speaker_color": (37, 99, 235),
        "cards": [
            {
                "title": "Isotonic Probability Calibration Impact",
                "accent": (16, 185, 129),
                "items": [
                    "Brier Score Reduction: 0.0690 → 0.0390 (43.5% error reduction)",
                    "Expected Calibration Error (ECE): 0.1006 → 0.0055 (94.5% error reduction!)",
                    "Clinical Significance: Predicted probabilities reflect true empirical clinical frequency",
                    "Reliability: Clinicians can trust a 70% risk alert to reflect true 70% population event rate"
                ]
            },
            {
                "title": "Leave-One-Patient-Out (LOPO) Benchmark (24 Patients)",
                "accent": (37, 99, 235),
                "items": [
                    "Mean ROC-AUC: 0.970 ± 0.027 (Hospital-grade discrimination across unseen subjects)",
                    "Mean PR-AUC: 0.858 ± 0.099 (Superior performance on heavily imbalanced class data)",
                    "Mean Sensitivity (Recall): 0.958 ± 0.041 (Minimizes dangerous missed hypoglycemic events)",
                    "Mean Specificity: 0.821 ± 0.088 (Suppresses false alerts during euglycemic states)",
                    "Mean F2 Score: 0.754 ± 0.065 (Double penalty weighting on false negatives)"
                ]
            }
        ]
    },

    # 06
    {
        "speaker": "R",
        "audio_text": """
Now let me describe the three core Digital Twin components that differentiate GlucoTwin
from a standard machine learning classifier.

The first differentiator is Continuous Patient Memory.
The PatientMemory module maintains a persistent log of past alert state transitions and risk episodes.
When the twin evaluates a new moment, it retrieves the top three most similar historical episodes
from that specific patient's own history — using strict backward-only access, never touching future data.
This personalises the twin's situational awareness in a way that a stateless classifier cannot.

The second differentiator is Trajectory Modeling with Validated Uncertainty.
GlucoTwin models the glucose trajectory using a damped-velocity ordinary differential equation,
with a decay constant k equal to 0.015 per minute.
We validated this trajectory model on the full cohort of 24 patients,
evaluating 11,471 matched trajectory pairs.
The full-cohort Mean Absolute Error was 32.55 minutes, with a median of 17.60 minutes.
46 percent of predictions were within 15 minutes of the true threshold crossing.
We report stratified uncertainty tiers — LOW and MODERATE — to communicate confidence honestly.
""",
        "category": "DIGITAL TWIN DIFFERENTIATORS (PART 1)",
        "title": "Continuous Patient Memory & Validated Trajectory ODE",
        "subtitle": "Retrospective Context Retrieval & Mathematical Time-to-Threshold Modeling",
        "badge_speaker": "Ronak Jain  |  ML Architecture & Lead Engineer",
        "speaker_color": (37, 99, 235),
        "cards": [
            {
                "title": "Differentiator 1: Continuous Patient Memory",
                "accent": (6, 182, 212),
                "items": [
                    "Historical State Persistence: Logs past alert state transitions and hypoglycemia events",
                    "Strict Temporal Safety: Search space strictly bounded to past records (t_past < T)",
                    "Top-3 Similar Episode Retrieval: Euclidean matching on glucose, velocity, and time-of-day",
                    "Personalized Grounding: Provides patient-specific context without modifying model weights"
                ]
            },
            {
                "title": "Differentiator 2: Damped-Velocity Trajectory ODE",
                "accent": (245, 158, 11),
                "items": [
                    "Mathematical Formulation: dG/dt = v0 * exp(-k*t), with decay constant k = 0.015 min⁻¹",
                    "Cohort Validation: Evaluated across 11,471 matched trajectory pairs on 24 patients",
                    "Cohort Mean Absolute Error (MAE): 32.55 min (Median MAE: 17.60 min)",
                    "Clinical Accuracy: 46% of all TTT predictions within ±15 minutes of threshold crossing",
                    "Honest Uncertainty Tiers: Stratified LOW and MODERATE bounds communicated to clinician"
                ]
            }
        ]
    },

    # 07
    {
        "speaker": "R",
        "audio_text": """
The Alert State Machine is our fourth major differentiator.
Instead of firing an alarm every time glucose dips below a raw threshold,
GlucoTwin uses a three-state finite state machine: NORMAL, WATCH, and ALARM.

Entry into the WATCH state requires the calibrated risk probability to exceed 0.13 for at least one sustained interval.
Exiting back to NORMAL requires the probability to drop below 0.08 — an 8-point hysteresis band.
A 15-minute cooldown prevents rapid re-entry after resolution.

The effect is dramatic.
Raw point-level threshold crossings generated 29.21 alarms per day.
Our stateful alert machine produced just 4.39 clinically meaningful alert episodes per day —
an 85 percent reduction in alarm burden — while retrospectively detecting all 300 documented hypoglycemic episodes
in the evaluated cohort with 100 percent recall.

The fifth differentiator is Model Self-Awareness and OOD Guardrails.
The TwinState version 6 API includes a complete model self-awareness block
that exposes all empirical metrics I just described — LOPO AUC, Brier scores, TTT MAE, horizon scaling caveats —
directly in every inference response.
The OOD Detector computes a Mahalanobis distance of the current feature vector from the training distribution
and scores a twin_confidence value between 0.0 and 1.0, so clinicians know when to trust the output less.
""",
        "category": "DIGITAL TWIN DIFFERENTIATORS (PART 2)",
        "title": "Stateful Alert FSM & OOD Guardrails",
        "subtitle": "85% Alarm Burden Reduction with 100% Retrospective Episode Recall",
        "badge_speaker": "Ronak Jain  |  ML Architecture & Lead Engineer",
        "speaker_color": (37, 99, 235),
        "cards": [
            {
                "title": "Alert State Machine (FSM) Hysteresis",
                "accent": (16, 185, 129),
                "items": [
                    "3-State Architecture: NORMAL → WATCH (p > 0.13) → ALARM (sustained risk)",
                    "8-Point Hysteresis Gap: Return to NORMAL requires probability to drop below 0.08",
                    "15-Minute Cooldown Guard: Eliminates rapid chattering alert transitions",
                    "Dramatic Alarm Reduction: 29.21 alarms/day → 4.39 episodes/day (85% reduction!)",
                    "Zero Safety Compromise: 300/300 documented hypo episodes detected (100% recall)"
                ]
            },
            {
                "title": "Out-of-Distribution (OOD) Guard & Self-Awareness",
                "accent": (239, 68, 68),
                "items": [
                    "Mahalanobis Distance: Measures feature displacement from training covariance matrix",
                    "Twin Confidence Metric: Normalized 0.0 to 1.0 confidence score returned on every query",
                    "CGM Continuity Check: Detects missing sample streams and sensor detachments",
                    "Model Self-Awareness Block: Version 6 API payload exposes empirical LOPO benchmarks directly"
                ]
            }
        ]
    },

    # 08
    {
        "speaker": "A",
        "audio_text": """
Now let me walk you through the GlucoTwin clinical dashboard, which I built using Streamlit and Plotly.

When you launch the dashboard with the command 'streamlit run app.py' from the project root,
you are immediately presented with the patient monitor interface.

On the left sidebar, you can select any of the 25 HUPA-UCM patients from a dropdown.
Below the selector, you see the patient ID, the timestamp of the latest glucose reading,
and whether we are in a nocturnal or daytime context.

The sidebar also shows the execution engine mode. In the default Live Inference mode,
the dashboard loads the production LightGBM classifier and isotonic calibrator from the models folder,
processes the patient's actual time-series data, and computes everything in real time.
""",
        "category": "CLINICAL DASHBOARD ARCHITECTURE",
        "title": "Clinical Dashboard Architecture & Navigation",
        "subtitle": "Built with Streamlit & Plotly: Real-Time Monitoring & Dual Execution Modes",
        "badge_speaker": "Aishni Rathore  |  Dashboard Design & Integration Lead",
        "speaker_color": (147, 51, 234),
        "cards": [
            {
                "title": "Interactive Navigation & Sidebar Controls",
                "accent": (147, 51, 234),
                "items": [
                    "Instant Launch: Simple 'streamlit run app.py' command with zero complex setup",
                    "Multi-Patient Dropdown: Select between all 25 clinical subjects (HUPA0001P - HUPA0025P)",
                    "Context Awareness: Displays real-time observation timestamp and Nocturnal status",
                    "Patient Demographic Details: Age, gender, diabetes duration, insulin delivery mode"
                ]
            },
            {
                "title": "Dual Execution Engine Pipeline",
                "accent": (6, 182, 212),
                "items": [
                    "Live Inference Mode (Primary): Loads production LightGBM & Isotonic calibrators",
                    "Real-Time Feature Compute: Calculates 50-feature stack directly on selected patient data",
                    "Demo Fallback Mode (Fail-Safe): Switches to verified snapshot if data/models are missing",
                    "Status Banner: Green verified inference confirmation indicating active live calibrator"
                ]
            }
        ]
    },

    # 09
    {
        "speaker": "A",
        "audio_text": """
The main panel opens with a status banner.
In live inference mode, a green banner confirms that the digital twin is actively tracking the selected patient.
The banner shows the exact timestamp being evaluated and confirms that the isotonic calibrator is active.

Section 1 is the Current Patient State panel.
It displays five live metrics: current glucose in milligrams per deciliter, 15-minute glucose velocity,
the calibrated risk score out of 100, the alert status — which is NORMAL, WATCH, WARNING, or CRITICAL —
and the glucose trend classification: Stable, Falling, Rapidly Falling, Rising, or Rapidly Rising.

For example, for patient HUPA0004P at timestamp 2018-07-20, we see a glucose of 131 milligrams per deciliter,
a slight downward velocity of minus 0.3 milligrams per deciliter per minute,
a risk score of 0.5 out of 100, and an alert status of NORMAL. Completely safe.
""",
        "category": "DASHBOARD SECTION 1: LIVE MONITOR",
        "title": "Section 1 — Current Patient State Panel",
        "subtitle": "Five Real-Time Clinical Vitals & Calibrated Alert Tiering",
        "badge_speaker": "Aishni Rathore  |  Dashboard Design & Integration Lead",
        "speaker_color": (147, 51, 234),
        "cards": [
            {
                "title": "5 Live Clinical Metric Cards",
                "accent": (37, 99, 235),
                "items": [
                    "Current Blood Glucose: Displayed in mg/dL with color-coded glycemic danger zones",
                    "15-Minute Glucose Velocity: Rate of change in mg/dL/min (e.g. -0.3 mg/dL/min)",
                    "Calibrated Hypo Risk Score: 0 to 100 probabilistic scale calibrated with isotonic regression",
                    "Alert Machine Status: NORMAL (green), WATCH (yellow), WARNING (orange), CRITICAL (red)",
                    "Glycemic Trend: Classified as Stable, Falling, Rapidly Falling, Rising, Rapidly Rising"
                ]
            },
            {
                "title": "Clinical Status Banner & Live Inspection",
                "accent": (16, 185, 129),
                "items": [
                    "Live Tracking Badge: Green verified badge confirms active model computation",
                    "Timestamp Precision: Evaluates specific clinical moment with zero lag",
                    "Exemplar Patient HUPA0004P: Glucose 131 mg/dL, Velocity -0.3 mg/dL/min, Risk 0.5/100",
                    "Clear Risk Stratification: Immediate situational clarity for healthcare providers"
                ]
            }
        ]
    },

    # 10
    {
        "speaker": "A",
        "audio_text": """
Below the current state, Section 2 shows the Multi-Horizon Risk Predictions.
GlucoTwin generates risk estimates for 30, 60, 90, and 120 minutes ahead.
The 60-minute horizon is the primary model output, computed with the isotonic calibrator.
The 30, 90, and 120 minute estimates are derived projections from the primary model,
clearly labeled to communicate that they are scaled extrapolations, not independently trained models.

Section 3 is the Glucose Trend Chart — an interactive Plotly time series visualization
showing the patient's glucose trajectory over the last 3 hours,
with color-coded regions marking the hypoglycemic zone below 70 and the near-hypo zone below 80.
Historical hypo events from the patient's record are marked with red triangles.
""",
        "category": "DASHBOARD SECTIONS 2 & 3: PREDICTIONS & TRENDS",
        "title": "Multi-Horizon Risk & Interactive Trend Chart",
        "subtitle": "Forecast Extrapolations Across 30–120 Minutes & Dynamic Plotly Visualization",
        "badge_speaker": "Aishni Rathore  |  Dashboard Design & Integration Lead",
        "speaker_color": (147, 51, 234),
        "cards": [
            {
                "title": "Section 2: Multi-Horizon Risk Estimates",
                "accent": (6, 182, 212),
                "items": [
                    "30-Minute Horizon: Short-term risk projection reflecting immediate downward momentum",
                    "60-Minute Horizon (Primary): Core model output calibrated via isotonic regression",
                    "90-Minute & 120-Minute Horizons: Forward scaling curves for nocturnal sleep planning",
                    "Transparent Clinical Honesty: Labeled as scaled projections, avoiding false precision"
                ]
            },
            {
                "title": "Section 3: Interactive 3-Hour Glucose Trend Chart",
                "accent": (245, 158, 11),
                "items": [
                    "Dynamic Plotly Engine: Full zoom, hover, inspect, and pan capabilities",
                    "Glycemic Bands: Red zone (< 70 mg/dL hypo), Yellow band (70-80 mg/dL near-hypo)",
                    "Retrospective CGM Line: High-resolution 3-hour patient time series",
                    "Documented Hypo Markers: Historical events marked with prominent red indicator triangles"
                ]
            }
        ]
    },

    # 11
    {
        "speaker": "A",
        "audio_text": """
Section 4 is the What-If Interactive Metabolic Simulator — one of the most powerful features of GlucoTwin.
I implemented two simulation modes here.

The Standardised Interventions panel instantly runs the three pre-computed scenarios:
baseline, 15 grams fast carbs, and 25 grams snack, and shows the 120-minute projected trajectories
side by side in a Plotly chart, with the 70 milligram threshold clearly marked.

Below that, the Custom Intervention Sliders panel lets you configure your own scenario live.
You can adjust carbohydrate intake from 0 to 80 grams, select fast or complex absorption,
add 0 to 8 units of bolus insulin, set an exercise intensity, and choose a horizon from 30 to 180 minutes.
The dashboard recalculates and re-renders the counterfactual trajectory every time you move a slider.
""",
        "category": "DASHBOARD SECTION 4: WHAT-IF SIMULATOR",
        "title": "Section 4 — What-If Metabolic Simulator",
        "subtitle": "Counterfactual In-Silico Scenario Testing with Interactive Sliders",
        "badge_speaker": "Aishni Rathore  |  Dashboard Design & Integration Lead",
        "speaker_color": (147, 51, 234),
        "cards": [
            {
                "title": "Standardized Clinical Scenarios Panel",
                "accent": (16, 185, 129),
                "items": [
                    "Baseline Projection: Natural unassisted metabolic trajectory under current velocity",
                    "15g Fast Carbs (Rule of 15): Standard rapid glucose rescue scenario",
                    "25g Bedtime Complex Snack: Sustained slow-release carbohydrate absorption profile",
                    "Threshold Collision Test: Evaluates if trajectory avoids the 70 mg/dL hypo line"
                ]
            },
            {
                "title": "Custom Interactive Intervention Sliders",
                "accent": (147, 51, 234),
                "items": [
                    "Carbohydrate Dosage Slider: 0 to 80 grams (Fast-acting vs Complex starch)",
                    "Bolus Insulin Injection: 0.0 to 8.0 Units of rapid-acting subcutaneous insulin",
                    "Exercise Intensity Selector: Sedentary, Mild, Moderate, High metabolic demand",
                    "Simulation Horizon: 30 to 180 minutes forward ODE integration",
                    "Instant Dynamic Rerendering: Real-time Bergman pharmacokinetic updates on slider change"
                ]
            }
        ]
    },

    # 12
    {
        "speaker": "A",
        "audio_text": """
Section 5 is the SHAP Explainability panel.
GlucoTwin computes SHAP TreeExplainer values live for every inference call.
The dashboard displays the top risk-driving features for the current prediction —
each shown with the feature name, its current value, the SHAP value magnitude,
and a color-coded direction indicator showing whether the feature is pushing risk up or down.

Section 6 is the Similar Historical Episodes panel.
The patient memory module retrieves the top 3 most similar past moments from this patient's own history —
based on glucose, velocity, and context similarity — and displays the glucose at that episode,
the alert level that occurred, and the time since that event.
This gives the clinician immediate comparative context from the same patient's lived experience.

Section 7 is the OOD Confidence Panel.
It shows the twin_confidence score, the Mahalanobis distance from the training distribution,
a CGM continuity check, and an overall OOD flag.
A high confidence score means the current observation is well within the model's training regime.
""",
        "category": "DASHBOARD SECTIONS 5, 6, 7: EXPLAINABILITY & TRUST",
        "title": "SHAP Explainability, Patient Memory & OOD Panel",
        "subtitle": "Transparent AI Decision Support Grounded in Clinical Interpretability",
        "badge_speaker": "Aishni Rathore  |  Dashboard Design & Integration Lead",
        "speaker_color": (147, 51, 234),
        "cards": [
            {
                "title": "Section 5: Live SHAP Feature Attribution",
                "accent": (37, 99, 235),
                "items": [
                    "Live TreeExplainer: Computes exact SHAP feature attributions on real-time queries",
                    "Top Risk Drivers: Highlights the most influential features governing the prediction",
                    "Directional Indicators: Green / Red bars showing whether features increase or mitigate risk",
                    "Feature Values: Exposes active raw sensor readings alongside their attribution weights"
                ]
            },
            {
                "title": "Section 6 & 7: Patient Memory & OOD Confidence",
                "accent": (245, 158, 11),
                "items": [
                    "Patient Memory: Top-3 historical episodes retrieved from patient's own retrospective past",
                    "Historical Comparison: Shows glucose, alert level reached, and days since episode",
                    "Mahalanobis Distance: Quantifies feature divergence from training population bounds",
                    "Twin Confidence Score: Transparent 0.0 to 1.0 reliability metric alerting clinician"
                ]
            }
        ]
    },

    # 13
    {
        "speaker": "A",
        "audio_text": """
Finally, the Demo Fallback Mode is a safety net I built into the dashboard architecture.
If the production model artifacts or patient dataset are absent —
for example when deploying to a new machine before data is loaded —
the dashboard automatically detects this and switches to a verified static demonstration payload
stored in data/dashboard_demo_payload.json.
A yellow banner clearly marks that the dashboard is in demo mode, and displays specific diagnostic details
about which component triggered the fallback.
This ensures the dashboard never crashes on a missing file — it always has something to show.

Multi-patient navigation works seamlessly. Simply change the patient ID in the sidebar dropdown,
and the entire dashboard — all panels, all charts, all SHAP explanations — refreshes automatically
for the newly selected patient, each with their own personalised twin state.
""",
        "category": "DASHBOARD RESILIENCE & MULTI-PATIENT UX",
        "title": "Zero-Crash Demo Fallback Mode & Seamless UX",
        "subtitle": "Guaranteed Reliability Across Any Deployment Environment",
        "badge_speaker": "Aishni Rathore  |  Dashboard Design & Integration Lead",
        "speaker_color": (147, 51, 234),
        "cards": [
            {
                "title": "Zero-Crash Demo Fallback Architecture",
                "accent": (16, 185, 129),
                "items": [
                    "Automatic Missing Artifact Detection: Gracefully intercepts missing model/dataset files",
                    "Verified Static Payload: Seamlessly falls back to data/dashboard_demo_payload.json",
                    "Diagnostic Inspection Banner: Clearly indicates fallback status and missing dependencies",
                    "Zero Blank Screens: Guaranteeing the UI functions perfectly for hackathon evaluators"
                ]
            },
            {
                "title": "Instant Multi-Patient Dynamic Navigation",
                "accent": (147, 51, 234),
                "items": [
                    "Sidebar Dropdown: Instant selection across all 25 clinical subjects",
                    "Full Reactive Rerender: Updates all 7 sections, charts, metrics, and SHAP weights",
                    "Personalized Set-Points: Reflects each patient's unique expanding baseline mean",
                    "Complete Clinical Isolation: Zero cross-talk between individual patient histories"
                ]
            }
        ]
    },

    # 14
    {
        "speaker": "R",
        "audio_text": """
Let me now summarise the complete empirical benchmark results for GlucoTwin.

In patient-independent Leave-One-Patient-Out cross-validation across 24 patients:
ROC-AUC of 0.970 plus or minus 0.027.
PR-AUC of 0.858 plus or minus 0.099.
Sensitivity of 0.958 plus or minus 0.041.
Specificity of 0.821 plus or minus 0.088.
F2 score of 0.754 plus or minus 0.065.

In the full-cohort chronological ablation on 305,556 samples:
ROC-AUC of 0.9468, PR-AUC of 0.8132, Sensitivity of 0.8628, Specificity of 0.8862.
Brier score of 0.0404, ECE of 0.0102.
Retrospective episode detection: 300 out of 300, which is 100 percent recall.
Mean advance lead time: 47.4 minutes. Median lead time: 48.8 minutes.
""",
        "category": "EMPIRICAL EVALUATION & VALIDATION",
        "title": "Comprehensive Empirical Benchmark Results",
        "subtitle": "Rigorous Scientific Evaluation on 305,556 Real-World Observations",
        "badge_speaker": "Ronak Jain  |  ML Architecture & Lead Engineer",
        "speaker_color": (37, 99, 235),
        "cards": [
            {
                "title": "Leave-One-Patient-Out (LOPO) CV (24 Patients)",
                "accent": (37, 99, 235),
                "items": [
                    "ROC-AUC: 0.970 ± 0.027 (Hospital-grade discrimination across unseen subjects)",
                    "PR-AUC: 0.858 ± 0.099 (High precision-recall under severe class imbalance)",
                    "Sensitivity: 0.958 ± 0.041 (Detects 95.8% of all hypoglycemic episodes)",
                    "Specificity: 0.821 ± 0.088 (Suppresses false alarms during safe euglycemia)",
                    "F2 Score: 0.754 ± 0.065 (Prioritizes sensitivity over precision for patient safety)"
                ]
            },
            {
                "title": "Full-Cohort Chronological Validation (305,556 Samples)",
                "accent": (16, 185, 129),
                "items": [
                    "ROC-AUC: 0.9468  |  PR-AUC: 0.8132",
                    "Sensitivity: 0.8628  |  Specificity: 0.8862",
                    "Calibrated Brier Score: 0.0404  |  ECE: 0.0102",
                    "Episode Detection Recall: 300 / 300 (100% Retrospective Recall)",
                    "Advance Warning Lead Time: Mean 47.4 minutes  |  Median 48.8 minutes"
                ]
            }
        ]
    },

    # 15
    {
        "speaker": "R",
        "audio_text": """
The complete verification and test suite is available in the repository.
We have 38 pytest tests covering every component of the system — passing on both Python 3.10 and Python 3.11,
confirmed by our GitHub Actions continuous integration pipeline.

The test suite includes:
34 machine learning engine tests covering data loading, preprocessing, feature engineering,
classifier training, LOPO validation, isotonic calibration, SHAP explainability, alert state machine hysteresis,
OOD detection, patient memory temporal safety, TTT trajectory logic, and serialization resilience.
And 4 dashboard integration tests confirming that production model artifacts load correctly,
live engine snapshots are valid, the What-If simulator produces correct intervention dynamics,
and the demo fallback payload is structurally intact.

All files in the repository are publicly accessible at github dot com slash Ronak1167 slash glucotwin.
The CI badge confirms a passing status on the main branch.
""",
        "category": "SOFTWARE ENGINEERING & CI/CD",
        "title": "Comprehensive Test Suite & CI/CD Verification",
        "subtitle": "38 Automated Tests Passing Across Python 3.10 & 3.11 Environments",
        "badge_speaker": "Ronak Jain  |  ML Architecture & Lead Engineer",
        "speaker_color": (37, 99, 235),
        "cards": [
            {
                "title": "34 Machine Learning Engine Unit Tests",
                "accent": (37, 99, 235),
                "items": [
                    "Data Pipeline: Ingestion, 5-min interpolation caps, closed-left rolling windows",
                    "Feature Engineering: 50 features, velocity, acceleration, IOB/COB, circadian vectors",
                    "Model Training: LightGBM classification, LOPO validation splits, fallback gates",
                    "Calibration & Trust: Isotonic regression, ECE computation, Brier score reductions",
                    "Digital Twin Core: FSM hysteresis, OOD Mahalanobis distance, ODE trajectory accuracy",
                    "Memory & Contracts: Temporal backward guard (< T), TwinState v6 JSON serialization"
                ]
            },
            {
                "title": "4 Dashboard Integration Tests & GitHub Actions CI",
                "accent": (16, 185, 129),
                "items": [
                    "test_dashboard_production_artifacts: Confirms model files load in production",
                    "test_dashboard_live_engine_snapshot: Validates live patient twin state generation",
                    "test_dashboard_simulator_interventions: Confirms What-If ODE response to carbs/insulin",
                    "test_dashboard_demo_fallback_payload: Validates fallback payload integrity",
                    "GitHub Actions CI: Matrix build passing on both Python 3.10 and Python 3.11"
                ]
            }
        ]
    },

    # 16
    {
        "speaker": "A",
        "audio_text": """
Before we close, I want to briefly cover our important research disclaimers.

GlucoTwin is an academic research prototype developed for the CRP Digital Twin Challenge 2026.
It is not a medical device. It is not FDA cleared or CE marked.
All predictions are statistical model probabilities based on retrospective observational data
from 25 Spanish adults with Type 1 Diabetes.
The What-If simulations are forward mathematical projections under simplified pharmacokinetic assumptions —
they do not guarantee clinical prevention of a hypoglycemic event.

The model was trained and evaluated on FreeStyle Libre 2 CGM data.
Performance on other sensor hardware, pediatric populations, gestational diabetes, or Type 2 Diabetes is unknown.
GlucoTwin is not integrated into any automated insulin delivery loop.

Please never use GlucoTwin outputs to replace clinical judgment, standard CGM monitoring, or physician consultation.
""",
        "category": "ETHICS, SAFETY & CLINICAL SCOPE",
        "title": "Clinical Research Disclaimers & Limitations",
        "subtitle": "Academic Research Prototype — Open-Loop Decision Support Boundaries",
        "badge_speaker": "Aishni Rathore  |  Dashboard Design & Integration Lead",
        "speaker_color": (239, 68, 68),
        "cards": [
            {
                "title": "Regulatory & Device Status",
                "accent": (239, 68, 68),
                "items": [
                    "Academic Prototype Only: Developed for CRP Digital Twin Challenge 2026",
                    "Not FDA / CE Cleared: NOT a certified medical device; strictly for research evaluation",
                    "Observational Cohort: Evaluated on 25 Spanish adult T1D subjects from HUPA-UCM",
                    "Hardware Specificity: FreeStyle Libre 2 CGM data; performance on Dexcom/Medtronic unverified"
                ]
            },
            {
                "title": "Clinical Use Boundaries & Ethical Constraints",
                "accent": (245, 158, 11),
                "items": [
                    "Open-Loop Advisory: Purely advisory decision support; NOT linked to automated insulin pumps",
                    "Mathematical Scenarios: What-If simulations are ODE projections, NOT medical prescriptions",
                    "Statistical Probabilities: All outputs are empirical estimates, not guaranteed outcomes",
                    "Physician Primacy: Never replace clinical judgment, standard CGM alarms, or medical advice"
                ]
            }
        ]
    },

    # 17
    {
        "speaker": "R",
        "audio_text": """
To summarise GlucoTwin's unique contributions:

One: A fully patient-independent validated digital twin engine with LOPO ROC-AUC of 0.970
and isotonic probability calibration achieving 94.5 percent ECE reduction.

Two: A stateful hysteresis alert architecture reducing daily alarm burden by 85 percent
while maintaining 100 percent retrospective episode detection.

Three: A metabolic counterfactual simulation engine for in-silico rescue carb scenario planning.

Four: A complete TwinState version 6 API with full model self-awareness, SHAP explanations,
OOD confidence scoring, and multi-horizon predictions.

Five: A production-grade Streamlit clinical dashboard with live inference, interactive What-If sliders,
patient memory visualization, and a robust demo fallback mode.

Six: A rigorous 38-test CI-verified test suite passing on Python 3.10 and 3.11.

Thank you for watching. The complete source code, documentation, architecture diagram, and this presentation
are all available at github dot com slash Ronak1167 slash glucotwin.
""",
        "category": "EXECUTIVE SUMMARY",
        "title": "Summary of GlucoTwin Core Innovations",
        "subtitle": "Six Key Breakthroughs Transforming Hypoglycemia Management",
        "badge_speaker": "Ronak Jain  |  ML Architecture & Lead Engineer",
        "speaker_color": (37, 99, 235),
        "cards": [
            {
                "title": "1. Patient-Independent ML Engine",
                "accent": (37, 99, 235),
                "items": [
                    "LOPO ROC-AUC 0.970 ± 0.027 across 24 unseen patients",
                    "Isotonic calibration reduces ECE by 94.5% (0.1006 → 0.0055)"
                ]
            },
            {
                "title": "2. Stateful Alert Architecture",
                "accent": (16, 185, 129),
                "items": [
                    "3-State Hysteresis FSM eliminates 85% of false alarms (29.21 → 4.39/day)",
                    "100% retrospective recall detecting all 300 documented hypo episodes"
                ]
            },
            {
                "title": "3. In-Silico Metabolic Simulator",
                "accent": (245, 158, 11),
                "items": [
                    "Bergman 2-compartment pharmacokinetic counterfactual projections",
                    "Simulates rescue carbs (15g fast / 25g complex) & custom bolus/exercise"
                ]
            },
            {
                "title": "4. Production Clinical UI & Quality",
                "accent": (147, 51, 234),
                "items": [
                    "Interactive Streamlit dashboard with What-If sliders & SHAP attribution",
                    "38/38 passing CI tests on Python 3.10 & 3.11 with green GitHub Actions"
                ]
            }
        ]
    },

    # 18
    {
        "speaker": "A",
        "audio_text": """
Thank you so much for your time. I am Aishni Rathore.
""",
        "category": "ACKNOWLEDGMENT & CLOSING",
        "title": "Thank You from Team GlucoTwin",
        "subtitle": "Aishni Rathore  |  Dashboard Design & Integration Lead",
        "badge_speaker": "Aishni Rathore  |  Dashboard Design & Integration Lead",
        "speaker_color": (147, 51, 234),
        "cards": [
            {
                "title": "Aishni Rathore — Key Contributions",
                "accent": (147, 51, 234),
                "items": [
                    "Clinical Dashboard Architecture (Streamlit & Plotly dynamic visuals)",
                    "What-If Interactive Simulation Sliders & Scenario Visualizers",
                    "Dashboard Integration Testing Suite & Automated QA Fixtures",
                    "Demo Fallback Guardrail Implementation & UI State Management",
                    "Student, Jaypee Institute of Information Technology (JIIT), Noida"
                ]
            },
            {
                "title": "Acknowledgments & Appreciation",
                "accent": (16, 185, 129),
                "items": [
                    "Organisers: Happiest Health & CRP Digital Twin Challenge 2026 Committee",
                    "Data Contributors: Mendeley Data & HUPA-UCM Research Group",
                    "Faculty Mentors: Department of Computer Science & Engineering, JIIT Noida",
                    "Open Source Community: Streamlit, Plotly, LightGBM, and Scikit-Learn ecosystems"
                ]
            }
        ]
    },

    # 19
    {
        "speaker": "R",
        "audio_text": """
And I am Ronak Jain. We look forward to your feedback on GlucoTwin.
Thank you, Happiest Health and the CRP Digital Twin Challenge 2026 committee.
""",
        "category": "REPOSITORY & SUBMISSION ACCESS",
        "title": "Open Source Codebase & Submission Deliverables",
        "subtitle": "Complete Source Code, Models, Documentation, and Benchmarks Available Online",
        "badge_speaker": "Ronak Jain  |  ML Architecture & Lead Engineer",
        "speaker_color": (37, 99, 235),
        "cards": [
            {
                "title": "Ronak Jain — Key Contributions",
                "accent": (37, 99, 235),
                "items": [
                    "End-to-End Leakage-Safe Time Series Machine Learning Pipeline",
                    "Feature Engineering (50-feature stack with IOB, COB, circadian phase)",
                    "Isotonic Probability Calibration & LOPO Patient-Independence Validation",
                    "Alert State Machine FSM, Patient Memory Module & Damped-Velocity ODE",
                    "Student, Jaypee Institute of Information Technology (JIIT), Noida"
                ]
            },
            {
                "title": "Complete Submission Deliverables & Links",
                "accent": (6, 182, 212),
                "items": [
                    "GitHub Repository: https://github.com/Ronak1167/glucotwin",
                    "Continuous Integration: 38/38 Tests Passing (Python 3.10 & 3.11)",
                    "Architecture Diagram: docs/submission/GlucoTwin_Architecture_Diagram.pdf",
                    "Presentation Slide Deck: docs/submission/GlucoTwin_Presentation.pptx",
                    "Demonstration Video: docs/submission/GlucoTwin_Demo_Video.mp4",
                    "License: Open Source MIT License for Reproducible Medical Science"
                ]
            }
        ]
    }
]

# ──────────────────────────────────────────────────────────
# AUDIO GENERATION (win32com SAPI5 direct COM)
# ──────────────────────────────────────────────────────────
import win32com.client

def generate_audio_segments():
    print("[1/4] Generating speech audio segments via SAPI5 COM...")
    speaker = win32com.client.Dispatch("SAPI.SpVoice")
    speaker.Rate = 0  # Natural conversational pacing

    voices = speaker.GetVoices()
    david = [v for v in voices if "david" in v.GetDescription().lower()][0]
    zira = [v for v in voices if "zira" in v.GetDescription().lower()][0]

    audio_paths = []
    durations = []

    for i, seg in enumerate(SEGMENTS):
        spk = seg["speaker"]
        wav_path = AUDIO_TMP / f"seg_{i:03d}_{spk}.wav"

        speaker.Voice = david if spk == "R" else zira
        text = seg["audio_text"].strip()

        stream = win32com.client.Dispatch("SAPI.SpFileStream")
        stream.Open(str(wav_path), 3, False)  # 3 = SSFMCreateForWrite
        speaker.AudioOutputStream = stream
        speaker.Speak(text)
        stream.Close()

        # Measure duration with wave
        with wave.open(str(wav_path), "rb") as w:
            dur = w.getnframes() / float(w.getframerate())

        audio_paths.append(wav_path)
        durations.append(dur)
        print(f"  Segment {i:02d} ({spk}): {dur:.1f}s -> {wav_path.name}")

    print(f"Total audio duration: {sum(durations):.1f}s (~{sum(durations)/60:.1f} min)")
    return audio_paths, durations

# ──────────────────────────────────────────────────────────
# SLIDE IMAGE GENERATION (Pillow high-resolution dark UI)
# ──────────────────────────────────────────────────────────
def render_slide_image(idx, seg):
    title_font, heading_font, body_font, badge_font, small_font, metric_font = get_fonts()

    # Background: modern deep navy
    img = Image.new("RGB", (W_VID, H_VID), (10, 15, 29))
    draw = ImageDraw.Draw(img)

    # Top accent bar
    draw.rectangle([0, 0, W_VID, 6], fill=seg["speaker_color"])

    # Header category
    draw.text((50, 20), seg["category"].upper(), fill=(148, 163, 184), font=badge_font)

    # Speaker badge (top right)
    spk_box = [W_VID - 420, 16, W_VID - 50, 44]
    draw.rounded_rectangle(spk_box, radius=6, fill=(30, 41, 59), outline=seg["speaker_color"], width=1)
    draw.text((W_VID - 235, 30), seg["badge_speaker"], fill=(241, 245, 249), anchor="mm", font=badge_font)

    # Main Title
    draw.text((50, 46), seg["title"], fill=(255, 255, 255), font=title_font)
    # Subtitle
    draw.text((50, 88), seg["subtitle"], fill=(147, 197, 253), font=body_font)

    # Card layout (2 columns)
    cards = seg["cards"]
    card_y = 125
    card_h = 530

    if len(cards) == 2:
        col_w = 575
        col_gap = 30
        for c_idx, card in enumerate(cards):
            c_x = 50 + c_idx * (col_w + col_gap)
            # Card background
            draw.rounded_rectangle([c_x, card_y, c_x + col_w, card_y + card_h], radius=10,
                                   fill=(17, 24, 39), outline=(51, 65, 85), width=1)
            # Card header line
            draw.rounded_rectangle([c_x, card_y, c_x + col_w, card_y + 46], radius=10, fill=(30, 41, 59))
            draw.rectangle([c_x, card_y + 20, c_x + col_w, card_y + 46], fill=(30, 41, 59))
            draw.rectangle([c_x, card_y + 44, c_x + col_w, card_y + 46], fill=card["accent"])

            # Card title
            draw.text((c_x + 18, card_y + 23), card["title"], fill=(255, 255, 255), anchor="lm", font=heading_font)

            # Bullet items
            item_y = card_y + 60
            for item in card["items"]:
                # bullet bullet dot
                draw.ellipse([c_x + 18, item_y + 5, c_x + 24, item_y + 11], fill=card["accent"])
                # text wrap
                words = item.split()
                line = ""
                for w in words:
                    test_line = f"{line} {w}".strip()
                    bbox = draw.textbbox((0, 0), test_line, font=body_font)
                    if (bbox[2] - bbox[0]) > (col_w - 55):
                        draw.text((c_x + 32, item_y), line, fill=(226, 232, 240), font=body_font)
                        item_y += 24
                        line = w
                    else:
                        line = test_line
                if line:
                    draw.text((c_x + 32, item_y), line, fill=(226, 232, 240), font=body_font)
                    item_y += 30

    elif len(cards) == 4:
        # 4 cards (2x2 grid)
        col_w = 575
        col_h = 250
        gap_x = 30
        gap_y = 20
        coords = [
            (50, card_y),
            (50 + col_w + gap_x, card_y),
            (50, card_y + col_h + gap_y),
            (50 + col_w + gap_x, card_y + col_h + gap_y),
        ]
        for c_idx, card in enumerate(cards):
            c_x, c_y = coords[c_idx]
            draw.rounded_rectangle([c_x, c_y, c_x + col_w, c_y + col_h], radius=8,
                                   fill=(17, 24, 39), outline=(51, 65, 85), width=1)
            draw.rounded_rectangle([c_x, c_y, c_x + col_w, c_y + 38], radius=8, fill=(30, 41, 59))
            draw.rectangle([c_x, c_y + 18, c_x + col_w, c_y + 38], fill=(30, 41, 59))
            draw.rectangle([c_x, c_y + 36, c_x + col_w, c_y + 38], fill=card["accent"])
            draw.text((c_x + 14, c_y + 19), card["title"], fill=(255, 255, 255), anchor="lm", font=heading_font)

            item_y = c_y + 50
            for item in card["items"]:
                draw.ellipse([c_x + 14, item_y + 5, c_x + 20, item_y + 11], fill=card["accent"])
                words = item.split()
                line = ""
                for w in words:
                    test_line = f"{line} {w}".strip()
                    bbox = draw.textbbox((0, 0), test_line, font=body_font)
                    if (bbox[2] - bbox[0]) > (col_w - 45):
                        draw.text((c_x + 26, item_y), line, fill=(226, 232, 240), font=body_font)
                        item_y += 22
                        line = w
                    else:
                        line = test_line
                if line:
                    draw.text((c_x + 26, item_y), line, fill=(226, 232, 240), font=body_font)
                    item_y += 28

    # Bottom footer bar
    draw.rectangle([0, H_VID - 42, W_VID, H_VID], fill=(15, 23, 42))
    draw.rectangle([0, H_VID - 43, W_VID, H_VID - 42], fill=(51, 65, 85))

    footer_left = "GlucoTwin: Predictive Metabolic Digital Twin for T1D  |  CRP Digital Twin Challenge 2026"
    footer_mid = "Ronak Jain & Aishni Rathore (JIIT Noida)"
    footer_right = f"Slide {idx+1:02d} / {len(SEGMENTS):02d}"

    draw.text((50, H_VID - 21), footer_left, fill=(148, 163, 184), anchor="lm", font=small_font)
    draw.text((W_VID // 2 + 100, H_VID - 21), footer_mid, fill=(203, 213, 225), anchor="mm", font=small_font)
    draw.text((W_VID - 50, H_VID - 21), footer_right, fill=(148, 163, 184), anchor="rm", font=small_font)

    slide_path = SLIDES_TMP / f"slide_{idx:03d}.png"
    img.save(str(slide_path))
    return slide_path

# ──────────────────────────────────────────────────────────
# VIDEO CLIP GENERATION & CONCATENATION (FFmpeg)
# ──────────────────────────────────────────────────────────
def build_final_video(audio_paths, durations):
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    print(f"[2/4] Rendering {len(SEGMENTS)} slide visual images...")
    slide_paths = [render_slide_image(i, seg) for i, seg in enumerate(SEGMENTS)]

    print(f"[3/4] Encoding {len(SEGMENTS)} individual video clips...")
    clip_paths = []
    for i in range(len(SEGMENTS)):
        slide = slide_paths[i]
        audio = audio_paths[i]
        clip_out = CLIPS_TMP / f"clip_{i:03d}.mp4"

        cmd = [
            ffmpeg_exe, "-y",
            "-loop", "1",
            "-framerate", str(FPS),
            "-i", str(slide),
            "-i", str(audio),
            "-c:v", "libx264",
            "-tune", "stillimage",
            "-preset", "ultrafast",
            "-crf", "22",
            "-c:a", "aac",
            "-b:a", "192k",
            "-pix_fmt", "yuv420p",
            "-shortest",
            str(clip_out)
        ]
        res = subprocess.run(cmd, capture_output=True)
        if res.returncode != 0:
            print(f"Error encoding clip {i}: {res.stderr.decode('utf-8', errors='ignore')}")
            sys.exit(1)
        clip_paths.append(clip_out)
        print(f"  Encoded clip {i:02d} / {len(SEGMENTS):02d} ({durations[i]:.1f}s)")

    print("[4/4] Concatenating clips into final submission video...")
    concat_txt = OUT_DIR / "concat_clips.txt"
    with open(concat_txt, "w") as f:
        for c in clip_paths:
            # Format path for ffmpeg concat demuxer
            safe_path = str(c).replace("\\", "/")
            f.write(f"file '{safe_path}'\n")

    final_video = OUT_DIR / "GlucoTwin_Demo_Video.mp4"
    concat_cmd = [
        ffmpeg_exe, "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_txt),
        "-c", "copy",
        str(final_video)
    ]
    res = subprocess.run(concat_cmd, capture_output=True)
    if res.returncode != 0:
        print(f"Error concatenating clips: {res.stderr.decode('utf-8', errors='ignore')}")
        sys.exit(1)

    vid_size_mb = os.path.getsize(str(final_video)) / (1024 * 1024)
    print(f"\n=======================================================")
    print(f"[SUCCESS] Final Video Ready!")
    print(f"Path: {final_video}")
    print(f"File Size: {vid_size_mb:.2f} MB")
    print(f"=======================================================")
    return final_video

if __name__ == "__main__":
    audio_paths, durations = generate_audio_segments()
    build_final_video(audio_paths, durations)

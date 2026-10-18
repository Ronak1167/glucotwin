"""
GlucoTwin — Streamlit Digital Twin Dashboard
app.py

Complete end-to-end integration:
  - Connects to trained HypoClassifier production model & IsotonicCalibrator.
  - Feeds preprocessed patient time series from HUPA-UCM dataset.
  - Computes feature matrix, fits OOD detector, and indexes historical episodes.
  - Evaluates live Digital Twin state (risk, multi-horizon predictions, SHAP explainability).
  - Provides genuinely interactive what-if metabolic simulations.
  - Automatically falls back to static demonstration data if model/data are missing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GlucoTwin — Digital Twin Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom Styling ────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main-title {
    font-size: 2.3rem;
    font-weight: 700;
    margin-bottom: 0.1rem;
    color: #0f172a;
}
.subtitle {
    color: #475569;
    font-size: 1.05rem;
    margin-bottom: 1.2rem;
}
.metric-card {
    padding: 1rem;
    border-radius: 10px;
    border: 1px solid #e2e8f0;
    background: #f8fafc;
}
.status-pill-live {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 600;
    background-color: #dcfce7;
    color: #166534;
}
.status-pill-demo {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 600;
    background-color: #fef3c7;
    color: #92400e;
}
</style>
""", unsafe_allow_html=True)

logger = logging.getLogger(__name__)

# ── Project Paths ─────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
DATA_PATH = _ROOT / "data" / "dashboard_demo_payload.json"
MODEL_PATH = _ROOT / "models" / "glucotwin_production_model.joblib"
CALIBRATOR_PATH = _ROOT / "models" / "glucotwin_calibrator.joblib"
def _resolve_data_dir() -> Path:
    candidates = [
        _ROOT / "data" / "raw" / "hupa_ucm" / "Preprocessed",
        _ROOT / "data" / "raw" / "hupa_ucm" / "HUPA-UCM Diabetes Dataset" / "Preprocessed",
        _ROOT / "data" / "raw" / "hupa_ucm",
    ]
    for c in candidates:
        if c.exists() and any(c.glob("HUPA*.csv")):
            return c
    return candidates[0]

DATA_DIR = _resolve_data_dir()


# ── Utility Functions ─────────────────────────────────────────────────────────

def safe_float(value: Any, default: Optional[float] = 0.0) -> Optional[float]:
    """Safely convert value to float or return default if None/invalid."""
    if value is None:
        return default
    try:
        val = float(value)
        return default if np.isnan(val) else val
    except (TypeError, ValueError):
        return default


def get_available_patients() -> List[str]:
    """Discover preprocessed patient CSVs in the dataset directory."""
    if not DATA_DIR.exists():
        return ["HUPA0001P"]
    files = sorted(DATA_DIR.glob("HUPA*P.csv"))
    if not files:
        files = sorted(DATA_DIR.glob("HUPA*.csv"))
    patients = [f.stem for f in files]
    return patients if patients else ["HUPA0001P"]


# ── Cached Resource Loaders ───────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading trained GlucoTwin classifier...")
def load_production_classifier():
    """Load serialized production HypoClassifier bundle."""
    import joblib
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner="Loading and preprocessing patient record...", ttl=1800)
def load_patient_preprocessed(patient_id: str) -> pd.DataFrame:
    """Load and preprocess patient data with chronological sorting."""
    from src.data.loader import load_patient_raw, COLUMN_MAP
    from src.data.preprocessing import preprocess_patient

    patient_file = DATA_DIR / f"{patient_id}.csv"
    if not patient_file.exists():
        raise FileNotFoundError(f"Patient file not found: {patient_file}")

    df_raw = load_patient_raw(patient_file)
    df_pp = preprocess_patient(df_raw, COLUMN_MAP)
    if "timestamp" in df_pp.columns:
        df_pp = df_pp.sort_values("timestamp").reset_index(drop=True)
    return df_pp


@st.cache_resource(show_spinner="Initializing Digital Twin engine and historical memory...")
def initialize_digital_twin_engine(patient_id: str):
    """
    Constructs and fully initializes GlucoTwinEngine:
    - Loads production classifier
    - Loads patient record
    - Computes feature matrix
    - Fits OOD detector on patient feature distribution
    - Indexes historical episodes for precedent matching
    """
    from src.twin.engine import GlucoTwinEngine
    from src.interface.api import GlucoTwinAPI
    from src.data.loader import COLUMN_MAP
    from src.features.engineer import build_feature_matrix

    clf = load_production_classifier()
    engine = GlucoTwinEngine(
        patient_id=patient_id,
        classifier=clf,
        column_map=COLUMN_MAP,
    )

    try:
        df_pp = load_patient_preprocessed(patient_id)
        if len(df_pp) >= 20:
            X_all, _ = build_feature_matrix(df_pp, COLUMN_MAP, vectorized=True)
            engine.fit_ood_detector(X_all)
            engine.set_historical_index(df_pp, X_all)
    except Exception as exc:
        logger.warning(f"Could not index patient history or fit OOD for {patient_id}: {exc}")

    api = GlucoTwinAPI(engine=engine)
    return engine, api


def load_demo_fallback_data() -> Dict[str, Any]:
    """Load pre-computed static demonstration payload."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Demo payload missing at {DATA_PATH}")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Sidebar: Mode & Patient Selector ──────────────────────────────────────────

st.sidebar.header("Patient Monitor")

available_patients = get_available_patients()
default_patient = "HUPA0001P" if "HUPA0001P" in available_patients else available_patients[0]
selected_patient_id = st.sidebar.selectbox(
    "Select Patient Record",
    available_patients,
    index=available_patients.index(default_patient) if default_patient in available_patients else 0,
)

# ── Data Fetching: Live Engine vs. Demo Fallback ──────────────────────────────

live_engine = None
live_api = None
current_patient_df = None
data: Dict[str, Any] = {}
execution_mode = "demo"
error_details = None

# Attempt Live Inference
if MODEL_PATH.exists() and (DATA_DIR / f"{selected_patient_id}.csv").exists():
    try:
        current_patient_df = load_patient_preprocessed(selected_patient_id)
        live_engine, live_api = initialize_digital_twin_engine(selected_patient_id)

        if len(current_patient_df) >= 20:
            data = live_api.get_twin_snapshot(current_patient_df, current_idx=-1)
            execution_mode = "live"
        else:
            error_details = f"Patient record has only {len(current_patient_df)} readings (minimum 20 required)."
    except Exception as exc:
        error_details = str(exc)
        logger.exception("Error executing live digital twin inference.")

# Fallback to Demo Payload if live inference not possible
if execution_mode != "live":
    try:
        data = load_demo_fallback_data()
        execution_mode = "demo"
    except Exception as exc:
        st.error(f"Critical Error: Unable to load demonstration data: {exc}")
        st.stop()

# ── Extract Key Variables ─────────────────────────────────────────────────────

patient_id = data.get("patient_id", selected_patient_id)
timestamp = data.get("timestamp", "N/A")
glucose = safe_float(data.get("current_glucose_mgdl"), 100.0)
velocity = safe_float(data.get("glucose_velocity_15min"), 0.0)
trend = str(data.get("glucose_trend", "STABLE"))
risk_score = safe_float(data.get("risk_score"), 0.0)
alert_level = str(data.get("alert_level", "NORMAL"))
is_nocturnal = bool(data.get("is_nocturnal", False))
iob = safe_float(data.get("active_iob_u"), 0.0)
cob = safe_float(data.get("active_cob_g"), 0.0)

# ── Sidebar Metadata Display ──────────────────────────────────────────────────

st.sidebar.write(f"**Patient ID:** {patient_id}")
st.sidebar.write(f"**Latest Reading:** {timestamp}")
st.sidebar.write(f"**Nocturnal Context:** {'Active (23:00 - 06:00)' if is_nocturnal else 'Daytime'}")
st.sidebar.divider()

st.sidebar.caption("Execution Engine")
if execution_mode == "live":
    st.sidebar.markdown('<span class="status-pill-live">LIVE INFERENCE</span>', unsafe_allow_html=True)
    st.sidebar.write("Backend: GlucoTwinEngine + Production Classifier")
else:
    st.sidebar.markdown('<span class="status-pill-demo">DEMO MODE</span>', unsafe_allow_html=True)
    st.sidebar.write("Backend: Static Demonstration Snapshot")

st.sidebar.caption("Twin State Contract")
st.sidebar.write(f"Version {data.get('twin_state_version', '6')}")

# ── Header & Regulatory Banners ───────────────────────────────────────────────

st.markdown('<div class="main-title">🧬 GlucoTwin</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Personalized early-warning and simulation of nocturnal glucose deterioration</div>',
    unsafe_allow_html=True,
)

st.info(
    "**Research prototype — clinician decision support only.** "
    "All predictions and counterfactual simulations are model-derived computational scenarios "
    "and do not constitute clinical diagnoses or treatment recommendations."
)

if execution_mode == "live":
    st.success(
        f"🟢 **LIVE ENGINE CONNECTED** — Digital Twin actively tracking patient **{patient_id}**. "
        f"Snapshot evaluated at timestamp `{timestamp}` using the trained LightGBM/RandomForest "
        "production model, isotonic probability calibrator, and patient memory."
    )
else:
    if error_details:
        st.error(
            f"⚠️ **Live Backend Unavailable — Displaying Demonstration Snapshot**\n\n"
            f"Reason: `{error_details}`\n\n"
            "The dashboard has safely fallen back to verified static HUPA-UCM demonstration data."
        )
    else:
        st.warning(
            f"⚠️ **DEMO MODE** — Displaying static demonstration snapshot for patient **{patient_id}** "
            f"at timestamp `{timestamp}`. Pre-computed from HUPA-UCM cohort data."
        )

# ── Section 1: Current Patient State ─────────────────────────────────────────

st.subheader("1. Current Patient State")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Current Glucose", f"{glucose:.1f} mg/dL")

with col2:
    st.metric("15-Min Velocity", f"{velocity:+.1f} mg/dL/15min")

with col3:
    st.metric("Risk Score", f"{risk_score:.1f} / 100")

with col4:
    alert_color = {
        "NORMAL": "normal",
        "WATCH": "warning",
        "WARNING": "warning",
        "CRITICAL": "danger",
    }.get(alert_level, "normal")
    st.metric("Alert Status", alert_level)

with col5:
    st.metric("Glucose Trend", trend.replace("_", " ").title())

st.divider()

# ── Section 2: Digital Twin State & Confidence ────────────────────────────────

st.subheader("2. Digital Twin State & Confidence")

d1, d2, d3, d4 = st.columns(4)

with d1:
    st.metric("Active Insulin on Board (IOB)", f"{iob:.2f} U")

with d2:
    st.metric("Active Carbs on Board (COB)", f"{cob:.1f} g")

with d3:
    dev_baseline = safe_float(data.get("glucose_deviation_from_baseline"), None)
    st.metric(
        "Baseline Deviation",
        f"{dev_baseline:+.1f} mg/dL" if dev_baseline is not None else "N/A",
    )

with d4:
    dev_ewm = safe_float(data.get("glucose_deviation_from_ewm"), None)
    st.metric(
        "EWM Deviation",
        f"{dev_ewm:+.1f} mg/dL" if dev_ewm is not None else "N/A",
    )

confidence = data.get("twin_confidence") or {}
conf_level = str(confidence.get("confidence_level", "HIGH"))
conf_score = safe_float(confidence.get("confidence_score"), 1.0)
is_ood = bool(confidence.get("out_of_distribution", False))

c_col1, c_col2, c_col3 = st.columns(3)

with c_col1:
    st.metric("Twin Confidence Level", conf_level)

with c_col2:
    st.metric("Confidence Score", f"{conf_score:.3f}")

with c_col3:
    st.metric("Out of Distribution (OOD)", "YES" if is_ood else "NO")

components = confidence.get("components")
if components:
    cgm_cont = safe_float(components.get("cgm_continuity"), 1.0)
    sig_comp = safe_float(components.get("signal_completeness"), 1.0)
    st.caption(
        f"Confidence breakdown: CGM Continuity: {cgm_cont*100:.0f}% | "
        f"Signal Completeness: {sig_comp*100:.0f}% | "
        f"Feature Range: {safe_float(components.get('feature_range_score'), 1.0)*100:.0f}%"
    )

st.divider()

# ── Section 3: Future Hypoglycemia Risk ───────────────────────────────────────

st.subheader("3. Future Hypoglycemia Risk (Multi-Horizon)")

predictions = data.get("predictions_by_horizon", {})
p_cols = st.columns(4)

for col, horizon in zip(p_cols, ["30min", "60min", "90min", "120min"]):
    p = predictions.get(horizon, {})
    with col:
        if p.get("is_calibrated"):
            prob = safe_float(p.get("calibrated_probability"), 0.0)
            st.metric(f"{horizon} Risk", f"{prob * 100:.1f}%")
            st.caption("CALIBRATED (Isotonic)")
        else:
            prob = safe_float(p.get("raw_model_probability"), 0.0)
            st.metric(f"{horizon} Risk", f"{prob * 100:.1f}%")
            st.caption("DERIVED ESTIMATE (Uncalibrated)")

st.caption(
    "The 60-minute prediction is the primary calibrated model prediction (operating threshold: 0.130). "
    "30, 90, and 120-minute predictions are model-derived estimates scaled from the primary model."
)

st.divider()

# ── Section 4: Explainability & Risk Drivers ──────────────────────────────────

st.subheader("4. Why Is the Risk This Level? (Explainability)")

narrative = data.get("clinical_narrative") or data.get("model_interpretation")
if narrative:
    st.info(f"**Twin Interpretation:** {narrative}")

risk_factors = data.get("top_risk_factors", [])
if risk_factors:
    factor_rows = []
    for factor in risk_factors:
        direction = str(factor.get("direction", ""))
        factor_rows.append({
            "Biomarker / Feature": factor.get("feature", ""),
            "Current Value": factor.get("value", ""),
            "Feature Importance": round(safe_float(factor.get("importance"), 0.0), 4),
            "SHAP Attribution": round(safe_float(factor.get("shap_value"), 0.0), 4),
            "Physiological Effect": direction.replace("_", " ").title(),
        })
    st.dataframe(pd.DataFrame(factor_rows), use_container_width=True, hide_index=True)

st.divider()

# ── Section 5: Historical Episode Matching ────────────────────────────────────

st.subheader("5. Historical Precedent Matching")

episodes = data.get("similar_historical_episodes", [])

if episodes:
    ep_rows = []
    for ep in episodes:
        ep_rows.append({
            "Historical Date / Time": ep.get("timestamp", ""),
            "Pattern Similarity": f"{safe_float(ep.get('similarity_score_pct'), 0.0):.1f}%",
            "Starting Glucose": f"{safe_float(ep.get('historical_glucose'), 0.0):.1f} mg/dL",
            "Subsequent Nadir": f"{safe_float(ep.get('subsequent_nadir_mgdl'), 0.0):.1f} mg/dL",
            "Hypoglycemia Event": "Yes (Nadir < 70)" if ep.get("hypo_occurred") else "No (Safe)",
        })
    st.dataframe(pd.DataFrame(ep_rows), use_container_width=True, hide_index=True)

    for i, ep in enumerate(episodes, 1):
        status = "HYPOGLYCEMIA OCCURRED" if ep.get("hypo_occurred") else "STAYED IN SAFE RANGE"
        st.write(
            f"**Precedent {i}:** Similarity {safe_float(ep.get('similarity_score_pct'), 0.0):.1f}% — `{status}`"
        )
        st.caption(ep.get("summary", ""))
else:
    st.info(
        "No historical precedents matched. Historical matching indexes past episodes "
        "from the patient's own record with a minimum 6-hour temporal buffer to prevent leakage."
    )

st.divider()

# ── Section 6: What-If Digital Twin Simulation ───────────────────────────────

st.subheader("6. What-If Digital Twin Simulation")
st.warning("Computational simulation under altered physiological inputs — not a medical treatment recommendation.")

# Select Simulation Mode: Preset vs Custom Interactive
sim_mode = st.radio(
    "Simulation Mode",
    ["Preset Computational Scenarios", "Custom Interactive Intervention"],
    horizontal=True,
)

from src.simulation.simulator import MetabolicSimulator
simulator = MetabolicSimulator()

sim_display: Dict[str, Any] = {}
sim_title = ""

if sim_mode == "Preset Computational Scenarios":
    preset_simulations = data.get("what_if_simulations", [])
    if preset_simulations:
        scenario_names = [s.get("scenario", f"Scenario {i+1}") for i, s in enumerate(preset_simulations)]
        selected_scenario_idx = st.selectbox(
            "Select a precomputed scenario",
            range(len(scenario_names)),
            format_func=lambda x: scenario_names[x],
        )
        sim_display = preset_simulations[selected_scenario_idx]
        sim_title = scenario_names[selected_scenario_idx]
    else:
        # Generate on the fly if missing from payload
        sim_display = simulator.simulate_trajectory(
            current_glucose=glucose,
            current_velocity=velocity,
            active_iob=iob,
            horizon_minutes=120,
        )
        sim_title = "Baseline Trajectory (No intervention)"
else:
    # Interactive custom intervention controls
    st.write("##### Configure Counterfactual Intervention")
    ic1, ic2, ic3, ic4 = st.columns(4)

    with ic1:
        custom_carbs = st.slider("Carbohydrate Intake (g)", min_value=0, max_value=80, value=15, step=5)
    with ic2:
        custom_carb_type = st.selectbox("Absorption Rate", ["fast", "complex"], index=0)
    with ic3:
        custom_bolus = st.slider("Bolus Insulin (U)", min_value=0.0, max_value=8.0, value=0.0, step=0.5)
    with ic4:
        custom_exercise = st.selectbox("Exercise", ["none", "light", "moderate"], index=0)

    custom_horizon = st.select_slider(
        "Simulation Horizon (minutes)",
        options=[30, 60, 90, 120, 180],
        value=120,
    )

    # Execute simulation in real time
    sim_display = simulator.simulate_trajectory(
        current_glucose=glucose,
        current_velocity=velocity,
        active_iob=iob,
        horizon_minutes=custom_horizon,
        carb_intervention_g=float(custom_carbs),
        carb_type=custom_carb_type,
        bolus_intervention_u=float(custom_bolus),
        exercise_intensity=custom_exercise,
    )
    sim_title = f"Custom Simulation: {custom_carbs}g ({custom_carb_type}), {custom_bolus}U insulin, {custom_exercise} exercise"
    st.caption("⚡ Live Counterfactual Simulation computed via MetabolicSimulator")

# Render Simulation Trajectory & Metrics
if sim_display and "trajectory" in sim_display:
    trajectory = sim_display.get("trajectory", [])
    if trajectory:
        sim_df = pd.DataFrame(trajectory)
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=sim_df["minute"],
            y=sim_df["glucose_baseline"],
            mode="lines+markers",
            name="Baseline Trajectory (Natural Decay)",
            line=dict(color="#64748b", dash="dash"),
        ))

        fig.add_trace(go.Scatter(
            x=sim_df["minute"],
            y=sim_df["glucose_simulated"],
            mode="lines+markers",
            name="Simulated Counterfactual Trajectory",
            line=dict(color="#2563eb", width=3),
        ))

        fig.add_hline(
            y=70,
            line_dash="dot",
            line_color="#dc2626",
            annotation_text="Hypoglycemia Threshold: 70 mg/dL",
            annotation_font_color="#dc2626",
        )

        fig.update_layout(
            xaxis_title="Time Horizon (Minutes)",
            yaxis_title="Glucose (mg/dL)",
            hovermode="x unified",
            height=430,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=30, b=20),
        )

        st.plotly_chart(fig, use_container_width=True)

    # 3 Summary Metrics
    sb1, sb2, sb3 = st.columns(3)

    baseline_data = sim_display.get("baseline", {})
    counterfactual_data = sim_display.get("counterfactual", {})

    with sb1:
        sim_nadir = safe_float(
            counterfactual_data.get("simulated_nadir_mgdl"),
            safe_float(sim_display.get("simulated_nadir_mgdl"), None),
        )
        st.metric(
            "Simulated Nadir",
            f"{sim_nadir:.1f} mg/dL" if sim_nadir is not None else "N/A",
        )

    with sb2:
        threshold_crossed = bool(counterfactual_data.get("simulated_threshold_crossing", False))
        st.metric("Threshold Crossing Predicted", "YES" if threshold_crossed else "NO")

    with sb3:
        delta_nadir = safe_float(counterfactual_data.get("delta_nadir_mgdl"), None)
        if delta_nadir is not None and abs(delta_nadir) > 0.05:
            st.metric("Nadir Change vs. Baseline", f"{delta_nadir:+.1f} mg/dL")
        else:
            st.metric("Nadir Change vs. Baseline", "0.0 mg/dL")
            st.caption("Matches baseline trajectory")

    # Scenario Note
    if sim_display.get("simulated_threshold_crossing_prevented"):
        st.success("✅ Computational finding: Counterfactual scenario prevented threshold crossing below 70 mg/dL.")
    elif sim_display.get("simulated_threshold_crossing_induced"):
        st.error("⚠️ Computational finding: Counterfactual scenario induced a threshold crossing below 70 mg/dL.")

st.divider()

# ── Section 7: Model & Data Quality ───────────────────────────────────────────

st.subheader("7. Model Self-Awareness & Data Quality")

quality = data.get("data_quality") or {}
model_info = data.get("model_self_awareness") or {}

q1, q2, q3, q4 = st.columns(4)

with q1:
    st.metric("CGM Signal Validity", "VALID" if quality.get("cgm_valid") else "INVALID")

with q2:
    st.metric("CGM Gap Detected", f"{safe_float(quality.get('cgm_gap_minutes'), 0.0):.1f} min")

with q3:
    st.metric("Calibration Status", str(model_info.get("calibration_status", "CALIBRATED")))

with q4:
    st.metric("Prediction Horizon", f"{model_info.get('prediction_horizon_min', 60)} min")

# Cohort Validation Benchmarks
lopo_auc = safe_float(model_info.get("lopo_roc_auc"), None)
lopo_sens = safe_float(model_info.get("lopo_sensitivity"), None)
brier_cal = safe_float(model_info.get("test_brier_calibrated"), None)

if lopo_auc is not None:
    q5, q6, q7 = st.columns(3)
    with q5:
        st.metric("LOPO Cross-Val ROC-AUC", f"{lopo_auc:.3f}")
    with q6:
        st.metric("LOPO Sensitivity", f"{lopo_sens:.3f}" if lopo_sens else "N/A")
    with q7:
        st.metric("Test Brier Score", f"{brier_cal:.3f}" if brier_cal else "N/A")

st.caption(
    "All model benchmarks derived from Leave-One-Patient-Out (LOPO) cross-validation "
    "on the HUPA-UCM cohort (24 patients, 11,471 observations). "
    "Generalization to other demographics or sensor hardware requires independent validation."
)
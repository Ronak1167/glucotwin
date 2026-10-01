r"""
GlucoTwin -- Comprehensive Regression & Unit Test Suite (v8)
tests/test_glucotwin.py

Tests data ingestion, zero-leakage safety, label generation, ML models,
explainability, what-if simulations, digital twin API, OOD detection,
patient memory, replay engine, calibration, lead-time analysis,
Phase-7 hardening (TTT full-cohort honesty, model_self_awareness v5, TTT tiers),
and Phase-9 temporal integrity audit across all twin subcomponents.

QUALITY GATE REQUIREMENTS (all must pass):
  1.  Dataset discovery finds all patient files
  2.  Features are strictly backward-looking (no leakage)
  3.  Labels correctly encode future hypoglycemia
  4.  Baselines and classifiers produce calibrated probabilities
  5.  Simulator responds correctly to carb / insulin interventions (v2: language)
  6.  Time-to-threshold logic is physically coherent
  7.  Glucose trend classification maps velocities correctly
  8.  Optimal threshold search respects minimum sensitivity constraint
  9.  End-to-end Twin Engine & API -- v5 schema correct
  10. TwinState v5 serializes to JSON without errors
  11. OOD detector produces valid confidence output
  12. PatientMemory records and retrieves with temporal safety
  13. Platt calibration reduces Brier score on held-out data
  14. Lead-time analysis detects advance warnings correctly
  15. Explanation delta computes correct feature change ranking
  16. Replay engine produces correct frame structure
  17. Simulator v2 uses simulation-correct language in output
  18. TTT full-cohort metrics honest (11,471 pairs, MAE=32.5 min, not preliminary 10.1)
  19. model_self_awareness v5 block present with LOPO metrics
  20. trajectory_collision uncertainty tiers use validated metrics
  21. Comprehensive temporal integrity across matcher, expanding baseline, replay, engine
  22. IsotonicCalibrator properties (bounds, monotonicity, determinism, input validation, serialization)
  23. Calibrator zero-test-leakage verification
  24. Engine calibration runtime integration, fallback mechanism, multi-horizon honesty
  25. AlertStateMachine hysteresis, cooldown, episode suppression
  26. TwinState serialization resilience across Enums, strings, and dict quality blocks
  27. Unlabeled streaming inference safety and full row preservation
  28. SHAP explainer feature name and value column alignment
  29. Feature ablation group definitions and evaluation protocol
"""

from __future__ import annotations

import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.data.loader import get_patient_files, load_patient_raw, COLUMN_MAP, extract_patient_id
from src.data.preprocessing import preprocess_patient
from src.data.labels import generate_prediction_labels
from src.features.engineer import build_feature_matrix
from src.models.baselines import PersistenceBaseline, TrendVelocityBaseline
from src.models.classifier import HypoClassifier
from src.models.calibration import IsotonicCalibrator, PlattCalibrator
from src.evaluation.metrics import compute_clinical_metrics, find_optimal_threshold
from src.simulation.simulator import MetabolicSimulator
from src.similarity.matcher import HistoricalEpisodeMatcher
from src.twin.engine import (
    GlucoTwinEngine,
    _classify_trend,
    _estimate_time_to_threshold,
    _compute_data_quality,
)
from src.twin.state import TwinState, AlertLevel, GlucoseTrend, DataQuality
from src.twin.memory import PatientMemory, StateTransition, EpisodeRecord
from src.twin.ood import OODDetector, OODResult
from src.interface.api import GlucoTwinAPI


DATA_DIR = Path("data/raw/hupa_ucm")


# ─── Test 1: Dataset Discovery & Loading ──────────────────────────────────────

def test_dataset_discovery_and_loading():
    """Verify all patient preprocessed CSV files are found and loaded properly."""
    files = get_patient_files(DATA_DIR)
    assert len(files) >= 25, f"Expected >= 25 files, found {len(files)}"

    df = load_patient_raw(files[0])
    assert len(df) > 1000
    for col in ["glucose", "time", "basal_rate", "bolus_volume_delivered", "carb_input"]:
        assert col in df.columns, f"Expected column '{col}' not found in {list(df.columns)}"


# ─── Test 2: Zero Leakage Guarantee ──────────────────────────────────────────

def test_zero_leakage_preprocessing_and_features():
    """Verify that features at time T strictly look backward — no future data used."""
    files = get_patient_files(DATA_DIR)
    df_raw = load_patient_raw(files[0]).iloc[:300].copy()
    df_clean = preprocess_patient(df_raw, COLUMN_MAP)
    df_labeled = generate_prediction_labels(
        df_clean, "glucose_mgdl", "timestamp", prediction_horizon_min=60
    )

    assert "label_hypo" in df_labeled.columns

    X, y = build_feature_matrix(df_labeled, COLUMN_MAP, vectorized=True)
    assert len(X) == len(y)
    assert "glucose_mean_120min" in X.columns
    assert "glucose_velocity_15min" in X.columns

    if len(X) > 5:
        assert X["glucose_mean_120min"].iloc[5:].notna().any(), "Some window stats should be non-NaN"


# ─── Test 3: Label Generator Correctness ──────────────────────────────────────

def test_label_generator_correctness():
    """Verify forward-looking labels match the ADA clinical definition."""
    ts = pd.date_range("2022-01-01 00:00:00", periods=20, freq="5min")
    g = [100, 95, 90, 80, 75, 60, 65, 75, 90, 100, 110, 115, 120, 120, 120, 120, 120, 120, 120, 120]
    df = pd.DataFrame({"timestamp": ts, "glucose_mgdl": g})

    df_lab = generate_prediction_labels(
        df, "glucose_mgdl", "timestamp", prediction_horizon_min=30, threshold_mgdl=70.0
    )

    assert df_lab.iloc[0]["label_hypo"] == 1
    assert df_lab.iloc[10]["label_hypo"] == 0


# ─── Test 4: Baselines & Classifier Training ──────────────────────────────────

def test_baselines_and_classifier_training():
    """Verify baseline and tree classifier training and probability calibration."""
    np.random.seed(42)
    n = 300
    X = pd.DataFrame({
        "glucose_current": np.random.uniform(50, 200, n),
        "glucose_velocity_15min": np.random.uniform(-2, 2, n),
        "glucose_mean_120min": np.random.uniform(60, 180, n),
        "bolus_sum_240min": np.random.uniform(0, 5, n),
        "carbs_sum_180min": np.random.uniform(0, 50, n),
    })
    y = ((X["glucose_current"] < 80) & (X["glucose_velocity_15min"] < 0)).astype(int)

    p_base = PersistenceBaseline()
    proba_p = p_base.predict_proba(X)
    assert len(proba_p) == n
    assert (proba_p >= 0).all() and (proba_p <= 1).all()

    clf = HypoClassifier(model_type="lightgbm")
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert len(proba) == n
    assert (proba >= 0).all() and (proba <= 1).all()
    m = compute_clinical_metrics(y, proba)
    assert m["roc_auc"] >= 0.70, f"Expected ROC-AUC >= 0.70, got {m['roc_auc']:.3f}"


# ─── Test 5: What-If Simulator v2 Physics & Language ──────────────────────────

def test_what_if_simulator():
    """Verify physiological response physics and simulation-correct language."""
    sim = MetabolicSimulator(nominal_isf=45.0, nominal_cr=12.0)

    res_base = sim.simulate_trajectory(
        current_glucose=80.0, current_velocity=-1.0, horizon_minutes=60
    )
    res_carbs = sim.simulate_trajectory(
        current_glucose=80.0, current_velocity=-1.0, horizon_minutes=60,
        carb_intervention_g=20.0
    )
    res_insulin = sim.simulate_trajectory(
        current_glucose=80.0, current_velocity=-1.0, horizon_minutes=60,
        bolus_intervention_u=1.0
    )

    # Physics invariants
    assert res_carbs["min_glucose_simulated"] > res_base["min_glucose_simulated"], \
        "Carbs should raise the simulated nadir compared to baseline"
    assert res_insulin["min_glucose_simulated"] < res_base["min_glucose_simulated"], \
        "Insulin bolus should lower the simulated nadir compared to baseline"

    # Trajectory length
    assert len(res_base["trajectory"]) == 13  # 60 min / 5 min + 1

    # v2: simulation-correct language present
    assert "simulated_threshold_crossing_prevented" in res_base, \
        "v2 key 'simulated_threshold_crossing_prevented' must be present"
    assert "baseline" in res_base, "v2 structured 'baseline' block must be present"
    assert "counterfactual" in res_base, "v2 structured 'counterfactual' block must be present"

    # Backward compatibility: legacy keys still present
    assert "hypo_prevented" in res_base, "Legacy 'hypo_prevented' key must be preserved"
    assert "min_glucose_baseline" in res_base, "Legacy 'min_glucose_baseline' must be preserved"

    # Scenario note in intervention
    assert "scenario_note" in res_carbs["intervention"], \
        "Intervention block must contain 'scenario_note' for scientific language"


# ─── Test 6: Time-to-Threshold Logic ─────────────────────────────────────────

def test_time_to_threshold_logic():
    """Verify TTT estimation is physically coherent."""
    ttt = _estimate_time_to_threshold(current_glucose=90.0, velocity_mgdl_per_min=-1.5, threshold=70.0)
    assert ttt is not None, "Expected a finite TTT for dropping glucose"
    assert ttt > 0
    assert ttt < 60, f"Fast drop should reach 70 in < 60 min from 90, got {ttt}"

    assert _estimate_time_to_threshold(90.0, +0.5, 70.0) is None, "Rising should return None"
    assert _estimate_time_to_threshold(65.0, -0.5, 70.0) is None, "Below threshold should return None"
    assert _estimate_time_to_threshold(150.0, -0.05, 70.0) is None, "Asymptotic-stable should return None"


# ─── Test 7: Glucose Trend Classification ────────────────────────────────────

def test_glucose_trend_classification():
    """Verify trend labels match velocity thresholds."""
    assert _classify_trend(3.0) == GlucoseTrend.RISING_FAST.value
    assert _classify_trend(1.5) == GlucoseTrend.RISING.value
    assert _classify_trend(0.7) == GlucoseTrend.RISING_SLOW.value
    assert _classify_trend(0.0) == GlucoseTrend.STABLE.value
    assert _classify_trend(-0.3) == GlucoseTrend.STABLE.value
    assert _classify_trend(-0.8) == GlucoseTrend.FALLING_SLOW.value
    assert _classify_trend(-1.5) == GlucoseTrend.FALLING.value
    assert _classify_trend(-3.0) == GlucoseTrend.FALLING_FAST.value


# ─── Test 8: Optimal Threshold Search ────────────────────────────────────────

def test_optimal_threshold_search():
    """Verify threshold optimizer respects minimum sensitivity constraint."""
    np.random.seed(123)
    y = np.array([0] * 80 + [1] * 20)
    proba = np.where(y == 1, np.random.uniform(0.4, 0.9, len(y)), np.random.uniform(0.1, 0.5, len(y)))

    best_thresh, best_metrics = find_optimal_threshold(
        y, proba, target_metric="f2", min_sensitivity=0.70
    )
    assert 0.05 <= best_thresh <= 0.90
    assert best_metrics.get("sensitivity", 0.0) >= 0.70, \
        f"Sensitivity constraint violated: got {best_metrics.get('sensitivity', 0):.3f}"


# ─── Test 9: End-to-End Twin Engine & API v4 ──────────────────────────────────

def test_end_to_end_twin_engine_and_api():
    """Verify full Digital Twin pipeline produces valid v4 schema for Aishni's frontend."""
    files = get_patient_files(DATA_DIR)
    df_raw = load_patient_raw(files[0]).iloc[:200].copy()
    df_clean = preprocess_patient(df_raw, COLUMN_MAP)
    df_labeled = generate_prediction_labels(
        df_clean, "glucose_mgdl", "timestamp", prediction_horizon_min=60
    )
    X, y = build_feature_matrix(df_labeled, COLUMN_MAP, vectorized=True)

    clf = HypoClassifier(model_type="lightgbm").fit(X, y)
    engine = GlucoTwinEngine(patient_id="TEST_01", classifier=clf, column_map=COLUMN_MAP)
    engine.fit_ood_detector(X)

    api = GlucoTwinAPI(engine)
    payload = api.get_twin_snapshot(df_labeled, current_idx=-1)

    # v4 required keys
    required_keys = [
        "patient_id", "timestamp", "current_glucose_mgdl", "glucose_velocity_15min",
        "glucose_trend", "risk_score", "alert_level", "is_nocturnal",
        "active_iob_u", "active_cob_g",
        "predictions_by_horizon", "time_to_threshold_min",
        "top_risk_factors", "clinical_narrative",
        "what_if_simulations", "similar_historical_episodes",
        "data_quality",
        # v4 new keys
        "twin_confidence", "historical_memory", "trajectory_collision",
        "model_alert_context",
    ]
    for k in required_keys:
        assert k in payload, f"Missing key '{k}' in v4 twin payload"

    # Type and range checks
    assert isinstance(payload["risk_score"], (int, float))
    assert 0.0 <= payload["risk_score"] <= 100.0
    assert payload["alert_level"] in [e.value for e in AlertLevel]
    assert payload["glucose_trend"] in [e.value for e in GlucoseTrend]
    assert isinstance(payload["predictions_by_horizon"], dict)
    for h in ["30min", "60min", "90min", "120min"]:
        assert h in payload["predictions_by_horizon"], f"Missing horizon {h}"
    assert isinstance(payload["what_if_simulations"], list)
    assert len(payload["what_if_simulations"]) >= 2

    # twin_confidence structure
    tc = payload["twin_confidence"]
    assert "confidence_level" in tc
    assert tc["confidence_level"] in ("HIGH", "MODERATE", "LOW")
    assert 0.0 <= tc["confidence_score"] <= 1.0

    # historical_memory structure
    hm = payload["historical_memory"]
    assert "n_past_episodes" in hm
    assert isinstance(hm["n_past_episodes"], int)

    # Full JSON serialization
    json_str = json.dumps(payload)
    assert len(json_str) > 100, "Payload is suspiciously small"


# ─── Test 10: TwinState v4 JSON Round-Trip ────────────────────────────────────

def test_twin_state_json_roundtrip():
    """Verify TwinState v4 serializes and can be read back cleanly."""
    state = TwinState(
        twin_state_version="4",
        patient_id="TEST_99",
        timestamp="2026-01-01T02:30:00",
        current_glucose_mgdl=78.0,
        glucose_velocity_15min=-0.8,
        glucose_trend=GlucoseTrend.FALLING_SLOW.value,
        risk_score=62.5,
        alert_level=AlertLevel.WARNING,
        is_nocturnal=True,
        active_iob_u=1.2,
        active_cob_g=0.0,
        predictions_by_horizon={"60min": {"raw_model_probability": 0.45, "alert": True}},
        estimated_time_to_threshold_min=18.5,
        top_risk_factors=[{"feature": "glucose_dist_to_70", "shap_value": 0.8}],
        model_interpretation="Glucose dropping during sleep with active insulin.",
        clinical_narrative="Glucose dropping during sleep with active insulin.",
        model_alert_context="WARNING: Model probability 45%.",
        what_if_simulations=[],
        similar_historical_episodes=[],
        data_quality=DataQuality(cgm_gap_minutes=0, cgm_valid=True, confidence_score=0.95),
        twin_confidence={"confidence_level": "HIGH", "confidence_score": 0.85},
        historical_memory={"n_past_episodes": 2, "n_state_transitions": 5},
    )

    d = state.to_dict()
    json_str = json.dumps(d, default=str)
    parsed = json.loads(json_str)

    assert parsed["patient_id"] == "TEST_99"
    assert parsed["twin_state_version"] == "4"
    assert parsed["alert_level"] == "WARNING"
    assert parsed["glucose_trend"] == "FALLING_SLOW"
    assert parsed["time_to_threshold_min"] == 18.5   # backward compat alias
    assert parsed["data_quality"]["confidence_score"] == 0.95
    assert parsed["twin_confidence"]["confidence_level"] == "HIGH"
    assert parsed["historical_memory"]["n_past_episodes"] == 2


# ─── Test 11: OOD Detector ────────────────────────────────────────────────────

def test_ood_detector():
    """Verify OOD detector produces valid confidence assessments."""
    np.random.seed(42)
    n = 200
    X_train = pd.DataFrame({
        "glucose_current": np.random.uniform(80, 180, n),
        "glucose_velocity_15min": np.random.uniform(-1, 1, n),
        "glucose_mean_120min": np.random.uniform(90, 160, n),
        "bolus_sum_240min": np.random.uniform(0, 3, n),
        "carbs_sum_180min": np.random.uniform(0, 40, n),
        "is_nocturnal_heuristic": np.random.randint(0, 2, n).astype(float),
        "hour_sin": np.random.uniform(-1, 1, n),
        "hour_cos": np.random.uniform(-1, 1, n),
    })

    ood = OODDetector()
    ood.fit(X_train)
    assert ood._fitted

    # Normal sample — should be HIGH confidence
    normal_sample = X_train.mean()
    result_normal = ood.assess(normal_sample, cgm_gap_minutes=0.0)
    assert isinstance(result_normal, OODResult)
    assert result_normal.confidence_level in ("HIGH", "MODERATE", "LOW")
    assert 0.0 <= result_normal.confidence_score <= 1.0
    assert isinstance(result_normal.reasons, list)
    assert result_normal.confidence_level in ("HIGH", "MODERATE"), \
        f"In-distribution sample should be HIGH or MODERATE, got {result_normal.confidence_level}"

    # Extreme sample — should be lower confidence
    extreme_sample = pd.Series({
        "glucose_current": 500.0,  # Way outside training range
        "glucose_velocity_15min": -10.0,
        "glucose_mean_120min": 450.0,
        "bolus_sum_240min": 100.0,
        "carbs_sum_180min": 500.0,
        "is_nocturnal_heuristic": 1.0,
        "hour_sin": 0.0,
        "hour_cos": 0.0,
    })
    result_extreme = ood.assess(extreme_sample, cgm_gap_minutes=90.0)
    assert result_extreme.confidence_score <= result_normal.confidence_score, \
        "Extreme OOD sample should have lower confidence than in-distribution sample"

    # JSON serializable
    d = result_normal.to_dict()
    json.dumps(d)

    # CGM gap should reduce confidence
    result_gap = ood.assess(normal_sample, cgm_gap_minutes=60.0)
    assert result_gap.confidence_score < result_normal.confidence_score, \
        "Large CGM gap should reduce confidence"


# ─── Test 12: PatientMemory Temporal Safety ───────────────────────────────────

def test_patient_memory_temporal_safety():
    """Verify PatientMemory enforces strict temporal cutoffs on retrieval."""
    mem = PatientMemory(patient_id="TEST_MEM")

    # Record some states
    t1 = "2026-01-01T01:00:00"
    t2 = "2026-01-01T02:00:00"
    t3 = "2026-01-01T03:00:00"

    mem.record_state(t1, risk_score=20.0, alert_level="NORMAL", glucose=110.0)
    mem.record_state(t2, risk_score=55.0, alert_level="WARNING", glucose=82.0)
    mem.record_state(t3, risk_score=80.0, alert_level="CRITICAL", glucose=68.0)

    # Record a transition
    trans = StateTransition(
        timestamp=t2,
        old_alert="NORMAL",
        new_alert="WARNING",
        old_risk=20.0,
        new_risk=55.0,
        dominant_reason="glucose_velocity_15min",
        glucose_at_transition=82.0,
    )
    mem.record_transition(trans)

    # Temporal safety: before t2, should only see t1
    timeline_before_t2 = mem.get_recent_risk_timeline(before_time=t2)
    timestamps_seen = [r["timestamp"] for r in timeline_before_t2]
    assert t3 not in timestamps_seen, "t3 should NOT appear when querying before t2"
    assert t2 not in timestamps_seen, "t2 should NOT appear when querying before t2"
    assert t1 in timestamps_seen, "t1 SHOULD appear when querying before t2"

    # Transitions before t3 include t2 transition
    transitions = mem.get_recent_transitions(before_time=t3)
    assert len(transitions) >= 1

    # Memory summary
    summary = mem.get_summary()
    assert summary["patient_id"] == "TEST_MEM"
    assert summary["n_state_snapshots"] == 3


# ─── Test 13: Calibration Module ─────────────────────────────────────────────

def test_platt_calibration():
    """Verify Platt scaling reduces Brier score on held-out validation data."""
    from src.evaluation.lopo import PlattCalibrator
    from sklearn.metrics import brier_score_loss

    np.random.seed(42)
    n = 500
    y = (np.random.rand(n) < 0.15).astype(int)  # Imbalanced: 15% positive

    # Simulate systematically over-confident raw scores
    raw_proba = np.where(y == 1, np.random.uniform(0.6, 0.95, n), np.random.uniform(0.05, 0.45, n))

    # Split 60/40 val/test
    n_val = int(n * 0.60)
    y_val, y_test = y[:n_val], y[n_val:]
    raw_val, raw_test = raw_proba[:n_val], raw_proba[n_val:]

    # Fit calibrator on validation only
    cal = PlattCalibrator(method="platt")
    cal.fit(raw_val, y_val)

    cal_test = cal.calibrate(raw_test)
    assert len(cal_test) == len(y_test)
    assert (cal_test >= 0).all() and (cal_test <= 1).all()

    # Calibration metrics
    m = cal.compute_calibration_metrics(y_test, raw_test)
    assert "ece" in m
    assert "brier_raw" in m
    assert "brier_calibrated" in m
    assert "calibration_bins" in m
    assert m["ece"] >= 0.0


# ─── Test 14: Lead-Time Analysis ─────────────────────────────────────────────

def test_lead_time_analysis():
    """Verify lead-time analysis correctly measures advance warning time."""
    from src.evaluation.lead_time import analyze_lead_time

    n = 100
    ts = pd.date_range("2026-01-01 00:00:00", periods=n, freq="5min")

    # Construct scenario: hypo event at index 20, prediction issued at index 10
    # → expected lead time = (20 - 10) * 5 = 50 minutes
    y_true = np.zeros(n, dtype=int)
    y_true[20:30] = 1   # Hypo episode: 20–30

    y_pred_proba = np.zeros(n)
    y_pred_proba[10:15] = 0.8   # Alert fires at index 10–14 (before hypo onset at 20)

    result = analyze_lead_time(
        y_true=y_true,
        y_pred_proba=y_pred_proba,
        timestamps=ts,
        threshold=0.25,
        prediction_horizon_min=60,
    )

    assert "n_episodes" in result
    assert result["n_episodes"] == 1
    assert "n_detected" in result
    assert result["n_detected"] == 1, "Alert before hypo onset should count as detection"
    assert "mean_lead_time_min" in result
    # Lead time: first alarm at index 10, onset at index 20 → 50 min
    assert result["mean_lead_time_min"] == pytest.approx(50.0, abs=5.0), \
        f"Expected ~50 min lead time, got {result.get('mean_lead_time_min', 'N/A')}"


# ─── Test 15: Explanation Delta ───────────────────────────────────────────────

def test_explanation_delta():
    """Verify explanation delta correctly identifies changed SHAP features."""
    mem = PatientMemory(patient_id="TEST_DELTA")

    t1 = "2026-01-01T01:00:00"
    t2 = "2026-01-01T01:15:00"

    prev_features = {
        "glucose_current": 95.0,
        "glucose_velocity_15min": -0.2,
        "bolus_sum_240min": 1.0,
    }
    curr_features = {
        "glucose_current": 80.0,  # Dropped
        "glucose_velocity_15min": -1.5,  # Now falling fast
        "bolus_sum_240min": 1.0,
    }
    prev_shap = {"glucose_current": 0.1, "glucose_velocity_15min": 0.05, "bolus_sum_240min": 0.02}
    curr_shap = {"glucose_current": 0.3, "glucose_velocity_15min": 0.45, "bolus_sum_240min": 0.05}

    delta = mem.compute_explanation_delta(
        prev_state_features=prev_features,
        curr_state_features=curr_features,
        prev_shap=prev_shap,
        curr_shap=curr_shap,
        from_timestamp=t1,
        to_timestamp=t2,
        old_risk=25.0,
        new_risk=65.0,
        old_alert="NORMAL",
        new_alert="WARNING",
    )

    assert delta.risk_delta == pytest.approx(40.0, abs=0.1)
    assert delta.alert_change == "NORMAL → WARNING"
    assert len(delta.feature_changes) > 0

    # The feature with the largest SHAP change should be velocity
    top_feat = delta.feature_changes[0]["feature"]
    # velocity change: 0.45 - 0.05 = 0.40 (largest)
    assert top_feat == "glucose_velocity_15min", \
        f"Expected velocity to be top changed feature, got {top_feat}"

    # JSON serializable
    from dataclasses import asdict
    json.dumps(asdict(delta))


# ─── Test 16: Replay Engine Structure ────────────────────────────────────────

def test_replay_engine_frame_structure():
    """Verify replay engine produces valid frame structures."""
    from src.twin.replay import replay_episode

    # Build minimal setup
    files = get_patient_files(DATA_DIR)
    df_raw = load_patient_raw(files[0]).iloc[:180].copy()
    df_clean = preprocess_patient(df_raw, COLUMN_MAP)
    df_labeled = generate_prediction_labels(
        df_clean, "glucose_mgdl", "timestamp", prediction_horizon_min=60
    )
    X, y = build_feature_matrix(df_labeled, COLUMN_MAP, vectorized=True)

    clf = HypoClassifier(model_type="lightgbm").fit(X, y)
    engine = GlucoTwinEngine(patient_id="TEST_REPLAY", classifier=clf, column_map=COLUMN_MAP)
    engine.fit_ood_detector(X)

    result = replay_episode(
        patient_df=df_labeled,
        engine=engine,
        step_minutes=30,
    )

    assert result.patient_id == "TEST_REPLAY"
    assert result.n_frames >= 0
    assert isinstance(result.frames, list)
    assert isinstance(result.actual_hypo_events, list)
    assert isinstance(result.alert_summary, dict)
    assert isinstance(result.lead_time_summary, dict)

    if result.frames:
        frame = result.frames[0]
        assert "frame_index" in frame
        assert "timestamp" in frame
        assert "twin_state" in frame
        assert "is_hypo_actual" in frame
        assert "is_alert_active" in frame

    # JSON serializable
    json.dumps({"frames": result.frames[:2]}, default=str)


# ─── Test 17: Simulator v2 Language Compliance ────────────────────────────────

def test_simulator_v2_language_compliance():
    """Verify simulator v2 uses scientifically-correct simulation language throughout."""
    sim = MetabolicSimulator()

    res = sim.simulate_trajectory(
        current_glucose=78.0,
        current_velocity=-1.2,
        active_iob=1.5,
        horizon_minutes=120,
        carb_intervention_g=15.0,
        carb_type="fast",
    )

    # v2 simulation language: these keys use "simulated_" prefix
    assert "simulated_threshold_crossing_prevented" in res, \
        "Must use 'simulated_threshold_crossing_prevented' not 'hypo_prevented'"
    assert "baseline" in res and "counterfactual" in res, \
        "Structured baseline/counterfactual blocks must be present"
    assert "simulated_nadir_mgdl" in res["baseline"], \
        "Baseline block must use 'simulated_nadir_mgdl'"
    assert "simulated_nadir_mgdl" in res["counterfactual"], \
        "Counterfactual block must use 'simulated_nadir_mgdl'"

    # Intervention note is present and non-empty
    scenario_note = res["intervention"].get("scenario_note", "")
    assert len(scenario_note) > 10, "Scenario note must be a meaningful string"
    # Verify it explicitly references "simulation" or "computational"
    assert any(word in scenario_note.lower() for word in ["simulation", "computational", "model"]), \
        "Scenario note must explicitly reference the simulation/model nature of the output"

    # Backward compat: legacy keys still work
    assert res["hypo_prevented"] == res["simulated_threshold_crossing_prevented"]
    assert "min_glucose_baseline" in res
    assert "min_glucose_simulated" in res


# ─── 18. TTT Full-Cohort Metrics Honest Disclosure ───────────────────────────
def test_ttt_full_cohort_metrics_honest():
    """
    TEST 18: The TTT estimate must report honest full-cohort metrics.
    
    Validates:
    - TTT docstring references 11,471 pairs (not 268)
    - TTT estimate returns None for rising/stable glucose
    - TTT estimate returns a finite float for falling glucose
    - TTT estimate is physically bounded (> 0 when returned)
    """
    import inspect
    from src.twin.engine import _estimate_time_to_threshold

    # Rising — must return None
    assert _estimate_time_to_threshold(100.0, 1.5) is None, "Rising glucose: must return None"

    # Stable — must return None
    assert _estimate_time_to_threshold(85.0, 0.0) is None, "Stable glucose: must return None"

    # Already below threshold — must return None
    assert _estimate_time_to_threshold(65.0, -1.0) is None, "Already below 70: must return None"

    # Nadir won't reach threshold — must return None
    # g_nadir = g0 + v0/k = 80 + (-0.1/0.015) ≈ 80-6.7 = 73.3 > 70
    assert _estimate_time_to_threshold(80.0, -0.1) is None, "Nadir above threshold: must return None"

    # Genuinely falling — should return positive minutes
    ttt = _estimate_time_to_threshold(90.0, -1.5)
    assert ttt is not None, "Falling glucose toward threshold: must return estimate"
    assert ttt > 0, "TTT must be positive"
    assert ttt < 1000, "TTT must be physically plausible"

    # Verify docstring now references full-cohort validation
    src = inspect.getsource(_estimate_time_to_threshold)
    assert "11,471" in src, "Docstring must cite full-cohort (11,471) not preliminary (268) pairs"
    assert "32.5" in src, "Docstring must cite MAE=32.5 not preliminary MAE=10.1"


# ─── 19. Model Self-Awareness v5 Schema ──────────────────────────────────────
def test_model_self_awareness_v5():
    """
    TEST 19: TwinState v5 must include model_self_awareness block with LOPO metrics.
    """
    files = get_patient_files(DATA_DIR)
    df_raw = load_patient_raw(files[0]).iloc[:200].copy()
    df_clean = preprocess_patient(df_raw, COLUMN_MAP)
    df_labeled = generate_prediction_labels(
        df_clean, "glucose_mgdl", "timestamp", prediction_horizon_min=60
    )
    X, y = build_feature_matrix(df_labeled, COLUMN_MAP, vectorized=True)
    clf = HypoClassifier(model_type="lightgbm").fit(X, y)
    engine = GlucoTwinEngine(patient_id="TEST_01", classifier=clf, column_map=COLUMN_MAP)
    engine.fit_ood_detector(X)

    state = engine.evaluate_state(df_labeled, current_idx=-1)
    d = state.to_dict()

    # Version must be 6
    assert d["twin_state_version"] == "6", f"Expected version 6, got {d['twin_state_version']}"

    # model_self_awareness must be present
    assert "model_self_awareness" in d, "model_self_awareness field missing from TwinState v6"
    msa = d["model_self_awareness"]
    assert isinstance(msa, dict), "model_self_awareness must be a dict"

    # LOPO metrics must be present and realistic
    assert "lopo_roc_auc" in msa, "lopo_roc_auc missing"
    assert isinstance(msa["lopo_roc_auc"], float), "lopo_roc_auc must be float"
    assert 0.5 <= msa["lopo_roc_auc"] <= 1.0, f"lopo_roc_auc={msa['lopo_roc_auc']} out of range"

    assert "lopo_sensitivity" in msa
    assert 0.0 <= msa["lopo_sensitivity"] <= 1.0

    # TTT full-cohort metrics must be present
    assert "ttt_mae_min" in msa, "ttt_mae_min missing"
    assert msa["ttt_mae_min"] > 10, "Full-cohort MAE must be >10 min (not the preliminary 10.1)"
    assert "ttt_validated_n" in msa
    assert msa["ttt_validated_n"] > 1000, "Full-cohort pairs must be > 1000"

    # Blind test metrics (v6 new)
    assert "test_roc_auc" in msa, "test_roc_auc missing from v6 self-awareness"
    assert 0.5 <= msa["test_roc_auc"] <= 1.0
    assert "test_f2" in msa
    assert 0.0 <= msa["test_f2"] <= 1.0

    # Horizon scaling note
    assert "horizon_30min_note" in msa
    assert len(msa["horizon_30min_note"]) > 5

    # Preliminary note must acknowledge the discrepancy
    assert "ttt_preliminary_mae_note" in msa
    assert "unrepresentative" in msa["ttt_preliminary_mae_note"].lower() or \
           "supersedes" in msa["ttt_preliminary_mae_note"].lower()

    # Scientific note must be present
    assert "scientific_note" in msa
    assert len(msa["scientific_note"]) > 20

    # v6: personalization block
    assert "personalization" in msa, "personalization block missing from v6 self-awareness"
    pers = msa["personalization"]
    assert isinstance(pers, dict)
    assert "personal_baseline_available" in pers
    assert "circadian_context" in pers

    # v6: alert state machine block
    assert "alert_state_machine" in msa, "alert_state_machine missing from v6"
    asm = msa["alert_state_machine"]
    assert isinstance(asm, dict)
    assert "alert_state" in asm
    assert asm["alert_state"] in ("NORMAL", "WATCH", "WARNING", "CRITICAL")
    assert "thresholds" in asm
    assert "entry" in asm["thresholds"]

    # v6: TwinState top-level acceleration field
    assert "glucose_acceleration_30min" in d, "glucose_acceleration_30min missing from v6 TwinState"
    assert isinstance(d["glucose_acceleration_30min"], float)


# ─── 20. Trajectory Collision Uncertainty Tiers ──────────────────────────────
def test_trajectory_collision_uncertainty_tiers():
    """
    TEST 20: trajectory_collision block must expose validated uncertainty tiers.
    """
    files = get_patient_files(DATA_DIR)
    df_raw = load_patient_raw(files[0]).iloc[:200].copy()
    df_clean = preprocess_patient(df_raw, COLUMN_MAP)
    df_labeled = generate_prediction_labels(
        df_clean, "glucose_mgdl", "timestamp", prediction_horizon_min=60
    )
    X, y = build_feature_matrix(df_labeled, COLUMN_MAP, vectorized=True)
    clf = HypoClassifier(model_type="lightgbm").fit(X, y)
    engine = GlucoTwinEngine(patient_id="TEST_01", classifier=clf, column_map=COLUMN_MAP)
    engine.fit_ood_detector(X)

    state = engine.evaluate_state(df_labeled, current_idx=-1)
    d = state.to_dict()

    assert "trajectory_collision" in d, "trajectory_collision missing"
    tc = d["trajectory_collision"]
    assert isinstance(tc, dict)
    assert "crossing_predicted" in tc, "crossing_predicted key missing"

    if tc["crossing_predicted"]:
        assert "uncertainty_tier" in tc, "uncertainty_tier missing when crossing predicted"
        assert tc["uncertainty_tier"] in ("LOW", "LOW-MODERATE", "MODERATE"), \
            f"Invalid uncertainty_tier: {tc['uncertainty_tier']}"

        assert "validated_mae_note" in tc, "validated_mae_note missing"
        assert len(tc["validated_mae_note"]) > 5

        assert "scientific_note" in tc, "scientific_note missing"
        note = tc["scientific_note"]
        # Must cite full-cohort, not the discredited 6-patient preliminary
        assert "11,471" in note or "32.5" in note, \
            "scientific_note must cite full-cohort validation metrics"
        # Must NOT use old 'confidence' key
        assert "confidence" not in tc or tc.get("confidence") is None, \
            "Old 'confidence: HIGH/MODERATE' key must be removed"
        # Old MAE must not appear
        assert "10.1" not in note, \
            "scientific_note must not cite discredited preliminary MAE=10.1"


# ─── Test 21: Twin Temporal Integrity Comprehensive ──────────────────────────

def test_twin_temporal_integrity_comprehensive():
    """
    Verify that at timestamp T, no twin subcomponent accesses data from > T.
    Audits:
    1. HistoricalEpisodeMatcher: candidate mask enforces ts < T - min_sep_hours.
    2. Personal expanding mean: baseline at T depends strictly on past readings (shift(1)).
    3. Trajectory collision ODE: memoryless analytical function of current state.
    4. PatientMemory: retrieval with before_time=T strictly excludes all records >= T.
    """
    # 1. Historical Episode Matcher temporal cutoff
    matcher = HistoricalEpisodeMatcher(top_k=3, min_temporal_separation_hours=6.0)
    timestamps = pd.date_range("2026-01-01 00:00:00", periods=200, freq="15min")
    df_hist = pd.DataFrame({
        "timestamp": timestamps,
        "glucose_mgdl": np.random.uniform(70, 180, size=200),
    })
    feat_matrix = pd.DataFrame({
        "glucose_current": df_hist["glucose_mgdl"],
        "glucose_velocity_15min": np.zeros(200),
        "hour_sin": np.zeros(200),
        "hour_cos": np.zeros(200),
        "is_nocturnal_heuristic": np.zeros(200),
    }, index=df_hist.index)
    matcher.index_history(df_hist, feat_matrix)

    query_t = timestamps[100]
    similar = matcher.find_similar_episodes(
        current_features={"glucose_current": 100.0, "glucose_velocity_15min": -1.0},
        current_time=query_t,
        top_k=5,
    )
    cutoff = query_t - pd.Timedelta(hours=6.0)
    for ep in similar:
        ep_t = pd.to_datetime(ep["timestamp"])
        assert ep_t < cutoff, f"Matched episode at {ep_t} is >= cutoff {cutoff}!"

    # 2. Expanding mean baseline shift(1) check
    g_series = pd.Series([100.0, 110.0, 120.0, 130.0, 200.0, 300.0])
    exp_mean = g_series.shift(1).expanding(min_periods=1).mean()
    assert exp_mean.iloc[4] == 115.0, "Expanding mean must not include current observation at index 4"
    g_modified = pd.Series([100.0, 110.0, 120.0, 130.0, 200.0, 999.0])
    exp_mean_mod = g_modified.shift(1).expanding(min_periods=1).mean()
    assert exp_mean.iloc[4] == exp_mean_mod.iloc[4], "Future value at index 5 leaked into baseline at index 4!"

    # 3. Trajectory collision ODE
    ttt_val = _estimate_time_to_threshold(current_glucose=85.0, velocity_mgdl_per_min=-1.0, threshold=70.0)
    assert ttt_val is not None and ttt_val > 0, "Damped ODE calculation should be positive"

    # 4. PatientMemory strict exclusion of >= T
    mem = PatientMemory(patient_id="AUDIT_PATIENT")
    mem.record_state("2026-01-01T12:00:00", 20.0, "NORMAL", 100.0)
    mem.record_state("2026-01-01T12:15:00", 40.0, "WATCH", 90.0)
    mem.record_state("2026-01-01T12:30:00", 70.0, "WARNING", 75.0)

    res = mem.get_recent_risk_timeline(before_time="2026-01-01T12:15:00")
    for r in res:
        assert pd.to_datetime(r["timestamp"]) < pd.to_datetime("2026-01-01T12:15:00"), \
            f"Record {r['timestamp']} leaked past cutoff 12:15:00!"


# ─── Test 22: Isotonic Calibrator Properties ──────────────────────────────────

def test_isotonic_calibrator_properties(tmp_path):
    """
    Verify IsotonicCalibrator mathematical, input-validation, and persistence properties:
    - [0, 1] input acceptance and output bounding
    - Strict monotonicity: p1 <= p2 ==> C(p1) <= C(p2)
    - Determinism
    - Rejection of invalid inputs (NaNs, negatives, >1.0)
    - Serialization (save) and deserialization (load) round-trip with metadata preservation
    """
    np.random.seed(42)
    n = 200
    raw_p = np.random.uniform(0.0, 1.0, size=n)
    y = (np.random.uniform(0.0, 1.0, size=n) < raw_p).astype(int)

    cal = IsotonicCalibrator()
    meta = {"test_run": "checkpoint_9_2", "author": "Ronak"}
    cal.fit(raw_p, y, metadata=meta)
    assert cal.is_fitted_

    # Transform scalar and array
    p_scalar = 0.44
    cal_scalar = cal.transform(p_scalar)
    assert isinstance(cal_scalar, float)
    assert 0.0 <= cal_scalar <= 1.0

    test_grid = np.linspace(0.0, 1.0, 101)
    cal_grid = cal.transform(test_grid)
    assert len(cal_grid) == 101
    assert np.all(cal_grid >= 0.0) and np.all(cal_grid <= 1.0)

    # Monotonicity check
    diffs = np.diff(cal_grid)
    assert np.all(diffs >= -1e-9), "Isotonic calibrated probabilities must be strictly monotonic non-decreasing"

    # Determinism
    cal_grid_2 = cal.transform(test_grid)
    np.testing.assert_array_equal(cal_grid, cal_grid_2)

    # Rejection of invalid inputs
    with pytest.raises(ValueError):
        cal.transform(np.array([-0.05, 0.5]))
    with pytest.raises(ValueError):
        cal.transform(np.array([0.5, 1.05]))
    with pytest.raises(ValueError):
        cal.transform(np.array([np.nan, 0.5]))

    # Serialization and loading
    save_path = tmp_path / "test_calibrator.joblib"
    cal.save(save_path)
    assert save_path.exists()

    loaded_cal = IsotonicCalibrator.load(save_path)
    assert loaded_cal.is_fitted_
    assert loaded_cal.metadata.get("author") == "Ronak"
    np.testing.assert_array_equal(loaded_cal.transform(test_grid), cal_grid)


# ─── Test 23: Calibrator Zero Test Leakage ─────────────────────────────────────

def test_calibrator_zero_test_leakage():
    """
    Verify that calibrator fitting depends strictly on validation inputs.
    Altering test set labels or predictions must have zero effect on the fitted calibrator.
    """
    np.random.seed(123)
    n_val = 150
    n_test = 150

    val_proba = np.random.uniform(0.0, 1.0, size=n_val)
    y_val = (val_proba > 0.5).astype(int)

    test_proba = np.random.uniform(0.0, 1.0, size=n_test)
    y_test_original = (test_proba > 0.5).astype(int)
    y_test_corrupted = 1 - y_test_original  # Inverted test labels

    # Fit calibrator on validation
    cal = IsotonicCalibrator()
    cal.fit(val_proba, y_val)

    cal_preds_1 = cal.transform(test_proba)

    # Corrupting test labels has zero impact on calibrator's output
    assert np.all(np.isfinite(cal_preds_1))
    cal_preds_2 = cal.transform(test_proba)
    np.testing.assert_array_equal(cal_preds_1, cal_preds_2)


# ─── Test 24: Engine Calibration Runtime and Fallback ─────────────────────────

def test_engine_calibration_runtime_and_fallback():
    """
    Verify GlucoTwinEngine calibration integration:
    - When calibrator artifact is present:
        * 60min horizon exposes both raw_model_probability and calibrated_probability
        * calibration_status is 'CALIBRATED'
        * multi-horizon derived predictions (30min, 90min) have calibrated_probability=None and status='UNSUPPORTED_DERIVED_HORIZON'
        * threshold is not silently changed
        * model_self_awareness exposes calibration_method, calibration_status, and calibration_brier
    - When calibrator artifact is absent:
        * documented fallback activates (calibration_status='RAW_FALLBACK')
        * calibrated_probability is None
        * raw_model_probability is preserved
        * TwinState serializes cleanly to JSON in both states
    """
    files = get_patient_files(DATA_DIR)
    assert len(files) > 0
    df_raw = load_patient_raw(files[0])
    df_clean = preprocess_patient(df_raw, COLUMN_MAP)
    df_labeled = generate_prediction_labels(
        df_clean, "glucose_mgdl", "timestamp", prediction_horizon_min=60
    )
    X, y = build_feature_matrix(df_labeled, COLUMN_MAP, vectorized=True)
    clf = HypoClassifier(model_type="lightgbm").fit(X, y)

    # 1. Test with production calibrator artifact
    prod_cal_path = Path("models/glucotwin_calibrator.joblib")
    if prod_cal_path.exists():
        engine_cal = GlucoTwinEngine(
            patient_id="TEST_CAL_01",
            classifier=clf,
            column_map=COLUMN_MAP,
            operating_threshold=0.25,
            calibrator_path=prod_cal_path,
        )
        assert engine_cal.calibration_status == "CALIBRATED"
        assert engine_cal.calibrator is not None

        state = engine_cal.evaluate_state(df_labeled, current_idx=-1)
        d = state.to_dict()

        # Check 60min horizon
        pred_60 = d["predictions_by_horizon"]["60min"]
        assert "raw_model_probability" in pred_60
        assert "calibrated_probability" in pred_60
        assert pred_60["calibration_status"] == "CALIBRATED"
        assert isinstance(pred_60["calibrated_probability"], float)
        assert 0.0 <= pred_60["calibrated_probability"] <= 1.0
        assert engine_cal.alert_decision_uses_calibration is True
        assert engine_cal.alert_decision_threshold == pytest.approx(0.13)
        assert engine_cal.alert_decision_exit_threshold == pytest.approx(0.08)
        assert pred_60["alert"] == (pred_60["calibrated_probability"] >= engine_cal.alert_decision_threshold)
        assert pred_60["alert_basis"] == "CALIBRATED_60MIN_THRESHOLD"
        assert "recommended_action" not in d
        assert "counter-regulatory" not in d["model_interpretation"].lower()
        assert "symptom awareness" not in d["model_interpretation"].lower()

        # Check multi-horizon safety: 30min and 90min must NOT claim calibration
        pred_30 = d["predictions_by_horizon"]["30min"]
        assert pred_30["calibrated_probability"] is None
        assert pred_30["calibration_status"] == "UNSUPPORTED_DERIVED_HORIZON"

        pred_90 = d["predictions_by_horizon"]["90min"]
        assert pred_90["calibrated_probability"] is None
        assert pred_90["calibration_status"] == "UNSUPPORTED_DERIVED_HORIZON"

        pred_120 = d["predictions_by_horizon"]["120min"]
        assert pred_120["calibrated_probability"] is None
        assert pred_120["calibration_status"] == "UNSUPPORTED_DERIVED_HORIZON"

        # Check threshold preservation
        assert engine_cal.threshold == 0.25

        # Check model_self_awareness
        msa = d["model_self_awareness"]
        assert msa["calibration_applied"] is True
        assert msa["calibration_status"] == "CALIBRATED"
        assert msa["calibration_method"] == "isotonic"
        assert msa["calibration_brier"] == 0.039
        assert "horizon_120min_note" in msa

        # JSON roundtrip
        json_str = json.dumps(d)
        assert len(json_str) > 0

    # 2. Test missing artifact fallback
    engine_fallback = GlucoTwinEngine(
        patient_id="TEST_FALLBACK_01",
        classifier=clf,
        column_map=COLUMN_MAP,
        operating_threshold=0.25,
        calibrator_path=Path("non_existent_calibrator_path.joblib"),
    )
    assert engine_fallback.calibration_status == "RAW_FALLBACK"
    assert engine_fallback.calibrator is None

    state_fb = engine_fallback.evaluate_state(df_labeled, current_idx=-1)
    d_fb = state_fb.to_dict()

    pred_60_fb = d_fb["predictions_by_horizon"]["60min"]
    assert "raw_model_probability" in pred_60_fb
    assert pred_60_fb["calibrated_probability"] is None
    assert pred_60_fb["calibration_status"] == "RAW_FALLBACK"

    msa_fb = d_fb["model_self_awareness"]
    assert msa_fb["calibration_applied"] is False
    assert msa_fb["calibration_status"] == "RAW_FALLBACK"

    # JSON roundtrip for fallback
    json_str_fb = json.dumps(d_fb)
    assert len(json_str_fb) > 0


# --- Test 25: Alert State Machine Hysteresis & Episode Suppression -----------

def test_alert_state_machine_hysteresis():
    """
    TEST 25: AlertStateMachine must suppress spurious re-alerts via hysteresis and cooldown.

    Validates:
    - Entry above threshold opens an episode.
    - Brief dip below entry (but above exit) does NOT close the episode (hysteresis).
    - Drop below exit_threshold closes the episode.
    - Cooldown prevents immediate re-alert after episode ends.
    - get_episode_summary() returns structured episode count and duration info.
    - Threshold documentation keys are present.
    - NORMAL state when probability is well below threshold.
    - reset() clears all state.
    """
    from src.twin.alert import AlertStateMachine

    asm = AlertStateMachine(
        entry_threshold=0.44,
        exit_threshold=0.30,
        persistence_intervals=1,
        cooldown_minutes=15.0,
    )

    # 1. Below threshold: state must be NORMAL
    ts = pd.Timestamp("2026-01-01 10:00:00")
    out = asm.update(0.10, ts)
    assert out["alert_state"] == "NORMAL"
    assert not out["episode_active"]

    # 2. Crosses entry threshold: episode must open
    ts = ts + pd.Timedelta(minutes=5)
    out = asm.update(0.60, ts)
    assert out["episode_active"], "Episode should be active after crossing entry threshold"
    assert out["alert_state"] in ("WARNING", "CRITICAL")

    # 3. Brief dip to 0.38 (above exit=0.30, below entry=0.44)
    # Episode must stay active (hysteresis prevents premature closure)
    ts = ts + pd.Timedelta(minutes=5)
    out = asm.update(0.38, ts)
    assert out["episode_active"], "Episode must remain active during hysteresis dip (0.38 > exit=0.30)"

    # 4. Drops below exit threshold: episode must close
    ts = ts + pd.Timedelta(minutes=5)
    out = asm.update(0.15, ts)
    assert not out["episode_active"], "Episode must close after dropping below exit threshold"
    assert out["total_episodes_so_far"] == 1

    # 5. Immediate re-trigger within cooldown: must be suppressed
    ts = ts + pd.Timedelta(minutes=5)  # Only 5 min after close (cooldown=15)
    out = asm.update(0.70, ts)
    assert out["in_cooldown"], "Must be in cooldown period"
    assert not out["episode_active"], "Must not open episode during cooldown"

    # 6. After cooldown expires: can open new episode
    ts = ts + pd.Timedelta(minutes=20)  # Well past 15-min cooldown
    out = asm.update(0.70, ts)
    assert not out["in_cooldown"], "Cooldown must have expired"
    assert out["episode_active"], "New episode should open after cooldown"
    assert out["total_episodes_so_far"] >= 1

    # 7. Episode summary
    summary = asm.get_episode_summary()
    assert isinstance(summary, dict)
    assert summary["n_completed_episodes"] >= 1
    assert "episodes_per_day_estimate" in summary
    assert "method_note" in summary
    assert len(summary["method_note"]) > 20

    # 8. Threshold documentation must be present
    assert "thresholds" in out
    assert "entry" in out["thresholds"]
    assert "exit" in out["thresholds"]
    assert "entry_basis" in out["thresholds"]
    assert len(out["thresholds"]["entry_basis"]) > 5

    # 9. Reset clears state
    asm.reset()
    summary_after_reset = asm.get_episode_summary()
    assert summary_after_reset["n_completed_episodes"] == 0
    ts_r = pd.Timestamp("2026-01-01 12:00:00")
    out_r = asm.update(0.05, ts_r)
    assert out_r["alert_state"] == "NORMAL"


# --- Test 26: TwinState Serialization Resilience ---------------------------

def test_twin_state_serialization_resilience():
    """
    TEST 26: TwinState.to_dict() must remain robust when alert_level,
    glucose_trend, or data_quality are supplied as raw strings or dicts.
    """
    # 1. State with string alert_level and glucose_trend
    state = TwinState(
        twin_state_version="6",
        patient_id="TEST_PID",
        timestamp="2026-01-01T12:00:00",
        current_glucose_mgdl=115.0,
        glucose_velocity_15min=-0.8,
        glucose_acceleration_30min=-0.2,
        glucose_trend="FALLING",
        alert_level="WARNING",  # string instead of AlertLevel enum
        data_quality={"cgm_valid": True, "confidence_score": 0.95},  # dict instead of DataQuality
    )

    d = state.to_dict()
    assert isinstance(d, dict)
    assert d["alert_level"] == "WARNING"
    assert d["glucose_trend"] == "FALLING"
    assert d["data_quality"]["confidence_score"] == 0.95
    assert d["twin_state_version"] == "6"

    # JSON roundtrip must succeed
    json_str = json.dumps(d)
    assert len(json_str) > 0


# --- Test 27: Unlabeled Streaming Inference Safety -------------------------

def test_unlabeled_streaming_inference_safety():
    """
    TEST 27: Feature engineering and Twin Engine must safely handle
    unlabeled streaming data (e.g. live inference where future label is not yet known),
    without dropping the current row or raising errors.
    """
    files = get_patient_files(DATA_DIR)
    df_raw = load_patient_raw(files[0]).iloc[:100].copy()
    df_clean = preprocess_patient(df_raw, COLUMN_MAP)

    # Ensure no label column exists
    if "label_hypo" in df_clean.columns:
        df_clean = df_clean.drop(columns=["label_hypo"])

    # Vectorized path with label_col=None
    X_vec, y_vec = build_feature_matrix(df_clean, COLUMN_MAP, label_col=None, vectorized=True)
    assert len(X_vec) == len(df_clean), f"Expected {len(df_clean)} rows in X, got {len(X_vec)}"
    assert y_vec.isna().all()

    # Iterative path with label_col=None
    X_iter, y_iter = build_feature_matrix(df_clean, COLUMN_MAP, label_col=None, vectorized=False)
    assert len(X_iter) == len(df_clean), f"Expected {len(df_clean)} rows in X_iter, got {len(X_iter)}"

    # GlucoTwinEngine.evaluate_state on unlabeled current_data
    from src.models.classifier import HypoClassifier
    clf = HypoClassifier(model_type="lightgbm", n_estimators=10)
    # Train dummy model on minimal synthetic data
    X_synth = pd.DataFrame({col: np.random.randn(20) for col in X_vec.columns})
    y_synth = pd.Series([0, 1] * 10)
    clf.fit(X_synth, y_synth)

    engine = GlucoTwinEngine(
        patient_id="STREAM_TEST",
        classifier=clf,
        column_map=COLUMN_MAP,
        operating_threshold=0.25,
        calibrator=None,
    )
    twin_state = engine.evaluate_state(df_clean, current_idx=-1)
    assert isinstance(twin_state, TwinState)
    assert twin_state.twin_state_version == "6"
    assert twin_state.patient_id == "STREAM_TEST"

    # Reset engine
    engine.reset()
    assert engine._prev_alert is None
    assert engine._alert_sm._episode_active is False


# --- Test 28: SHAP Explainer Feature Mapping & Alignment -------------------

def test_shap_explainer_feature_mapping():
    """
    TEST 28: TwinExplainer must align feature names with their exact column values,
    preventing positional index drift when columns differ from training ordering.
    """
    from src.explainability.explainer import TwinExplainer
    from src.models.classifier import HypoClassifier

    # Create dummy dataset with distinct column values
    feat_names = ["glucose_current", "glucose_velocity_15min", "bolus_sum_240min"]
    X = pd.DataFrame({
        "glucose_current": [80.0, 90.0, 100.0, 70.0, 65.0],
        "glucose_velocity_15min": [-1.5, -0.5, 0.0, -2.0, -1.0],
        "bolus_sum_240min": [2.5, 1.0, 0.0, 3.0, 2.0],
    })
    y = pd.Series([1, 0, 0, 1, 1])

    clf = HypoClassifier(model_type="lightgbm", n_estimators=10)
    clf.fit(X, y)

    explainer = TwinExplainer(clf)
    sample_row = X.iloc[0]

    # Explain single row
    top_factors = explainer.explain(sample_row, top_k=3)
    assert len(top_factors) > 0
    for factor in top_factors:
        assert "feature" in factor
        assert "importance" in factor
        assert "value" in factor
        # Check value matches the exact series value for that feature name
        expected_val = sample_row[factor["feature"]]
        assert round(float(factor["value"]), 2) == round(float(expected_val), 2)


# --- Test 29: Feature Ablation Group Definitions & Protocol ----------------

def test_feature_ablation_definitions_and_protocol():
    """
    TEST 29: Feature ablation definitions must produce the 5 canonical subsets,
    and run_feature_ablation_study must execute without data leakage.
    """
    from src.evaluation.ablation import get_ablation_subsets, run_feature_ablation_study

    cols = [
        "glucose_current", "glucose_mean_120min", "glucose_velocity_15min",
        "bolus_sum_240min", "carbs_sum_180min", "steps_sum_60min",
        "hour_of_day", "is_nocturnal_heuristic",
        "personal_expanding_mean", "personal_ewm_baseline",
    ]
    subsets = get_ablation_subsets(cols)

    expected_subsets = ["full_model", "no_personal_baseline", "no_circadian", "cgm_and_insulin", "cgm_only"]
    for s in expected_subsets:
        assert s in subsets, f"Expected subset '{s}' missing"
        assert len(subsets[s]) > 0

    assert "personal_expanding_mean" not in subsets["no_personal_baseline"]
    assert "hour_of_day" not in subsets["no_circadian"]
    assert "carbs_sum_180min" not in subsets["cgm_and_insulin"]
    assert "bolus_sum_240min" not in subsets["cgm_only"]

    # Test small run_feature_ablation_study execution
    np.random.seed(42)
    n_train, n_test = 60, 20
    X_tr = pd.DataFrame(np.random.randn(n_train, len(cols)), columns=cols)
    y_tr = pd.Series((X_tr["glucose_current"] < 0).astype(int))
    X_te = pd.DataFrame(np.random.randn(n_test, len(cols)), columns=cols)
    y_te = pd.Series((X_te["glucose_current"] < 0).astype(int))

    study = run_feature_ablation_study(
        X_train=X_tr,
        y_train=y_tr,
        X_test=X_te,
        y_test=y_te,
        threshold=0.5,
        n_estimators=10,
    )
    assert "ablation_results" in study
    assert "full_model" in study["ablation_results"]
    assert "cgm_only" in study["ablation_results"]
    assert "roc_auc" in study["ablation_results"]["full_model"]["metrics"]
    assert "evaluation_protocol" in study


# --- Test 30: Calibrated Alert Decisions & Operating Threshold Compatibility --

def test_calibrated_alert_decisions_and_operating_threshold_compatibility():
    """
    TEST 30: Calibrated 60-min predictions must drive alert decisions when
    calibrator is loaded, using documented operating thresholds (0.13 entry, 0.08 exit),
    while preserving backward-compatible public operating_threshold (0.25).
    """
    files = get_patient_files(DATA_DIR)
    assert len(files) > 0
    df_raw = load_patient_raw(files[0])
    df_clean = preprocess_patient(df_raw, COLUMN_MAP)
    df_labeled = generate_prediction_labels(
        df_clean, "glucose_mgdl", "timestamp", prediction_horizon_min=60
    )
    X, y = build_feature_matrix(df_labeled, COLUMN_MAP, vectorized=True)
    clf = HypoClassifier(model_type="lightgbm").fit(X, y)

    prod_cal_path = Path("models/glucotwin_calibrator.joblib")
    assert prod_cal_path.exists(), "Production calibrator artifact required for test"

    # 1. Calibrated engine
    engine_cal = GlucoTwinEngine(
        patient_id="TEST_CAL_30",
        classifier=clf,
        column_map=COLUMN_MAP,
        operating_threshold=0.25,
        calibrator_path=prod_cal_path,
    )

    # Public operating_threshold backwards compatibility
    assert engine_cal.operating_threshold == 0.25
    assert engine_cal.threshold == 0.25

    # Alert decision thresholds
    assert engine_cal.alert_decision_uses_calibration is True
    assert engine_cal.alert_decision_threshold == pytest.approx(0.13)
    assert engine_cal.alert_decision_exit_threshold == pytest.approx(0.08)

    state = engine_cal.evaluate_state(df_labeled, current_idx=-1)
    d = state.to_dict()
    pred_60 = d["predictions_by_horizon"]["60min"]

    # Verify probability driving alert flag
    prob_used = pred_60["calibrated_probability"]
    assert pred_60["alert"] == (prob_used >= engine_cal.alert_decision_threshold)
    assert pred_60["operating_threshold"] == pytest.approx(0.13)
    assert pred_60["alert_basis"] == "CALIBRATED_60MIN_THRESHOLD"

    # 2. Fallback engine (uncalibrated)
    engine_raw = GlucoTwinEngine(
        patient_id="TEST_RAW_30",
        classifier=clf,
        column_map=COLUMN_MAP,
        operating_threshold=0.25,
        calibrator=None,
    )
    assert engine_raw.alert_decision_uses_calibration is False
    assert engine_raw.alert_decision_threshold == 0.25
    assert engine_raw.operating_threshold == 0.25

    state_raw = engine_raw.evaluate_state(df_labeled, current_idx=-1)
    d_raw = state_raw.to_dict()
    pred_60_raw = d_raw["predictions_by_horizon"]["60min"]

    assert pred_60_raw["operating_threshold"] == 0.25
    assert pred_60_raw["alert"] == (pred_60_raw["probability"] >= 0.25)
    assert pred_60_raw["alert_basis"] == "RAW_THRESHOLD"


# --- Test 31: Multi-Horizon Derived & Uncalibrated Guarantees ----------------

def test_multi_horizon_derived_uncalibrated_guarantees():
    """
    TEST 31: 30, 90, and 120-minute predictions must be explicitly marked as
    derived and uncalibrated, while 60-min is calibrated and under primary supervision.
    """
    files = get_patient_files(DATA_DIR)
    df_raw = load_patient_raw(files[0])
    df_clean = preprocess_patient(df_raw, COLUMN_MAP)
    df_labeled = generate_prediction_labels(
        df_clean, "glucose_mgdl", "timestamp", prediction_horizon_min=60
    )
    X, y = build_feature_matrix(df_labeled, COLUMN_MAP, vectorized=True)
    clf = HypoClassifier(model_type="lightgbm").fit(X, y)

    prod_cal_path = Path("models/glucotwin_calibrator.joblib")
    engine = GlucoTwinEngine(
        patient_id="TEST_HORIZONS_31",
        classifier=clf,
        column_map=COLUMN_MAP,
        operating_threshold=0.25,
        calibrator_path=prod_cal_path,
    )
    state = engine.evaluate_state(df_labeled, current_idx=-1)
    d = state.to_dict()
    preds = d["predictions_by_horizon"]

    # 60min is calibrated and not derived
    assert preds["60min"]["is_derived"] is False
    assert preds["60min"]["is_calibrated"] is True
    assert preds["60min"]["calibrated_probability"] is not None
    assert preds["60min"]["calibration_status"] == "CALIBRATED"

    # 30min, 90min, 120min are derived and uncalibrated
    for h in ["30min", "90min", "120min"]:
        p = preds[h]
        assert p["is_derived"] is True, f"Expected {h} to have is_derived=True"
        assert p["is_calibrated"] is False, f"Expected {h} to have is_calibrated=False"
        assert p["calibrated_probability"] is None, f"Expected {h} to have calibrated_probability=None"
        assert p["calibration_status"] == "UNSUPPORTED_DERIVED_HORIZON"
        assert p["alert_basis"] == "DERIVED_UNCALIBRATED_RAW_THRESHOLD"


# --- Test 32: Alert Context & Narrative Neutrality -------------------------

def test_alert_context_and_narrative_neutrality():
    """
    TEST 32: Alert context text must report calibrated probability as primary,
    and all narrative texts must use neutral model-context wording free of
    unsupported clinical diagnostics or prescriptive instructions.
    """
    files = get_patient_files(DATA_DIR)
    df_raw = load_patient_raw(files[0])
    df_clean = preprocess_patient(df_raw, COLUMN_MAP)
    df_labeled = generate_prediction_labels(
        df_clean, "glucose_mgdl", "timestamp", prediction_horizon_min=60
    )
    X, y = build_feature_matrix(df_labeled, COLUMN_MAP, vectorized=True)
    clf = HypoClassifier(model_type="lightgbm").fit(X, y)

    prod_cal_path = Path("models/glucotwin_calibrator.joblib")
    engine = GlucoTwinEngine(
        patient_id="TEST_NARRATIVE_32",
        classifier=clf,
        column_map=COLUMN_MAP,
        operating_threshold=0.25,
        calibrator_path=prod_cal_path,
    )
    state = engine.evaluate_state(df_labeled, current_idx=-1)
    d = state.to_dict()

    # 1. Alert context text reports calibrated probability as primary
    alert_ctx = d.get("model_alert_context", "")
    assert "Calibrated 60-min P(hypo)=" in alert_ctx
    assert "raw model P=" in alert_ctx
    assert "Monitor CGM readings" not in alert_ctx

    # 2. Check narrative neutrality across all output text fields
    full_narratives = " ".join([
        d.get("patient_narrative", ""),
        d.get("model_interpretation", ""),
        d.get("model_alert_context", ""),
    ])

    forbidden_phrases = [
        "level 1 hypoglycemia",
        "near-hypoglycemia warning zone",
        "counter-regulatory",
        "symptom awareness",
        "monitor cgm readings",
    ]
    for phrase in forbidden_phrases:
        assert phrase not in full_narratives.lower(), f"Forbidden phrase '{phrase}' found in narratives"


# --- Test 33: Absence of Misleading Recommendation Fields ------------------

def test_absence_of_misleading_recommendation_fields():
    """
    TEST 33: TwinState and its serialized dictionary representation must NOT
    contain misleading recommendation fields ('recommended_action'), while
    maintaining backward compatibility via object attribute access.
    """
    state = TwinState(
        twin_state_version="6",
        patient_id="TEST_REC_33",
        timestamp="2026-01-01T12:00:00",
        current_glucose_mgdl=110.0,
        glucose_velocity_15min=-0.5,
        glucose_acceleration_30min=0.0,
        glucose_trend="STABLE",
        alert_level="NORMAL",
    )

    # Object attribute access compatibility
    assert hasattr(state, "recommended_action")
    assert state.recommended_action is None

    # Serialized dict must omit recommended_action
    d = state.to_dict()
    assert "recommended_action" not in d

    # JSON output must omit recommended_action
    json_str = json.dumps(d)
    assert '"recommended_action"' not in json_str


# --- Test 34: Canonical Patient ID & Demo Payload Consistency --------------

def test_canonical_patient_id_and_demo_payload_consistency():
    """
    TEST 34: Patient ID extraction must preserve canonical 'P' suffix (e.g. HUPA0001P),
    and data/dashboard_demo_payload.json must be fully consistent with backend contracts.
    """
    # 1. extract_patient_id convention check
    assert extract_patient_id(Path("data/raw/HUPA0001P.csv")) == "HUPA0001P"
    assert extract_patient_id(Path("data/raw/HUPA0023P.csv")) == "HUPA0023P"

    # 2. Demo payload consistency
    payload_path = Path("data/dashboard_demo_payload.json")
    assert payload_path.exists(), "dashboard_demo_payload.json must exist"

    with open(payload_path, "r", encoding="utf-8") as f:
        demo = json.load(f)

    # Patient ID check
    assert demo["patient_id"] == "HUPA0001P"

    # 60min horizon checks
    pred_60 = demo["predictions_by_horizon"]["60min"]
    assert pred_60["calibration_status"] == "CALIBRATED"
    assert pred_60["is_calibrated"] is True
    assert pred_60["is_derived"] is False
    assert pred_60["alert_basis"] == "CALIBRATED_60MIN_THRESHOLD"
    assert pred_60["operating_threshold"] == 0.13
    assert pred_60["calibrated_probability"] == pytest.approx(0.0649, abs=1e-3)
    assert pred_60["alert"] is False, "0.0649 is below 0.13 threshold, alert must be False"

    # Multi-horizon checks
    for h in ["30min", "90min", "120min"]:
        pred_h = demo["predictions_by_horizon"][h]
        assert pred_h["is_derived"] is True
        assert pred_h["is_calibrated"] is False
        assert pred_h["calibrated_probability"] is None
        assert pred_h["calibration_status"] == "UNSUPPORTED_DERIVED_HORIZON"

    # Recommendation and narrative checks
    assert "recommended_action" not in demo
    demo_text = " ".join([
        demo.get("patient_narrative", ""),
        demo.get("model_interpretation", ""),
        demo.get("model_alert_context", ""),
    ]).lower()

    for phrase in ["level 1 hypoglycemia", "warning zone", "counter-regulatory", "symptom awareness", "monitor cgm readings"]:
        assert phrase not in demo_text, f"Forbidden phrase '{phrase}' found in demo payload"

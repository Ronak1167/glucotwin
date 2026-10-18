"""
GlucoTwin — Dashboard Integration and Simulation Tests
tests/test_dashboard_integration.py

Verifies end-to-end integration between:
- Production classifier and calibrator artifacts
- Preprocessed patient time series
- GlucoTwinEngine state evaluation, historical indexing, and OOD detection
- Live what-if simulation engine
- Graceful fallback on missing or corrupted artifacts
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import joblib

from src.data.loader import COLUMN_MAP, load_patient_raw
from src.data.preprocessing import preprocess_patient
from src.features.engineer import build_feature_matrix
from src.twin.engine import GlucoTwinEngine
from src.interface.api import GlucoTwinAPI
from src.simulation.simulator import MetabolicSimulator


MODEL_PATH = Path("models/glucotwin_production_model.joblib")
CALIBRATOR_PATH = Path("models/glucotwin_calibrator.joblib")
DATA_DIR = Path("data/raw/hupa_ucm/Preprocessed")
DEMO_PATH = Path("data/dashboard_demo_payload.json")


def test_production_artifacts_exist_and_load():
    """Verify serialized production model and calibrator exist and load cleanly."""
    assert MODEL_PATH.exists(), f"Missing production model at {MODEL_PATH}"
    clf = joblib.load(MODEL_PATH)
    assert hasattr(clf, "predict_proba"), "Classifier must implement predict_proba"
    assert hasattr(clf, "feature_names_"), "Classifier must have feature_names_"
    assert len(clf.feature_names_) == 40, f"Expected 40 features, got {len(clf.feature_names_)}"

    assert CALIBRATOR_PATH.exists(), f"Missing calibrator at {CALIBRATOR_PATH}"
    cal = joblib.load(CALIBRATOR_PATH)
    assert hasattr(cal, "transform"), "Calibrator must implement transform"
    p_test = np.array([0.1, 0.2, 0.5])
    cal_out = cal.transform(p_test)
    assert len(cal_out) == len(p_test)
    assert np.all((cal_out >= 0.0) & (cal_out <= 1.0))


def test_live_engine_snapshot_with_ood_and_history():
    """Verify live snapshot evaluates with OOD detector and historical episode matching."""
    patient_file = DATA_DIR / "HUPA0001P.csv"
    assert patient_file.exists(), f"Missing patient file: {patient_file}"

    df_raw = load_patient_raw(patient_file)
    df_pp = preprocess_patient(df_raw, COLUMN_MAP)
    assert len(df_pp) >= 20

    clf = joblib.load(MODEL_PATH)
    engine = GlucoTwinEngine(
        patient_id="HUPA0001P",
        classifier=clf,
        column_map=COLUMN_MAP,
    )
    X_all, _ = build_feature_matrix(df_pp, COLUMN_MAP, vectorized=True)
    engine.fit_ood_detector(X_all)
    engine.set_historical_index(df_pp, X_all)

    api = GlucoTwinAPI(engine=engine)
    snapshot = api.get_twin_snapshot(df_pp, current_idx=-1)

    assert snapshot["patient_id"] == "HUPA0001P"
    assert isinstance(snapshot["current_glucose_mgdl"], float)
    assert 0.0 <= snapshot["risk_score"] <= 100.0
    assert snapshot["alert_level"] in ["NORMAL", "WATCH", "WARNING", "CRITICAL"]

    # Historical episodes must be populated
    episodes = snapshot.get("similar_historical_episodes", [])
    assert len(episodes) > 0, "Historical indexing should return matched precedents"
    for ep in episodes:
        assert "similarity_score_pct" in ep
        assert "subsequent_nadir_mgdl" in ep

    # OOD confidence must be assessed
    confidence = snapshot.get("twin_confidence", {})
    assert confidence.get("confidence_level") in ["HIGH", "MODERATE", "LOW"]
    assert 0.0 <= confidence.get("confidence_score", -1.0) <= 1.0

    # Multi-horizon predictions
    preds = snapshot.get("predictions_by_horizon", {})
    for h in ["30min", "60min", "90min", "120min"]:
        assert h in preds
        assert "raw_model_probability" in preds[h]


def test_what_if_simulator_intervention_dynamics():
    """Verify metabolic simulator updates trajectory and detects threshold crossings correctly."""
    sim = MetabolicSimulator()

    # Scenario 1: Downward drift without intervention -> crosses below 70
    res_base = sim.simulate_trajectory(
        current_glucose=85.0,
        current_velocity=-0.8,
        active_iob=0.5,
        horizon_minutes=120,
        carb_intervention_g=0.0,
    )
    assert res_base["baseline"]["simulated_nadir_mgdl"] < 70.0
    assert res_base["counterfactual"]["simulated_threshold_crossing"] is True

    # Scenario 2: 20g fast-acting carbs -> prevents crossing
    res_rescue = sim.simulate_trajectory(
        current_glucose=85.0,
        current_velocity=-0.8,
        active_iob=0.5,
        horizon_minutes=120,
        carb_intervention_g=20.0,
        carb_type="fast",
    )
    assert res_rescue["counterfactual"]["simulated_nadir_mgdl"] > res_base["counterfactual"]["simulated_nadir_mgdl"]
    assert res_rescue["counterfactual"]["delta_nadir_mgdl"] > 0
    assert res_rescue["simulated_threshold_crossing_prevented"] is True


def test_demo_fallback_payload_integrity():
    """Verify static demo payload contains all required fields and valid types."""
    assert DEMO_PATH.exists()
    with open(DEMO_PATH, "r", encoding="utf-8") as f:
        demo = json.load(f)

    required_keys = [
        "patient_id", "timestamp", "current_glucose_mgdl", "glucose_velocity_15min",
        "glucose_trend", "risk_score", "alert_level", "is_nocturnal",
        "active_iob_u", "active_cob_g", "predictions_by_horizon",
        "top_risk_factors", "what_if_simulations", "similar_historical_episodes",
        "data_quality", "twin_confidence", "model_self_awareness",
    ]
    for key in required_keys:
        assert key in demo, f"Demo payload missing required key: {key}"

    assert isinstance(demo["current_glucose_mgdl"], (int, float))
    assert isinstance(demo["what_if_simulations"], list)
    assert len(demo["what_if_simulations"]) >= 1

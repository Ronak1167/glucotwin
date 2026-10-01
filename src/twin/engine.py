r"""
GlucoTwin — Digital Twin Intelligence Engine (v4)
src/twin/engine.py

Central intelligence coordinator implementing the "Twin Mirror" five questions:
  NOW:     What is happening? (current metabolic state, OOD confidence)
  NEXT:    What is likely to happen? (multi-horizon ML predictions)
  WHY:     Why did the Twin produce this prediction? (SHAP, explanation delta)
  MEMORY:  Have we seen this state before? (historical episode matching)
  WHAT-IF: How does the future change under a computational scenario?

SCIENTIFIC LANGUAGE POLICY:
- All outputs are "predicted", "estimated", "simulated", "model-derived"
- "model_alert_context" describes what the model output means — NOT clinical prescriptions
- "simulated_threshold_crossing_prevented" is a simulation flag — NOT a clinical guarantee
- The distinction between PREDICTION, SIMULATION, and CLINICAL DECISION is explicit

LEAKAGE GUARANTEE:
- evaluate_state() only receives data up to and including time T
- PatientMemory only retrieves records from before T
- HistoricalEpisodeMatcher enforces temporal separation
"""

from __future__ import annotations

import logging
import time as _time
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

from src.twin.state import TwinState, AlertLevel, GlucoseTrend, DataQuality
from src.twin.memory import PatientMemory, StateTransition
from src.twin.ood import OODDetector, OODResult
from src.twin.alert import AlertStateMachine
from src.models.classifier import HypoClassifier
from src.models.calibration import IsotonicCalibrator
from src.features.engineer import build_feature_matrix
from src.explainability.explainer import TwinExplainer
from src.similarity.matcher import HistoricalEpisodeMatcher
from src.simulation.simulator import MetabolicSimulator

logger = logging.getLogger(__name__)

# Validation-derived production alert thresholds. Raw and calibrated probabilities
# are different scales, so the corresponding operating thresholds must not be mixed.
CALIBRATED_ALERT_ENTRY_THRESHOLD = 0.13
CALIBRATED_ALERT_EXIT_THRESHOLD = 0.08


def _classify_trend(velocity_mgdl_per_min: float) -> str:
    """Map CGM rate-of-change velocity to named trend label."""
    v = velocity_mgdl_per_min
    if v > 2.0:
        return GlucoseTrend.RISING_FAST.value
    elif v > 1.0:
        return GlucoseTrend.RISING.value
    elif v > 0.5:
        return GlucoseTrend.RISING_SLOW.value
    elif v > -0.5:
        return GlucoseTrend.STABLE.value
    elif v > -1.0:
        return GlucoseTrend.FALLING_SLOW.value
    elif v > -2.0:
        return GlucoseTrend.FALLING.value
    else:
        return GlucoseTrend.FALLING_FAST.value


def _estimate_time_to_threshold(
    current_glucose: float,
    velocity_mgdl_per_min: float,
    threshold: float = 70.0,
    decay_constant: float = 0.015,
) -> Optional[float]:
    """
    Estimate minutes until glucose reaches threshold at current trajectory.

    Model: g(t) = g0 + v0*(1 - exp(-k*t))/k   [damped-velocity ODE]

    Assumptions:
    - Velocity decays exponentially with constant k (half-life ~46 min)
    - No new insulin or carbohydrate inputs after t=0
    - Parameters: k=0.015/min (fixed population value)

    EMPIRICAL VALIDATION (HUPA-UCM, 24/25 patients, 11,471 matched pairs):
    - MAE: 32.5 min  | Median AE: 17.6 min  | Std: 46.8 min
    - Mean signed bias: +10.8 min | Median signed bias: -0.1 min
    - Within 10 min: 36.2% | Within 15 min: 46.0% | Within 30 min: 65.1%
    - P25/P75/P90 AE: 6.1 / 42.6 / 77.7 min
    - Near-threshold (70-85 mg/dL): MAE=23.5 min, within-15min=59.2% (more reliable)
    - Slow drop (<1 mg/dL/min):     MAE=37.3 min (higher uncertainty)
    - Fast drop (>2 mg/dL/min):     MAE=24.1 min (more reliable)
    - Excluded: HUPA0009P — no threshold crossings in record
    - NOTE: Preliminary estimate from 6 patients (MAE=10.1 min) was unrepresentative.
      Full cohort reveals substantially larger error and high tail variance (max=2134 min).

    SCIENTIFIC CAVEATS (factual, no clinical interpretation):
    - Error magnitude (~33 min MAE) is large relative to the 60-min prediction window
    - Does NOT model insulin action, meal absorption, or counter-regulation
    - Output is a supplementary directional indicator only

    Returns None when:
    - Glucose already below threshold
    - Velocity positive (rising)
    - Asymptotic nadir above threshold (won't reach it)
    """
    if current_glucose <= threshold:
        return None
    if velocity_mgdl_per_min >= 0:
        return None

    g_nadir = current_glucose + velocity_mgdl_per_min / decay_constant
    if g_nadir >= threshold:
        return None

    try:
        term = 1.0 - ((threshold - current_glucose) * decay_constant / velocity_mgdl_per_min)
        if term <= 0:
            return 0.0
        t_est = -np.log(term) / decay_constant
        return round(float(t_est), 1) if t_est > 0 else 0.0
    except (ValueError, ZeroDivisionError):
        return None


def _compute_data_quality(row_feat: pd.Series) -> DataQuality:
    """Assess data quality indicators from feature row."""
    glucose_valid = pd.notna(row_feat.get("glucose_current", np.nan))
    glucose_val = float(row_feat.get("glucose_current", 100.0)) if glucose_valid else 0.0
    cgm_gap = 0.0
    if pd.isna(row_feat.get("glucose_velocity_15min", np.nan)):
        cgm_gap = 15.0
    insulin_available = pd.notna(row_feat.get("bolus_sum_240min", np.nan))
    meals_available = pd.notna(row_feat.get("carbs_sum_180min", np.nan))
    confidence = 1.0
    if not glucose_valid:
        confidence -= 0.5
    if cgm_gap >= 15:
        confidence -= 0.15
    if not insulin_available:
        confidence -= 0.15
    if not meals_available:
        confidence -= 0.10
    return DataQuality(
        cgm_gap_minutes=cgm_gap,
        cgm_valid=glucose_valid and 20.0 <= glucose_val <= 600.0,
        insulin_available=bool(insulin_available),
        meals_available=bool(meals_available),
        confidence_score=round(max(0.0, min(1.0, confidence)), 3),
    )


class GlucoTwinEngine:
    """
    Personalized Digital Twin for Hypoglycemia Prediction & Simulation.

    Implements the Twin Mirror five questions:
    NOW / NEXT / WHY / MEMORY / WHAT-IF

    Maintains state between calls to enable:
    - State transition tracking
    - Explanation deltas
    - Patient memory
    """

    def __init__(
        self,
        patient_id: str,
        classifier: HypoClassifier,
        column_map: Dict[str, str],
        operating_threshold: float = 0.25,
        calibrator: Optional[Any] = None,
        calibrator_path: Optional[str | Path] = Path("models/glucotwin_calibrator.joblib"),
    ):
        self.patient_id = patient_id
        self.classifier = classifier
        self.column_map = column_map
        self.threshold = operating_threshold

        self.explainer = TwinExplainer(self.classifier)
        self.matcher = HistoricalEpisodeMatcher(top_k=3)
        self.simulator = MetabolicSimulator()
        self.memory = PatientMemory(patient_id)
        self.ood_detector = OODDetector()

        # Calibration integration (Phase 9.2)
        self.calibrator = None
        self.calibration_status = "RAW_FALLBACK"
        if calibrator is not None:
            self.calibrator = calibrator
            self.calibration_status = "CALIBRATED"
            logger.debug(f"[{patient_id}] Using explicitly passed calibrator instance.")
        elif calibrator_path is not None:
            cal_p = Path(calibrator_path)
            if cal_p.exists():
                try:
                    self.calibrator = IsotonicCalibrator.load(cal_p)
                    self.calibration_status = "CALIBRATED"
                    logger.debug(f"[{patient_id}] Loaded production calibrator from {cal_p}")
                except Exception as e:
                    logger.warning(
                        f"[{patient_id}] Failed loading calibrator from {cal_p}: {e}. "
                        "Falling back to uncalibrated probabilities."
                    )
                    self.calibrator = None
                    self.calibration_status = "RAW_FALLBACK"
            else:
                self.calibrator = None
                self.calibration_status = "RAW_FALLBACK"

        # Previous state for transition tracking
        self._prev_alert: Optional[str] = None
        self._prev_risk: Optional[float] = None
        self._prev_timestamp: Optional[str] = None
        self._prev_features: Optional[Dict[str, float]] = None
        self._prev_shap: Optional[Dict[str, float]] = None

        # Alert state machine uses the same probability scale as the production decision.
        # Keep self.threshold unchanged for backward compatibility and benchmark callers.
        if self.calibration_status == "CALIBRATED":
            self.alert_decision_threshold = CALIBRATED_ALERT_ENTRY_THRESHOLD
            self.alert_decision_exit_threshold = CALIBRATED_ALERT_EXIT_THRESHOLD
            self.alert_decision_uses_calibration = True
        else:
            self.alert_decision_threshold = operating_threshold
            self.alert_decision_exit_threshold = round(operating_threshold * 0.68, 3)
            self.alert_decision_uses_calibration = False

        self._alert_sm = AlertStateMachine(
            entry_threshold=self.alert_decision_threshold,
            exit_threshold=self.alert_decision_exit_threshold,
            persistence_intervals=1,
            cooldown_minutes=15.0,
            use_calibrated=self.alert_decision_uses_calibration,
        )

    def fit_ood_detector(self, X_train: pd.DataFrame) -> None:
        """Fit OOD detector on training feature matrix. Must be training data only."""
        self.ood_detector.fit(X_train)

    def set_historical_index(
        self,
        df_preprocessed: pd.DataFrame,
        X_features: pd.DataFrame,
    ) -> None:
        """Index historical data for episode similarity. Must be training split only."""
        self.matcher.index_history(df_preprocessed, X_features)

    def evaluate_state(
        self,
        current_data: pd.DataFrame,
        current_idx: int = -1,
    ) -> TwinState:
        """
        Evaluate digital twin state at specified point in data stream.

        LEAKAGE GUARANTEE: Caller MUST NOT include future rows in current_data.

        Returns: TwinState — complete intelligence snapshot at time T.
        """
        t0_perf = _time.perf_counter()

        ts_col = "timestamp" if "timestamp" in current_data.columns else \
                 self.column_map.get("timestamp", "timestamp")

        # Build feature matrix (only from data up to T)
        label_col = "label_hypo" if "label_hypo" in current_data.columns else None
        X_all, _ = build_feature_matrix(
            current_data, self.column_map,
            label_col=label_col,
            vectorized=True,
        )

        if len(X_all) == 0:
            raise ValueError("Feature matrix is empty — insufficient data for inference.")

        row_feat = X_all.iloc[current_idx]
        row_raw = current_data.iloc[current_idx]

        timestamp_val = str(row_raw.get(ts_col, pd.Timestamp.now()))

        # ── NOW: Current metabolic state ─────────────────────────────────────
        g_curr = float(row_feat.get("glucose_current", 120.0))
        v_curr = float(row_feat.get("glucose_velocity_15min", 0.0))
        a_curr = float(row_feat.get("glucose_accel_30min", 0.0))  # acceleration
        if np.isnan(g_curr):
            g_curr = 100.0
        if np.isnan(v_curr):
            v_curr = 0.0
        if np.isnan(a_curr):
            a_curr = 0.0

        # Personalized baseline deviations
        g_dev_baseline = row_feat.get("glucose_deviation_from_baseline", np.nan)
        g_dev_ewm = row_feat.get("glucose_deviation_from_ewm", np.nan)
        g_noc_dev = row_feat.get("glucose_nocturnal_deviation", np.nan)
        personal_mean = row_feat.get("personal_expanding_mean", np.nan)
        personal_ewm = row_feat.get("personal_ewm_baseline", np.nan)

        is_nocturnal = bool(row_feat.get("is_nocturnal_heuristic", 0) == 1)
        iob = float(row_feat.get("bolus_sum_240min", 0.0))
        cob = float(row_feat.get("carbs_sum_180min", 0.0))
        if np.isnan(iob):
            iob = 0.0
        if np.isnan(cob):
            cob = 0.0

        glucose_trend = _classify_trend(v_curr)

        # ── NEXT: ML Prediction ───────────────────────────────────────────────
        t1 = _time.perf_counter()
        proba_60 = float(self.classifier.predict_proba(row_feat.to_frame().T)[0])
        inference_ms = round((_time.perf_counter() - t1) * 1000, 1)

        # Calibrated probability (60-minute primary horizon only)
        calibrated_proba_60: Optional[float] = None
        cal_status_60 = self.calibration_status
        if self.calibrator is not None and self.calibration_status == "CALIBRATED":
            try:
                calibrated_proba_60 = float(self.calibrator.transform(proba_60))
            except Exception as e:
                logger.warning(f"Calibration transform error: {e}")
                calibrated_proba_60 = None
                cal_status_60 = "RAW_FALLBACK"

        # The 60-minute calibrated prediction drives alert decisions when production calibrator is loaded.
        if calibrated_proba_60 is not None and cal_status_60 == "CALIBRATED":
            alert_probability = calibrated_proba_60
            alert_threshold = self.alert_decision_threshold
            alert_basis = "CALIBRATED_60MIN_THRESHOLD"
        else:
            alert_probability = proba_60
            alert_threshold = self.threshold
            alert_basis = "RAW_THRESHOLD"

        alert_flag_60 = bool(alert_probability >= alert_threshold)

        # Multi-horizon estimates (scaled from primary model)
        # NOTE: These are scaled approximations, not independently trained horizon models.
        # They are explicitly marked as derived and uncalibrated.
        proba_30 = float(np.clip(
            proba_60 * 0.85 + (0.30 if (g_curr < 85 and v_curr < -0.8) else 0.0), 0.0, 1.0
        ))
        proba_90 = float(np.clip(
            proba_60 * 0.95 + (0.15 if iob > 1.5 else 0.0), 0.0, 1.0
        ))
        proba_120 = float(np.clip(
            proba_60 * 0.90 + (0.20 if iob > 2.0 else 0.0), 0.0, 1.0
        ))

        predictions = {
            "30min": {
                "raw_model_probability": round(proba_30, 3),
                "calibrated_probability": None,
                "calibration_status": "UNSUPPORTED_DERIVED_HORIZON",
                "alert": bool(proba_30 >= self.threshold),
                "alert_basis": "DERIVED_UNCALIBRATED_RAW_THRESHOLD",
                "is_derived": True,
                "is_calibrated": False,
                "note": "Scaled estimate from 60-min model (uncalibrated)",
            },
            "60min": {
                "raw_model_probability": round(proba_60, 3),
                "calibrated_probability": round(calibrated_proba_60, 4) if calibrated_proba_60 is not None else None,
                "calibration_status": cal_status_60,
                "alert": alert_flag_60,
                "alert_basis": alert_basis,
                "operating_threshold": round(alert_threshold, 4),
                "is_derived": False,
                "is_calibrated": bool(cal_status_60 == "CALIBRATED"),
                "note": (
                    "Primary model prediction" +
                    (" (calibrated via isotonic regression)" if cal_status_60 == "CALIBRATED" else " (raw fallback)")
                ),
            },
            "90min": {
                "raw_model_probability": round(proba_90, 3),
                "calibrated_probability": None,
                "calibration_status": "UNSUPPORTED_DERIVED_HORIZON",
                "alert": bool(proba_90 >= self.threshold),
                "alert_basis": "DERIVED_UNCALIBRATED_RAW_THRESHOLD",
                "is_derived": True,
                "is_calibrated": False,
                "note": "Scaled estimate from 60-min model (uncalibrated)",
            },
            "120min": {
                "raw_model_probability": round(proba_120, 3),
                "calibrated_probability": None,
                "calibration_status": "UNSUPPORTED_DERIVED_HORIZON",
                "alert": bool(proba_120 >= self.threshold),
                "alert_basis": "DERIVED_UNCALIBRATED_RAW_THRESHOLD",
                "is_derived": True,
                "is_calibrated": False,
                "note": "Scaled estimate from 60-min model (uncalibrated)",
            },
        }

        # ── Risk Score ────────────────────────────────────────────────────────
        # Risk score follows the same primary probability scale used for alerting.
        risk_score = alert_probability * 70.0
        if g_curr < 70:
            risk_score = max(risk_score, 90.0)
        elif g_curr < 85 and v_curr < -0.5:
            risk_score = max(risk_score, 75.0)
        elif g_curr < 100 and v_curr < -1.0:
            risk_score = max(risk_score, 60.0)
        if is_nocturnal and risk_score > 35.0:
            risk_score = min(100.0, risk_score + 10.0)
        risk_score = round(float(np.clip(risk_score, 0.0, 100.0)), 1)

        # Alert level
        if risk_score >= 75.0:
            alert = AlertLevel.CRITICAL
        elif risk_score >= 50.0:
            alert = AlertLevel.WARNING
        elif risk_score >= 25.0:
            alert = AlertLevel.WATCH
        else:
            alert = AlertLevel.NORMAL

        # ── Time-to-Threshold (model-derived estimate) ────────────────────────
        estimated_ttt = _estimate_time_to_threshold(g_curr, v_curr, threshold=70.0)

        # ── Trajectory Collision ──────────────────────────────────────────────
        trajectory_collision = None
        if estimated_ttt is not None:
            # Assign uncertainty tier based on full-cohort stratified validation:
            # Near-threshold (g<85) + fast drop (v<-2): MAE~24 min, within-15=59%
            # Near-threshold (g<85) alone: MAE~24 min, within-15=59%
            # Slow drop (<1 mg/dL/min): MAE~37 min (high uncertainty)
            near_threshold = g_curr < 85.0
            fast_drop = v_curr < -2.0
            slow_drop = abs(v_curr) < 1.0
            if near_threshold and fast_drop:
                uncertainty_tier = "MODERATE"
                mae_note = "MAE~24 min in this regime (fast drop, near threshold)"
            elif near_threshold:
                uncertainty_tier = "LOW-MODERATE"
                mae_note = "MAE~24 min in this regime (near threshold)"
            elif slow_drop:
                uncertainty_tier = "LOW"
                mae_note = "MAE~37 min in this regime (slow drop)"
            else:
                uncertainty_tier = "LOW"
                mae_note = "MAE~33 min overall"

            trajectory_collision = {
                "crossing_predicted": True,
                "estimated_minutes_to_crossing": estimated_ttt,
                "model": "damped_velocity_ODE",
                "uncertainty_tier": uncertainty_tier,
                "validated_mae_note": mae_note,
                "scientific_note": (
                    "Estimated from damped-velocity trajectory model. "
                    "Full-cohort validation (24/25 patients, 11,471 pairs): "
                    "MAE=32.5 min, within-30min=65.1%. "
                    "This is a supplementary directional indicator only. "
                    "It does not account for insulin, meals, or counter-regulation."
                ),
            }
        else:
            trajectory_collision = {"crossing_predicted": False}

        # ── WHY: Explainability ───────────────────────────────────────────────
        t2 = _time.perf_counter()
        explanation = self.explainer.explain_sample(row_feat, top_k=5)
        shap_ms = round((_time.perf_counter() - t2) * 1000, 1)

        top_risk_factors = explanation.get("top_risk_factors", [])
        model_interpretation = explanation.get("clinical_narrative", "")

        # Build SHAP dict for delta computation
        curr_shap = {f["feature"]: f.get("shap_value", 0.0) for f in top_risk_factors}

        # ── State Transition & Explanation Delta ──────────────────────────────
        state_transition = None
        explanation_delta = None

        if self._prev_alert is not None and self._prev_risk is not None:
            changed = (alert.value != self._prev_alert)
            transition = StateTransition(
                timestamp=timestamp_val,
                old_alert=self._prev_alert,
                new_alert=alert.value,
                old_risk=self._prev_risk,
                new_risk=risk_score,
                dominant_reason=(
                    top_risk_factors[0]["feature"] if top_risk_factors else "unknown"
                ),
                glucose_at_transition=g_curr,
            )

            if changed:
                self.memory.record_transition(transition)

            state_transition = {
                "changed": changed,
                "old_alert": self._prev_alert,
                "new_alert": alert.value,
                "old_risk": self._prev_risk,
                "new_risk": risk_score,
                "dominant_reason": transition.dominant_reason,
                "timestamp": timestamp_val,
            }

            # Explanation delta between this state and previous
            if self._prev_timestamp is not None:
                delta = self.memory.compute_explanation_delta(
                    prev_state_features=self._prev_features,
                    curr_state_features=row_feat.to_dict(),
                    prev_shap=self._prev_shap,
                    curr_shap=curr_shap,
                    from_timestamp=self._prev_timestamp,
                    to_timestamp=timestamp_val,
                    old_risk=self._prev_risk,
                    new_risk=risk_score,
                    old_alert=self._prev_alert,
                    new_alert=alert.value,
                )
                explanation_delta = {
                    "risk_delta": delta.risk_delta,
                    "alert_change": delta.alert_change,
                    "dominant_change_reason": delta.dominant_change_reason,
                    "feature_changes": delta.feature_changes[:3],
                }

        # Record current state to memory
        self.memory.record_state(
            timestamp=timestamp_val,
            risk_score=risk_score,
            alert_level=alert.value,
            glucose=g_curr,
            dominant_factor=top_risk_factors[0]["feature"] if top_risk_factors else "",
        )

        # Update previous state for next call
        self._prev_alert = alert.value
        self._prev_risk = risk_score
        self._prev_timestamp = timestamp_val
        self._prev_features = row_feat.to_dict()
        self._prev_shap = curr_shap

        # ── Model Alert Context (language-correct, non-prescriptive) ──────────
        ttt_str = (
            f" The model estimates approximately {estimated_ttt:.0f} minutes to "
            f"threshold at the current trajectory (full-cohort MAE: 32.5 min, median AE: 17.6 min)."
            if estimated_ttt is not None else ""
        )
        if alert == AlertLevel.CRITICAL:
            model_alert_context = (
                f"CRITICAL: Model predicted probability {alert_probability:.1%} "
                f"for threshold crossing within 60 min."
                f"{' (calibrated; raw model probability ' + format(proba_60, '.1%') + ')' if calibrated_proba_60 is not None else ''}"
                f"{ttt_str} "
                f"{'Nocturnal context is included as a model feature.' if is_nocturnal else ''}"
            )
        elif alert == AlertLevel.WARNING:
            model_alert_context = (
                f"WARNING: Model predicted probability {alert_probability:.1%} "
                f"for threshold crossing within 60 min."
                f"{' (calibrated; raw model probability ' + format(proba_60, '.1%') + ')' if calibrated_proba_60 is not None else ''}"
                f"{ttt_str}"
            )
        elif alert == AlertLevel.WATCH:
            model_alert_context = (
                f"WATCH: Mild downward drift detected. Model probability {alert_probability:.1%}."
            )
        else:
            model_alert_context = (
                f"NORMAL: Model probability {alert_probability:.1%}. "
                "Metabolic trajectory appears stable."
            )

        # ── MEMORY: Historical Episode Matching ───────────────────────────────
        t3 = _time.perf_counter()
        t_curr = pd.to_datetime(timestamp_val)
        similar_episodes = self.matcher.find_similar_episodes(
            current_features=row_feat,
            current_time=t_curr,
            top_k=3,
        )
        matcher_ms = round((_time.perf_counter() - t3) * 1000, 1)

        # Historical memory summary
        recent_timeline = self.memory.get_recent_risk_timeline(timestamp_val, n=10)
        past_episodes = self.memory.get_past_episodes(timestamp_val)
        historical_memory = {
            "n_past_episodes": len(past_episodes),
            "n_state_transitions": len(self.memory._transitions),
            "recent_risk_timeline": recent_timeline[-5:],
            "memory_summary": self.memory.get_summary(),
        }

        # ── WHAT-IF: Computational Scenarios ─────────────────────────────────
        t4 = _time.perf_counter()
        simulations = []

        # Scenario A: No intervention (baseline trajectory)
        sim_baseline = self.simulator.simulate_trajectory(
            current_glucose=g_curr, current_velocity=v_curr,
            active_iob=iob, horizon_minutes=120, carb_intervention_g=0.0,
        )
        simulations.append({
            "scenario": "No intervention — baseline simulated trajectory",
            "simulated_threshold_crossing_prevented": False,
            "baseline": sim_baseline["baseline"],
            "counterfactual": sim_baseline["counterfactual"],
            "simulated_nadir_mgdl": sim_baseline["min_glucose_simulated"],
            "trajectory": sim_baseline["trajectory"][:13],
        })

        # Scenario B: 15g fast-acting carbohydrate
        sim_carbs_15 = self.simulator.simulate_trajectory(
            current_glucose=g_curr, current_velocity=v_curr,
            active_iob=iob, horizon_minutes=120,
            carb_intervention_g=15.0, carb_type="fast",
        )
        simulations.append({
            "scenario": "15g fast-acting carbohydrate (computational scenario)",
            "simulated_threshold_crossing_prevented": sim_carbs_15["simulated_threshold_crossing_prevented"],
            "baseline": sim_carbs_15["baseline"],
            "counterfactual": sim_carbs_15["counterfactual"],
            "simulated_nadir_mgdl": sim_carbs_15["min_glucose_simulated"],
            "trajectory": sim_carbs_15["trajectory"][:13],
            "scenario_note": (
                "The simulator predicts NO threshold crossing under this scenario. "
                "This is a computational result — not a clinical recommendation."
                if sim_carbs_15["simulated_threshold_crossing_prevented"]
                else "Simulator predicts threshold crossing may still occur under this scenario."
            ),
        })

        # Scenario C: 25g bedtime complex carbohydrate
        sim_snack_25 = self.simulator.simulate_trajectory(
            current_glucose=g_curr, current_velocity=v_curr,
            active_iob=iob, horizon_minutes=120,
            carb_intervention_g=25.0, carb_type="complex",
        )
        simulations.append({
            "scenario": "25g complex carbohydrate snack (computational scenario)",
            "simulated_threshold_crossing_prevented": sim_snack_25["simulated_threshold_crossing_prevented"],
            "baseline": sim_snack_25["baseline"],
            "counterfactual": sim_snack_25["counterfactual"],
            "simulated_nadir_mgdl": sim_snack_25["min_glucose_simulated"],
            "trajectory": sim_snack_25["trajectory"][:13],
        })
        sim_ms = round((_time.perf_counter() - t4) * 1000, 1)

        # ── Alert State Machine update (Phase 9.8) ──────────────────────────────
        alert_sm_output = self._alert_sm.update(
            probability=alert_probability,
            timestamp=timestamp_val,
            risk_score=risk_score,
        )
        alert_episode_summary = self._alert_sm.get_episode_summary()

        # ── OOD / Twin Confidence ─────────────────────────────────────────────
        dq = _compute_data_quality(row_feat)
        ood_result = self.ood_detector.assess(row_feat, cgm_gap_minutes=dq.cgm_gap_minutes)
        twin_confidence = ood_result.to_dict()

        # ── Model Self-Awareness (v6: personalization + alert SM + test metrics) ───
        # All fields describe model behavior on HUPA-UCM — not clinical outcomes.
        model_self_awareness = {
            # TTT trajectory model
            "ttt_model": "damped_velocity_ODE",
            "ttt_mae_min": 32.5,
            "ttt_median_ae_min": 17.6,
            "ttt_within_30min_pct": 65.1,
            "ttt_validated_n": 11471,
            "ttt_validated_patients": "24/25",
            "ttt_preliminary_mae_note": (
                "Preliminary estimate (6 patients, MAE=10.1 min) was unrepresentative. "
                "Full-cohort (24 patients, n=11471, MAE=32.5 min) supersedes it."
            ),
            # Prediction model
            "prediction_horizon_min": 60,
            "horizon_30min_note": "Scaled estimate from 60-min model (not independently trained)",
            "horizon_90min_note": "Scaled estimate from 60-min model (not independently trained)",
            "horizon_120min_note": "Scaled estimate from 60-min model (not independently trained)",
            # LOPO cross-validation
            "lopo_roc_auc": 0.970,
            "lopo_roc_auc_std": 0.027,
            "lopo_sensitivity": 0.958,
            "lopo_specificity": 0.821,
            "lopo_f2": 0.754,
            "lopo_n_patients": 24,
            "lopo_threshold_used": 0.25,
            "operating_threshold": self.threshold,
            "production_alert_threshold": self.alert_decision_threshold,
            "production_alert_threshold_scale": ("calibrated" if self.alert_decision_uses_calibration else "raw"),
            "production_alert_exit_threshold": self.alert_decision_exit_threshold,
            "lopo_evaluated": True,
            # Blind test (chronological holdout)
            "test_roc_auc": 0.951,
            "test_sensitivity": 0.867,
            "test_specificity": 0.889,
            "test_f2": 0.751,
            "test_brier_raw": 0.069,
            "test_brier_calibrated": 0.039,
            # Calibration
            "calibration_applied": bool(cal_status_60 == "CALIBRATED"),
            "calibration_status": cal_status_60,
            "calibration_method": "isotonic" if cal_status_60 == "CALIBRATED" else "none",
            "calibration_brier": 0.039,
            "calibration_brier_raw": 0.069,
            "calibration_brier_isotonic": 0.039,
            "calibration_brier_platt": 0.0427,
            "calibration_recommended": "isotonic",
            "calibration_scope": "60-minute prediction horizon on HUPA-UCM validation cohort",
            "calibration_note": (
                "Isotonic calibration fit on validation split (Brier=0.0390 vs raw=0.0690 on test set). "
                f"Live inference status: {cal_status_60}."
            ),
            # Personalization live status
            "personalization": {
                "personal_baseline_available": bool(
                    isinstance(personal_mean, float) and not np.isnan(personal_mean)
                ),
                "ewm_baseline_available": bool(
                    isinstance(personal_ewm, float) and not np.isnan(personal_ewm)
                ),
                "glucose_deviation_from_personal_mean": (
                    round(float(g_dev_baseline), 1)
                    if isinstance(g_dev_baseline, (int, float)) and not np.isnan(float(g_dev_baseline) if not isinstance(g_dev_baseline, float) else g_dev_baseline)
                    else None
                ),
                "glucose_deviation_from_ewm": (
                    round(float(g_dev_ewm), 1)
                    if isinstance(g_dev_ewm, (int, float)) and not np.isnan(float(g_dev_ewm) if not isinstance(g_dev_ewm, float) else g_dev_ewm)
                    else None
                ),
                "glucose_nocturnal_deviation": (
                    round(float(g_noc_dev), 1)
                    if isinstance(g_noc_dev, (int, float)) and not np.isnan(float(g_noc_dev) if not isinstance(g_noc_dev, float) else g_noc_dev)
                    else None
                ),
                "circadian_context": "time-of-day 3-hour bins + nocturnal heuristic (23:00-06:00)",
            },
            # Alert state machine
            "alert_state_machine": alert_sm_output,
            "alert_episode_summary": alert_episode_summary,
            # Dataset
            "dataset": "HUPA-UCM",
            "dataset_n_patients": 24,
            "dataset_type": "retrospective CGM",
            "twin_state_version": "6",
            "scientific_note": (
                "All metrics from HUPA-UCM only. "
                "No clinical validation performed. "
                "Generalization to other populations or devices is unknown."
            ),
        }

        total_ms = round((_time.perf_counter() - t0_perf) * 1000, 1)
        logger.debug(
            f"[{self.patient_id}] evaluate_state: "
            f"total={total_ms}ms inference={inference_ms}ms "
            f"shap={shap_ms}ms matcher={matcher_ms}ms sim={sim_ms}ms"
        )

        return TwinState(
            twin_state_version="6",
            patient_id=self.patient_id,
            timestamp=timestamp_val,
            current_glucose_mgdl=round(g_curr, 1),
            glucose_velocity_15min=round(v_curr, 3),
            glucose_acceleration_30min=round(a_curr, 3),
            glucose_trend=glucose_trend,
            risk_score=risk_score,
            alert_level=alert,
            is_nocturnal=is_nocturnal,
            active_iob_u=round(iob, 3),
            active_cob_g=round(cob, 1),
            predictions_by_horizon=predictions,
            estimated_time_to_threshold_min=estimated_ttt,
            top_risk_factors=top_risk_factors,
            model_interpretation=model_interpretation,
            clinical_narrative=model_interpretation,
            model_alert_context=model_alert_context,
            what_if_simulations=simulations,
            similar_historical_episodes=similar_episodes,
            data_quality=dq,
            twin_confidence=twin_confidence,
            state_transition=state_transition,
            explanation_delta=explanation_delta,
            historical_memory=historical_memory,
            trajectory_collision=trajectory_collision,
            model_self_awareness=model_self_awareness,
            # Personalization fields (Phase 9.5/9.6)
            glucose_deviation_from_baseline=(
                round(float(g_dev_baseline), 1)
                if isinstance(g_dev_baseline, (int, float)) and not np.isnan(g_dev_baseline)
                else None
            ),
            glucose_deviation_from_ewm=(
                round(float(g_dev_ewm), 1)
                if isinstance(g_dev_ewm, (int, float)) and not np.isnan(g_dev_ewm)
                else None
            ),
        )

    def reset(self) -> None:
        """Reset internal session state, previous transition cache, and alert state machine."""
        self._prev_alert = None
        self._prev_risk = None
        self._prev_timestamp = None
        self._prev_features = None
        self._prev_shap = None
        if hasattr(self, "_alert_sm") and self._alert_sm is not None:
            self._alert_sm.reset()

    @property
    def operating_threshold(self) -> float:
        """Public operating threshold field (kept backward compatible for callers/benchmarks)."""
        return self.threshold

    @operating_threshold.setter
    def operating_threshold(self, value: float) -> None:
        self.threshold = value


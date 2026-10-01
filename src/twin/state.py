r"""
GlucoTwin — Digital Twin State Schema (v4)
src/twin/state.py

Defines the core state schema representing a patient's digital twin snapshot at time T.

Schema contract between Ronak's intelligence engine and Aishni's frontend dashboard.
All fields are typed, documented, and serializable to JSON.

SCIENTIFIC LANGUAGE POLICY:
- Use "predicted", "estimated", "simulated", "model-derived", "associated with"
- Never use "guarantees", "prevents", "treats", "prescribes", "clinically proven"
- Distinction between PREDICTION, SIMULATION, and CLINICAL DECISION is EXPLICIT
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional
from enum import Enum


class AlertLevel(str, Enum):
    NORMAL = "NORMAL"        # Low risk — no model-predicted threshold crossing imminent
    WATCH = "WATCH"          # Mild downward drift or entering watch zone (25-50)
    WARNING = "WARNING"      # Model predicts threshold crossing within horizon (50-75)
    CRITICAL = "CRITICAL"    # Model predicts imminent threshold crossing (>= 75)


class GlucoseTrend(str, Enum):
    """Clinical glucose trend — mirrors CGM arrow conventions."""
    RISING_FAST = "RISING_FAST"       # > +2 mg/dL/min
    RISING = "RISING"                  # +1 to +2 mg/dL/min
    RISING_SLOW = "RISING_SLOW"        # +0.5 to +1 mg/dL/min
    STABLE = "STABLE"                  # -0.5 to +0.5 mg/dL/min
    FALLING_SLOW = "FALLING_SLOW"      # -1 to -0.5 mg/dL/min
    FALLING = "FALLING"                # -2 to -1 mg/dL/min
    FALLING_FAST = "FALLING_FAST"      # < -2 mg/dL/min


@dataclass
class DataQuality:
    """
    Data quality flags at time T.
    Enables the frontend to show appropriate confidence indicators.
    """
    cgm_gap_minutes: float = 0.0
    cgm_valid: bool = True
    insulin_available: bool = True
    meals_available: bool = True
    confidence_score: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cgm_gap_minutes": self.cgm_gap_minutes,
            "cgm_valid": self.cgm_valid,
            "insulin_available": self.insulin_available,
            "meals_available": self.meals_available,
            "confidence_score": round(self.confidence_score, 3),
        }


@dataclass
class TwinState:
    """
    Complete snapshot of the patient's Digital Twin state at time T.

    Versioned at v5: adds model_self_awareness block (validated TTT metrics,
    prediction horizon honesty), updates TTT validated statistics to full cohort.

    This is the core data contract between:
    - Ronak's intelligence engine (producer)
    - Aishni's dashboard (consumer)

    All fields are JSON-serializable via to_dict().
    Backward compatible with v3 — all existing fields preserved.
    """
    # ── Version ─────────────────────────────────────────────────────────────
    twin_state_version: str = "6"

    # ── Identity ─────────────────────────────────────────────────────────────
    patient_id: str = ""
    timestamp: str = ""

    # ── Current Metabolic State ───────────────────────────────────────────────
    current_glucose_mgdl: float = 100.0
    glucose_velocity_15min: float = 0.0      # mg/dL per minute; negative = falling
    glucose_acceleration_30min: float = 0.0  # velocity change over 30 min (v6 new)
    glucose_trend: str = "STABLE"

    # ── Risk Assessment ───────────────────────────────────────────────────────
    risk_score: float = 0.0                    # Composite model risk 0–100
    alert_level: AlertLevel = AlertLevel.NORMAL
    is_nocturnal: bool = False

    # ── Physiology Context ────────────────────────────────────────────────────
    active_iob_u: float = 0.0                  # Estimated insulin on board (Units)
    active_cob_g: float = 0.0                  # Estimated carbs on board (grams)

    # ── Predictions ───────────────────────────────────────────────────────────
    predictions_by_horizon: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # ── Time-to-Threshold (Model-Derived Estimate) ────────────────────────────
    # SCIENTIFIC NOTE: This is a model-derived trajectory estimate, NOT a clinical guarantee.
    # It is computed from a damped-velocity ODE (decay constant k=0.015/min).
    # Full-cohort validation (HUPA-UCM, 11,471 matched pairs, 24/25 patients):
    #   MAE = 32.5 min | Median AE = 17.6 min | Bias (mean) = +10.8 min
    #   Within 15 min: 46.0% | Within 30 min: 65.1%
    #   P25/P75/P90 AE: 6.1 / 42.6 / 77.7 min
    # Preliminary estimate (6 patients, MAE=10.1 min) was unrepresentative.
    # Returns None when glucose is rising, stable, or already below threshold.
    estimated_time_to_threshold_min: Optional[float] = None

    # ── Explainability ────────────────────────────────────────────────────────
    top_risk_factors: List[Dict[str, Any]] = field(default_factory=list)
    model_interpretation: str = ""             # Replaces "clinical_narrative" (v4 rename)
    clinical_narrative: str = ""               # Kept for backward compatibility

    # ── Model Interpretation (NOT a clinical prescription) ───────────────────
    model_alert_context: str = ""              # What the model's output means in context

    # ── What-If Computational Scenarios ──────────────────────────────────────
    # SCIENTIFIC NOTE: These are computational simulations under altered input conditions.
    # "simulated_threshold_crossing_prevented" means the model's simulation did NOT predict
    # a threshold crossing under this scenario. It does NOT guarantee real-world outcomes.
    what_if_simulations: List[Dict[str, Any]] = field(default_factory=list)

    # ── Historical Precedents (Strictly Temporal) ─────────────────────────────
    similar_historical_episodes: List[Dict[str, Any]] = field(default_factory=list)

    # ── Data Quality ──────────────────────────────────────────────────────────
    data_quality: Optional[DataQuality] = None

    # ── OOD / Twin Confidence (v4 new) ────────────────────────────────────────
    twin_confidence: Optional[Dict[str, Any]] = None
    # Fields: confidence_level, confidence_score, out_of_distribution, reasons

    # ── State Transition (v4 new) ─────────────────────────────────────────────
    state_transition: Optional[Dict[str, Any]] = None
    # Fields: old_alert, new_alert, old_risk, new_risk, dominant_reason, changed

    # ── Explanation Delta (v4 new) ────────────────────────────────────────────
    explanation_delta: Optional[Dict[str, Any]] = None
    # Fields: risk_delta, alert_change, feature_changes, dominant_change_reason

    # ── Historical Memory Summary (v4 new) ────────────────────────────────────
    historical_memory: Optional[Dict[str, Any]] = None
    # Fields: n_past_episodes, n_transitions, recent_risk_timeline

    # ── Trajectory Collision (v4) ─────────────────────────────────────────────
    trajectory_collision: Optional[Dict[str, Any]] = None
    # Fields: crossing_predicted, estimated_minutes_to_crossing, uncertainty_tier,
    #         validated_mae_note, scientific_note

    # ── Model Self-Awareness (v5 new) ─────────────────────────────────────────
    # Exposes the validated limitations and confidence of each component.
    # Enables the frontend to show honest uncertainty indicators.
    # SCIENTIFIC LANGUAGE: All fields describe model behavior, not clinical outcomes.
    model_self_awareness: Optional[Dict[str, Any]] = None
    # Fields:
    #   ttt_mae_min: validated MAE of time-to-threshold estimate (full cohort)
    #   ttt_within_30min_pct: pct of estimates within 30 min of actual crossing
    #   prediction_horizon_min: the labeled horizon the classifier was trained on
    #   horizon_scaling_note: multi-horizon predictions are scaled, not independent
    #   lopo_roc_auc: LOPO cross-validation ROC-AUC (populated after LOPO run)
    #   lopo_sensitivity: LOPO recall (populated after LOPO run)
    #   calibration_brier: Brier score (populated after calibration analysis)

    # Personalization Context (v6 new)
    # Difference between current glucose and patient's own historical baseline.
    # None when personal baseline is not yet available (< 20 readings in history).
    # Strictly backward-looking: computed only from data before time T.
    glucose_deviation_from_baseline: Optional[float] = None  # vs. time-of-day bin mean
    glucose_deviation_from_ewm: Optional[float] = None        # vs. EWM baseline (alpha=0.05)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["alert_level"] = self.alert_level.value
        d["glucose_trend"] = self.glucose_trend

        # Keep backward-compatible field names
        d["time_to_threshold_min"] = self.estimated_time_to_threshold_min
        if not d["clinical_narrative"] and d["model_interpretation"]:
            d["clinical_narrative"] = d["model_interpretation"]

        if self.data_quality is not None:
            d["data_quality"] = self.data_quality.to_dict()
        return d

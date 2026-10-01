r"""
GlucoTwin — Patient Memory Abstraction
src/twin/memory.py

Provides a strictly temporal patient memory for the Digital Twin.
All memory retrieval at time T can ONLY access records from <= T.

Memory stores:
- Previous hypoglycemic episodes
- State transition history
- Historical risk states
- Dominant risk factor patterns

This is one of the core differentiators of a Digital Twin vs. a classifier:
the ability to reflect on its own past and use it to inform the present.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StateTransition:
    """Records a single alert-level transition in the twin's history."""
    timestamp: str
    old_alert: str
    new_alert: str
    old_risk: float
    new_risk: float
    dominant_reason: str   # Primary feature/explanation driving the transition
    glucose_at_transition: float


@dataclass
class EpisodeRecord:
    """Records a historical hypoglycemic episode seen by the twin."""
    episode_id: int
    start_time: str
    nadir_mgdl: float
    duration_min: float
    was_nocturnal: bool
    alert_triggered: bool  # Did the twin issue a warning before this?
    lead_time_min: Optional[float]  # Minutes of advance warning (None if no alert)


@dataclass
class ExplanationDelta:
    """Records the change in top explanatory factors between two states."""
    from_timestamp: str
    to_timestamp: str
    risk_delta: float
    alert_change: str   # e.g. "NORMAL → WARNING"
    feature_changes: List[Dict[str, Any]]  # [{"feature", "old_value", "new_value", "shap_delta"}]
    dominant_change_reason: str


class PatientMemory:
    """
    Temporal memory store for a single patient's Digital Twin.

    Provides:
    - State transition log
    - Episode history
    - Explanation deltas
    - Leakage-safe temporal retrieval

    LEAKAGE GUARANTEE: All `get_*` methods accept a `before_time` parameter.
    Records at or after `before_time` are NEVER returned.
    """

    def __init__(self, patient_id: str, max_history: int = 2000):
        self.patient_id = patient_id
        self.max_history = max_history
        self._transitions: List[StateTransition] = []
        self._episodes: List[EpisodeRecord] = []
        self._explanation_deltas: List[ExplanationDelta] = []
        self._risk_timeline: List[Dict[str, Any]] = []

    # ─── Writers ────────────────────────────────────────────────────────────

    def record_state(
        self,
        timestamp: str,
        risk_score: float,
        alert_level: str,
        glucose: float,
        dominant_factor: str = "",
    ) -> None:
        """Record a twin state snapshot to risk timeline."""
        self._risk_timeline.append({
            "timestamp": timestamp,
            "risk_score": round(risk_score, 1),
            "alert_level": alert_level,
            "glucose_mgdl": round(glucose, 1),
            "dominant_factor": dominant_factor,
        })
        if len(self._risk_timeline) > self.max_history:
            self._risk_timeline = self._risk_timeline[-self.max_history:]

    def record_transition(self, transition: StateTransition) -> None:
        """Record an alert-level state transition."""
        self._transitions.append(transition)
        if len(self._transitions) > self.max_history:
            self._transitions = self._transitions[-self.max_history:]

    def record_episode(self, episode: EpisodeRecord) -> None:
        """Record a detected hypoglycemic episode."""
        self._episodes.append(episode)

    def record_explanation_delta(self, delta: ExplanationDelta) -> None:
        """Record an explanation delta between two consecutive states."""
        self._explanation_deltas.append(delta)
        if len(self._explanation_deltas) > self.max_history:
            self._explanation_deltas = self._explanation_deltas[-self.max_history:]

    # ─── Readers (all strictly temporal — no future leakage) ──────────────

    def get_recent_transitions(
        self,
        before_time: str | pd.Timestamp,
        n: int = 5,
    ) -> List[Dict[str, Any]]:
        """Return N most recent state transitions before `before_time`."""
        t_cutoff = pd.to_datetime(before_time)
        result = [
            asdict(t) for t in self._transitions
            if pd.to_datetime(t.timestamp) < t_cutoff
        ]
        return result[-n:]

    def get_recent_risk_timeline(
        self,
        before_time: str | pd.Timestamp,
        n: int = 20,
    ) -> List[Dict[str, Any]]:
        """Return N most recent risk timeline entries before `before_time`."""
        t_cutoff = pd.to_datetime(before_time)
        result = [
            r for r in self._risk_timeline
            if pd.to_datetime(r["timestamp"]) < t_cutoff
        ]
        return result[-n:]

    def get_past_episodes(
        self,
        before_time: str | pd.Timestamp,
    ) -> List[Dict[str, Any]]:
        """Return all episodes that started before `before_time`."""
        t_cutoff = pd.to_datetime(before_time)
        return [
            asdict(ep) for ep in self._episodes
            if pd.to_datetime(ep.start_time) < t_cutoff
        ]

    def get_last_explanation_delta(
        self,
        before_time: str | pd.Timestamp,
    ) -> Optional[Dict[str, Any]]:
        """Return most recent explanation delta before `before_time`."""
        t_cutoff = pd.to_datetime(before_time)
        recent = [
            d for d in self._explanation_deltas
            if pd.to_datetime(d.to_timestamp) < t_cutoff
        ]
        return asdict(recent[-1]) if recent else None

    def get_last_state(
        self,
        before_time: str | pd.Timestamp,
    ) -> Optional[Dict[str, Any]]:
        """Return the last risk timeline entry before `before_time`."""
        timeline = self.get_recent_risk_timeline(before_time, n=1)
        return timeline[0] if timeline else None

    def compute_explanation_delta(
        self,
        prev_state_features: Optional[Dict[str, float]],
        curr_state_features: Dict[str, float],
        prev_shap: Optional[Dict[str, float]],
        curr_shap: Dict[str, float],
        from_timestamp: str,
        to_timestamp: str,
        old_risk: float,
        new_risk: float,
        old_alert: str,
        new_alert: str,
    ) -> ExplanationDelta:
        """
        Compute and record what changed between two consecutive states.

        Uses actual feature value changes and SHAP changes to explain transitions.
        Language is deliberately statistical — we use "associated with", not "caused".
        """
        feature_changes = []

        if prev_state_features is not None:
            # Find features with largest SHAP delta
            all_features = set(curr_shap.keys()) | set(prev_shap.keys() if prev_shap else [])
            deltas = []
            for feat in all_features:
                prev_s = prev_shap.get(feat, 0.0) if prev_shap else 0.0
                curr_s = curr_shap.get(feat, 0.0)
                shap_delta = curr_s - prev_s
                prev_v = prev_state_features.get(feat, float("nan"))
                curr_v = curr_state_features.get(feat, float("nan"))
                if abs(shap_delta) > 0.01:
                    deltas.append({
                        "feature": feat,
                        "old_value": round(float(prev_v), 3) if not np.isnan(prev_v) else None,
                        "new_value": round(float(curr_v), 3) if not np.isnan(curr_v) else None,
                        "shap_delta": round(float(shap_delta), 4),
                        "direction": "increased_risk" if shap_delta > 0 else "decreased_risk",
                    })
            # Sort by largest absolute SHAP change
            deltas.sort(key=lambda x: abs(x["shap_delta"]), reverse=True)
            feature_changes = deltas[:5]

        dominant = feature_changes[0]["feature"] if feature_changes else "unknown"
        dominant_direction = feature_changes[0]["direction"] if feature_changes else "unknown"
        alert_change = f"{old_alert} → {new_alert}"

        reason = (
            f"Risk {old_risk:.0f}→{new_risk:.0f}: "
            f"'{dominant}' was most associated with {dominant_direction} ({alert_change})"
            if feature_changes else
            f"Risk changed from {old_risk:.0f} to {new_risk:.0f}"
        )

        delta = ExplanationDelta(
            from_timestamp=from_timestamp,
            to_timestamp=to_timestamp,
            risk_delta=round(new_risk - old_risk, 1),
            alert_change=alert_change,
            feature_changes=feature_changes,
            dominant_change_reason=reason,
        )
        self.record_explanation_delta(delta)
        return delta

    def get_summary(self) -> Dict[str, Any]:
        """Return summary statistics of patient's stored memory."""
        return {
            "patient_id": self.patient_id,
            "n_state_snapshots": len(self._risk_timeline),
            "n_transitions": len(self._transitions),
            "n_episodes": len(self._episodes),
            "n_explanation_deltas": len(self._explanation_deltas),
        }

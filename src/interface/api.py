r"""
GlucoTwin — Frontend Integration & Dashboard API (v6)
src/interface/api.py

Clean, type-safe API for Aishni's frontend dashboard.
Provides endpoints for:
1. Patient Twin live snapshot (full TwinState v4)
2. Interactive custom What-If computational simulations
3. Glucose trend history for chart panels
4. Patient risk summary
5. Twin replay for historical validation/demo
6. Memory and transition history
7. OOD / twin confidence status

SCIENTIFIC LANGUAGE POLICY:
All descriptions use "simulated", "estimated", "model-derived" — NOT clinical prescription.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

from src.twin.engine import GlucoTwinEngine
from src.twin.state import TwinState, AlertLevel
from src.simulation.simulator import MetabolicSimulator
from src.models.classifier import HypoClassifier
from src.data.loader import COLUMN_MAP


class GlucoTwinAPI:
    """
    Public API interface for GlucoTwin dashboard integration.

    All public methods return plain Python dicts that are JSON-serializable.
    No pandas DataFrames or numpy types are returned.
    """

    def __init__(self, engine: Optional[GlucoTwinEngine] = None):
        self.engine = engine
        self.simulator = MetabolicSimulator()

    def get_twin_snapshot(
        self,
        current_data: pd.DataFrame,
        current_idx: int = -1,
    ) -> Dict[str, Any]:
        """
        Get complete twin state payload for dashboard visualization.

        Returns a full TwinState dict (v4) including:
        - Risk score and alert level
        - Multi-horizon predictions
        - SHAP explainability factors
        - What-if computational simulations
        - Historical episode matches
        - OOD / twin confidence
        - State transition and explanation delta
        - Historical memory summary
        """
        if self.engine is None:
            raise RuntimeError("Engine not initialized in API.")
        state = self.engine.evaluate_state(current_data, current_idx=current_idx)
        return state.to_dict()

    def get_glucose_trend_history(
        self,
        df: pd.DataFrame,
        last_n_hours: float = 3.0,
        ts_col: str = "timestamp",
        glucose_col: str = "glucose_mgdl",
    ) -> List[Dict[str, Any]]:
        """
        Return the last N hours of glucose readings for chart rendering.

        Returns:
            List of {timestamp, glucose_mgdl, is_hypo, is_near_hypo} dicts
        """
        if ts_col not in df.columns or glucose_col not in df.columns:
            return []

        df_copy = df.copy()
        df_copy[ts_col] = pd.to_datetime(df_copy[ts_col])
        df_copy = df_copy.sort_values(ts_col)

        cutoff = df_copy[ts_col].max() - pd.Timedelta(hours=last_n_hours)
        df_window = df_copy[df_copy[ts_col] >= cutoff]

        records = []
        for _, row in df_window.iterrows():
            g = float(row[glucose_col]) if pd.notna(row.get(glucose_col)) else None
            records.append({
                "timestamp": str(row[ts_col]),
                "glucose_mgdl": round(g, 1) if g is not None else None,
                "is_hypo": bool(g < 70) if g is not None else False,
                "is_near_hypo": bool(g < 90) if g is not None else False,
            })

        return records

    def run_what_if_simulation(
        self,
        current_glucose: float,
        current_velocity: float = 0.0,
        active_iob: float = 0.0,
        horizon_minutes: int = 120,
        carb_intervention_g: float = 0.0,
        carb_type: str = "fast",
        bolus_intervention_u: float = 0.0,
        exercise_intensity: str = "none",
    ) -> Dict[str, Any]:
        """
        Interactive endpoint for frontend slider/input-driven what-if simulation.

        This is the direct simulation entry point for the dashboard's interactive
        "What if I eat X grams?" scenario panels.

        Returns:
            Simulation result with trajectory, simulated nadir, and threshold-crossing flags.
            Note: All results are model-derived computational scenarios.
        """
        return self.simulator.simulate_trajectory(
            current_glucose=current_glucose,
            current_velocity=current_velocity,
            active_iob=active_iob,
            horizon_minutes=horizon_minutes,
            carb_intervention_g=carb_intervention_g,
            carb_type=carb_type,
            bolus_intervention_u=bolus_intervention_u,
            exercise_intensity=exercise_intensity,
        )

    def get_patient_risk_summary(
        self,
        df: pd.DataFrame,
        lookback_hours: float = 24.0,
        ts_col: str = "timestamp",
        glucose_col: str = "glucose_mgdl",
    ) -> Dict[str, Any]:
        """
        Compute summary statistics for a patient's recent glucose history.
        Returns time-in-range, hypo burden, and CGM statistics.
        """
        if ts_col not in df.columns or glucose_col not in df.columns:
            return {}

        df_copy = df.copy()
        df_copy[ts_col] = pd.to_datetime(df_copy[ts_col])
        cutoff = df_copy[ts_col].max() - pd.Timedelta(hours=lookback_hours)
        df_window = df_copy[df_copy[ts_col] >= cutoff].copy()

        g = pd.to_numeric(df_window[glucose_col], errors="coerce").dropna()

        if len(g) == 0:
            return {"error": "No valid glucose data in lookback window"}

        n = len(g)
        return {
            "lookback_hours": lookback_hours,
            "n_readings": n,
            "mean_glucose_mgdl": round(float(g.mean()), 1),
            "std_glucose_mgdl": round(float(g.std()), 1),
            "min_glucose_mgdl": round(float(g.min()), 1),
            "max_glucose_mgdl": round(float(g.max()), 1),
            "pct_very_low": round(float((g < 54).mean() * 100), 1),
            "pct_low": round(float((g < 70).mean() * 100), 1),
            "pct_near_hypo": round(float((g < 90).mean() * 100), 1),
            "pct_in_range": round(float(((g >= 70) & (g <= 180)).mean() * 100), 1),
            "pct_high": round(float((g > 180).mean() * 100), 1),
            "pct_very_high": round(float((g > 250).mean() * 100), 1),
            "n_hypo_events": int((g < 70).sum()),
            "hypo_burden_minutes": int((g < 70).sum()) * 5,
        }

    def get_twin_memory_summary(self) -> Dict[str, Any]:
        """
        Return the patient twin's memory summary.
        Includes transition count, episode count, and recent risk timeline.
        """
        if self.engine is None:
            raise RuntimeError("Engine not initialized in API.")
        return self.engine.memory.get_summary()

    def get_twin_confidence(
        self,
        current_data: pd.DataFrame,
        current_idx: int = -1,
    ) -> Dict[str, Any]:
        """
        Return the OOD/confidence assessment for the current patient state.

        Returns:
            {confidence_level, confidence_score, out_of_distribution, reasons, components}
        """
        if self.engine is None:
            raise RuntimeError("Engine not initialized in API.")
        from src.features.engineer import build_feature_matrix
        label_col = "label_hypo" if "label_hypo" in current_data.columns else None
        X_all, _ = build_feature_matrix(
            current_data, self.engine.column_map,
            label_col=label_col,
            vectorized=True,
        )
        if len(X_all) == 0:
            return {"error": "insufficient data"}
        row = X_all.iloc[current_idx]
        result = self.engine.ood_detector.assess(row)
        return result.to_dict()

    def get_state_transition_history(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Return recent state transitions from the twin's memory.
        Useful for the 'Risk Timeline' panel showing alert escalations/de-escalations.
        """
        if self.engine is None:
            raise RuntimeError("Engine not initialized in API.")
        ts_now = pd.Timestamp.now().isoformat()
        return self.engine.memory.get_recent_transitions(before_time=ts_now, n=n)

    def export_demo_json(
        self,
        output_path: Optional[str | Path],
        state_dict: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Export sample state to JSON for frontend UI mock/testing.

        When ``output_path`` is None the serialized dict is returned instead of
        writing a file.  This is useful for in-memory tests and direct API calls.
        """
        if output_path is None:
            # Return the JSON-safe dict without touching the filesystem.
            return json.loads(json.dumps(state_dict, default=str))
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(state_dict, f, indent=2, default=str)
        return None

r"""
GlucoTwin — Twin Replay Engine
src/twin/replay.py

Backend-only replay capability: reconstructs the twin's complete state history
for any patient and any time window.

This enables:
- Validation: verify twin correctly predicted historical events
- Debugging: trace why a specific alert was (or wasn't) triggered
- Research: analyze twin behavior across all patients
- Demo: show the twin evolving through a live historical hypoglycemic episode

CRITICAL LEAKAGE GUARANTEE:
At each replay step T, evaluate_state() receives ONLY data up to and including T.
No future rows are ever included in the replay window.
All historical memory at T only contains records from < T.

Architecture:
  replay_episode(patient_df, model, t_start, t_end, step_minutes)
  → List[ReplayFrame]

Each ReplayFrame is a complete TwinState + lead-time metadata.
"""

from __future__ import annotations

import logging
import time as _time
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ReplayFrame:
    """Single frame in a Twin replay sequence."""
    frame_index: int
    timestamp: str
    twin_state: Dict[str, Any]     # Full TwinState.to_dict() at this moment
    is_hypo_actual: bool           # Ground truth: was the patient actually in hypo?
    is_alert_active: bool          # Did the twin issue any alert here?
    lead_time_min: Optional[float] # If alert and hypo followed, minutes of advance warning


@dataclass
class ReplayResult:
    """Complete replay output for a patient episode."""
    patient_id: str
    start_time: str
    end_time: str
    n_frames: int
    frames: List[Dict[str, Any]]
    actual_hypo_events: List[Dict[str, Any]]   # Ground truth hypo periods
    alert_summary: Dict[str, Any]
    lead_time_summary: Dict[str, Any]


def replay_episode(
    patient_df: pd.DataFrame,
    engine,                # GlucoTwinEngine (already fitted)
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    step_minutes: int = 15,
    hypo_threshold_mgdl: float = 70.0,
    ts_col: str = "timestamp",
    glucose_col: str = "glucose_mgdl",
) -> ReplayResult:
    """
    Reconstruct the twin's state history through a patient time window.

    Args:
        patient_df: Full preprocessed patient DataFrame (all time)
        engine: Fitted GlucoTwinEngine for this patient
        start_time: Start of replay window (ISO string or None → use all)
        end_time: End of replay window (ISO string or None → use all)
        step_minutes: Replay time step (default: 15 min = one CGM interval)
        hypo_threshold_mgdl: Hypoglycemia threshold for ground-truth labeling
        ts_col: Timestamp column name
        glucose_col: Glucose column name

    Returns:
        ReplayResult with all frames and summary statistics
    """
    df = patient_df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.sort_values(ts_col).reset_index(drop=True)

    # Determine replay window
    t_min = df[ts_col].min()
    t_max = df[ts_col].max()

    t_start = pd.to_datetime(start_time) if start_time else t_min
    t_end = pd.to_datetime(end_time) if end_time else t_max

    # Require minimum context before replay begins (for rolling features to populate)
    min_context_hours = 2.0
    t_replay_begin = t_start + pd.Timedelta(hours=min_context_hours)
    if t_replay_begin > t_end:
        logger.warning("Replay window too short for context window — adjusting start")
        t_replay_begin = t_start + pd.Timedelta(minutes=30)

    # Build replay timestamps (every step_minutes)
    replay_times = pd.date_range(t_replay_begin, t_end, freq=f"{step_minutes}min")
    if len(replay_times) == 0:
        logger.warning("No replay timestamps generated in window")
        return ReplayResult(
            patient_id=engine.patient_id,
            start_time=str(t_start),
            end_time=str(t_end),
            n_frames=0,
            frames=[],
            actual_hypo_events=[],
            alert_summary={},
            lead_time_summary={},
        )

    frames: List[Dict[str, Any]] = []
    lead_times: List[float] = []
    n_alerts = 0
    n_hypo_actual = 0

    for frame_idx, t_now in enumerate(replay_times):
        # LEAKAGE GUARANTEE: slice DataFrame to only rows at or before t_now
        df_up_to_now = df[df[ts_col] <= t_now].copy()

        if len(df_up_to_now) < 10:
            continue  # Insufficient data for feature computation

        # Evaluate twin at this point
        try:
            state = engine.evaluate_state(df_up_to_now, current_idx=-1)
            state_dict = state.to_dict()
        except Exception as e:
            logger.debug(f"Replay frame {frame_idx} ({t_now}) failed: {e}")
            continue

        # Ground-truth: is the patient actually in hypo RIGHT NOW?
        curr_glucose = df_up_to_now.iloc[-1].get(glucose_col, 999)
        is_hypo_actual = bool(pd.to_numeric(curr_glucose, errors="coerce") < hypo_threshold_mgdl)
        if is_hypo_actual:
            n_hypo_actual += 1

        # Is an alert active?
        alert_level = state_dict.get("alert_level", "NORMAL")
        is_alert_active = alert_level in ("WARNING", "CRITICAL")
        if is_alert_active:
            n_alerts += 1

        # Lead-time: if alert now and hypo occurs in the future, compute advance warning
        lead_time = None
        if is_alert_active and not is_hypo_actual:
            future_mask = (df[ts_col] > t_now) & (df[ts_col] <= t_now + pd.Timedelta(minutes=90))
            future_glucose = pd.to_numeric(df.loc[future_mask, glucose_col], errors="coerce")
            if (future_glucose < hypo_threshold_mgdl).any():
                first_hypo_idx = future_glucose[future_glucose < hypo_threshold_mgdl].index[0]
                t_first_hypo = df.loc[first_hypo_idx, ts_col]
                lt = (t_first_hypo - t_now).total_seconds() / 60.0
                if lt > 0:
                    lead_time = round(float(lt), 1)
                    lead_times.append(lead_time)

        frame = ReplayFrame(
            frame_index=frame_idx,
            timestamp=str(t_now),
            twin_state=state_dict,
            is_hypo_actual=is_hypo_actual,
            is_alert_active=is_alert_active,
            lead_time_min=lead_time,
        )
        frames.append(asdict(frame))

    # Collect actual hypo events (continuous periods)
    glucose_series = pd.to_numeric(df[glucose_col], errors="coerce")
    hypo_mask = (glucose_series < hypo_threshold_mgdl) & df[ts_col].between(t_start, t_end)
    actual_hypo_events = []
    if hypo_mask.any():
        hypo_starts = hypo_mask & ~hypo_mask.shift(1, fill_value=False)
        hypo_ends = ~hypo_mask & hypo_mask.shift(1, fill_value=False)
        for start_idx, end_idx in zip(
            df[hypo_starts].index,
            list(df[hypo_ends].index) + [df.index[-1]],
        ):
            if start_idx <= end_idx:
                ep_g = glucose_series.iloc[start_idx:end_idx + 1].dropna()
                actual_hypo_events.append({
                    "start_time": str(df.loc[start_idx, ts_col]),
                    "end_time": str(df.loc[end_idx, ts_col]),
                    "nadir_mgdl": round(float(ep_g.min()), 1) if len(ep_g) > 0 else None,
                    "duration_min": round((df.loc[end_idx, ts_col] - df.loc[start_idx, ts_col]).total_seconds() / 60, 1),
                })

    # Lead-time summary
    lead_time_summary: Dict[str, Any] = {}
    if lead_times:
        lt_arr = np.array(lead_times)
        lead_time_summary = {
            "n_valid_warnings": len(lead_times),
            "mean_lead_time_min": round(float(lt_arr.mean()), 1),
            "median_lead_time_min": round(float(np.median(lt_arr)), 1),
            "p25_lead_time_min": round(float(np.percentile(lt_arr, 25)), 1),
            "p75_lead_time_min": round(float(np.percentile(lt_arr, 75)), 1),
            "pct_gt_15min": round(float((lt_arr > 15).mean() * 100), 1),
            "pct_gt_30min": round(float((lt_arr > 30).mean() * 100), 1),
        }
    else:
        lead_time_summary = {"n_valid_warnings": 0}

    alert_summary = {
        "n_frames": len(frames),
        "n_alerts": n_alerts,
        "n_hypo_actual": n_hypo_actual,
        "alert_rate_pct": round(n_alerts / len(frames) * 100, 1) if frames else 0.0,
    }

    return ReplayResult(
        patient_id=engine.patient_id,
        start_time=str(t_start),
        end_time=str(t_end),
        n_frames=len(frames),
        frames=frames,
        actual_hypo_events=actual_hypo_events,
        alert_summary=alert_summary,
        lead_time_summary=lead_time_summary,
    )

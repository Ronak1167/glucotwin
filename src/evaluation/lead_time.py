r"""
GlucoTwin — Lead-Time & False-Alarm Analysis
src/evaluation/lead_time.py

For each actual hypoglycemic episode, measures:
- When the first valid warning was issued (lead time)
- How many false alarms preceded this episode
- The daily false-alarm burden

These are critical for clinical viability:
- A system that catches every event but fires constantly is useless
- Lead time must be sufficient for a clinical intervention to work
- We target: >15 min lead time, <3 false alarms/night
"""

from __future__ import annotations

import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def analyze_lead_time(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    timestamps: pd.Series,
    threshold: float = 0.25,
    hypo_threshold_mgdl: float = 70.0,
    prediction_horizon_min: int = 60,
    sample_interval_min: float = 5.0,
) -> Dict[str, Any]:
    """
    Compute lead-time and false-alarm statistics.

    Lead time: for each true positive alert that precedes an actual hypo event,
    how many minutes of advance warning did the system provide?

    Algorithm:
    1. Group consecutive positive labels into "true episodes"
    2. For each episode, find the FIRST alarm issued before episode onset
    3. Lead time = episode_onset - first_alarm_time

    Args:
        y_true: Binary true labels (1=future hypo within horizon)
        y_pred_proba: Model predicted probabilities
        timestamps: Corresponding timestamps
        threshold: Alert decision threshold
        hypo_threshold_mgdl: Not used directly — embedded in labels
        prediction_horizon_min: Horizon used for label generation
        sample_interval_min: Sampling interval in minutes

    Returns:
        Dict with lead_time statistics and false_alarm burden
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_pred_proba) >= threshold).astype(int)
    # Convert to pd.Series regardless of whether it's DatetimeIndex, Series, or list
    ts = pd.Series(pd.to_datetime(timestamps).values).reset_index(drop=True)

    n = len(y_true)
    if n == 0:
        return {"error": "no samples"}

    # ── 1. Identify True Episode Onset Times ─────────────────────────────────
    # An episode starts when y_true transitions 0→1
    episode_onsets = []
    in_episode = False
    for i in range(n):
        if y_true[i] == 1 and not in_episode:
            episode_onsets.append(i)
            in_episode = True
        elif y_true[i] == 0:
            in_episode = False

    # ── 2. For each episode onset, find first alarm before it ─────────────────
    lead_times = []
    detected_episodes = 0
    missed_episodes = 0

    for onset_idx in episode_onsets:
        t_onset = ts.iloc[onset_idx]
        # Look back from onset to find first alarm
        # (Within prediction_horizon_min window before onset)
        lookback_start = t_onset - pd.Timedelta(minutes=prediction_horizon_min)
        lookback_mask = (ts <= t_onset) & (ts >= lookback_start)
        lookback_indices = np.where(lookback_mask)[0]

        first_alarm_idx = None
        for idx in lookback_indices:
            if y_pred[idx] == 1:
                first_alarm_idx = idx
                break  # First alarm before onset

        if first_alarm_idx is not None:
            lt = (t_onset - ts.iloc[first_alarm_idx]).total_seconds() / 60.0
            lead_times.append(float(lt))
            detected_episodes += 1
        else:
            missed_episodes += 1

    # ── 3. False Alarm Analysis ────────────────────────────────────────────────
    # False alarm: y_pred=1 but y_true=0
    fp_mask = (y_pred == 1) & (y_true == 0)
    n_fp = int(fp_mask.sum())

    # Group FPs into "alert bursts" (consecutive FP runs count as 1 alarm event)
    fp_events = 0
    in_fp_run = False
    for i in range(n):
        if fp_mask[i] and not in_fp_run:
            fp_events += 1
            in_fp_run = True
        elif not fp_mask[i]:
            in_fp_run = False

    # Duration of observation
    if len(ts) > 1:
        obs_hours = (ts.max() - ts.min()).total_seconds() / 3600.0
        obs_days = obs_hours / 24.0
        obs_nights = obs_hours / 24.0  # Approx: ~1 night per day
    else:
        obs_days = 1.0
        obs_nights = 1.0

    fp_rate_per_day = fp_events / obs_days if obs_days > 0 else 0.0
    fp_samples_per_day = n_fp / obs_days if obs_days > 0 else 0.0

    # ── 4. Lead-Time Statistics ───────────────────────────────────────────────
    result: Dict[str, Any] = {
        "n_episodes": len(episode_onsets),
        "n_detected": detected_episodes,
        "n_missed": missed_episodes,
        "episode_detection_rate_pct": round(
            detected_episodes / len(episode_onsets) * 100, 1
        ) if episode_onsets else 0.0,
        "n_fp_samples": n_fp,
        "n_fp_events": fp_events,
        "false_alarm_events_per_day": round(fp_rate_per_day, 2),
        "false_alarm_samples_per_day": round(fp_samples_per_day, 2),
    }

    if lead_times:
        lt_arr = np.array(lead_times)
        result.update({
            "mean_lead_time_min": round(float(lt_arr.mean()), 1),
            "median_lead_time_min": round(float(np.median(lt_arr)), 1),
            "std_lead_time_min": round(float(lt_arr.std()), 1),
            "p25_lead_time_min": round(float(np.percentile(lt_arr, 25)), 1),
            "p75_lead_time_min": round(float(np.percentile(lt_arr, 75)), 1),
            "min_lead_time_min": round(float(lt_arr.min()), 1),
            "max_lead_time_min": round(float(lt_arr.max()), 1),
            "pct_gt_15min": round(float((lt_arr > 15).mean() * 100), 1),
            "pct_gt_30min": round(float((lt_arr > 30).mean() * 100), 1),
        })
    else:
        result["lead_time_note"] = "No detected episodes with advance warning"

    return result


def compute_alert_burden_summary(
    per_patient_results: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Aggregate lead-time and false-alarm results across all patients.

    Never hides weak patients. Reports full distribution including worst cases.
    """
    mean_lts, fp_per_day = [], []

    for pid, r in per_patient_results.items():
        if "mean_lead_time_min" in r:
            mean_lts.append(r["mean_lead_time_min"])
        if "false_alarm_events_per_day" in r:
            fp_per_day.append(r["false_alarm_events_per_day"])

    result: Dict[str, Any] = {
        "n_patients": len(per_patient_results),
    }

    if mean_lts:
        lt_arr = np.array(mean_lts)
        result["lead_time"] = {
            "mean_across_patients_min": round(float(lt_arr.mean()), 1),
            "median_across_patients_min": round(float(np.median(lt_arr)), 1),
            "min_patient_min": round(float(lt_arr.min()), 1),
            "max_patient_min": round(float(lt_arr.max()), 1),
        }

    if fp_per_day:
        fp_arr = np.array(fp_per_day)
        result["false_alarm_burden"] = {
            "mean_fp_events_per_day": round(float(fp_arr.mean()), 2),
            "median_fp_events_per_day": round(float(np.median(fp_arr)), 2),
            "max_patient_fp_per_day": round(float(fp_arr.max()), 2),
        }

    return result

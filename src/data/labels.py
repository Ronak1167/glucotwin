"""
GlucoTwin — Label Generation
src/data/labels.py

Generates hypoglycemia prediction labels for the ML task.

Task: Binary classification
At time T, predict whether a hypoglycemic event (glucose < threshold) 
will occur within the next H minutes.

Key principles:
- Features use ONLY information available at or before T
- Target looks STRICTLY FORWARD from T
- Episodes are grouped to avoid treating every measurement as independent
- Leakage is explicitly prevented and documented

Label methodology follows:
ADA (American Diabetes Association) thresholds:
  Level 1 hypoglycemia: < 70 mg/dL  (3.9 mmol/L)
  Level 2 hypoglycemia: < 54 mg/dL  (3.0 mmol/L)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def identify_hypoglycemic_episodes(
    glucose: pd.Series,
    timestamps: pd.Series,
    threshold_mgdl: float = 70.0,
    min_duration_min: float = 15.0,
    merge_gap_min: float = 30.0,
) -> pd.DataFrame:
    """
    Identify continuous hypoglycemic episodes in a glucose time series.
    
    An episode is a contiguous period where glucose < threshold_mgdl,
    lasting at least min_duration_min minutes.
    
    Adjacent episodes within merge_gap_min of each other are merged into
    one continuous episode (prevents counting recovery spikes as separate events).
    
    Args:
        glucose: Glucose values (mg/dL)
        timestamps: Corresponding timestamps (pandas datetime)
        threshold_mgdl: Hypoglycemia threshold
        min_duration_min: Minimum episode duration to count
        merge_gap_min: Merge adjacent episodes within this gap
    
    Returns:
        DataFrame with columns: episode_id, start_time, end_time, 
                                 min_glucose, duration_min, n_readings
    """
    ts = pd.to_datetime(timestamps)
    df = pd.DataFrame({"glucose": glucose.values, "timestamp": ts.values})
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Mark readings below threshold
    df["hypo"] = df["glucose"] < threshold_mgdl
    
    # Find episode boundaries (transitions)
    df["episode_start"] = df["hypo"] & (~df["hypo"].shift(1, fill_value=False))
    df["episode_end"] = (~df["hypo"]) & (df["hypo"].shift(1, fill_value=False))
    
    # Extract raw episodes
    raw_episodes = []
    current_start = None
    
    for i, row in df.iterrows():
        if row["episode_start"]:
            current_start = row["timestamp"]
        if row["episode_end"] and current_start is not None:
            raw_episodes.append({
                "start_time": current_start,
                "end_time": row["timestamp"],  # First non-hypo point
            })
            current_start = None
    
    # Handle episode running to end of data
    if current_start is not None and df["hypo"].iloc[-1]:
        raw_episodes.append({
            "start_time": current_start,
            "end_time": df["timestamp"].iloc[-1],
        })
    
    if not raw_episodes:
        return pd.DataFrame(columns=[
            "episode_id", "start_time", "end_time", "min_glucose", "duration_min", "n_readings"
        ])
    
    # Merge episodes within merge_gap_min
    merged = []
    current = raw_episodes[0].copy()
    
    for ep in raw_episodes[1:]:
        gap_min = (ep["start_time"] - current["end_time"]).total_seconds() / 60
        if gap_min <= merge_gap_min:
            # Merge: extend current episode
            current["end_time"] = ep["end_time"]
        else:
            merged.append(current)
            current = ep.copy()
    merged.append(current)
    
    # Filter by minimum duration and compute statistics
    valid_episodes = []
    episode_id = 0
    
    for ep in merged:
        duration_min = (ep["end_time"] - ep["start_time"]).total_seconds() / 60
        
        if duration_min < min_duration_min:
            continue
        
        # Find glucose readings within this episode
        mask = (df["timestamp"] >= ep["start_time"]) & (df["timestamp"] <= ep["end_time"])
        ep_glucose = df.loc[mask, "glucose"].dropna()
        
        valid_episodes.append({
            "episode_id": episode_id,
            "start_time": ep["start_time"],
            "end_time": ep["end_time"],
            "min_glucose": float(ep_glucose.min()) if len(ep_glucose) > 0 else float("nan"),
            "duration_min": round(duration_min, 1),
            "n_readings": int(mask.sum()),
        })
        episode_id += 1
    
    if not valid_episodes:
        return pd.DataFrame(columns=[
            "episode_id", "start_time", "end_time", "min_glucose", "duration_min", "n_readings"
        ])
    
    return pd.DataFrame(valid_episodes)


def generate_prediction_labels(
    df: pd.DataFrame,
    glucose_col: str,
    ts_col: str,
    prediction_horizon_min: int = 60,
    threshold_mgdl: float = 70.0,
    min_event_duration_min: float = 15.0,
    merge_gap_min: float = 30.0,
    label_col: str = "label_hypo",
) -> pd.DataFrame:
    """
    Generate binary hypoglycemia prediction labels.
    
    For each timestamp T in the DataFrame:
        label = 1 if glucose < threshold_mgdl at ANY point in (T, T + horizon]
        label = 0 otherwise
    
    CRITICAL: This function ONLY uses future glucose for the label, never for features.
    The caller is responsible for ensuring features only use data up to T.
    
    Args:
        df: Patient DataFrame with glucose and timestamps
        glucose_col: Name of glucose column
        ts_col: Name of timestamp column
        prediction_horizon_min: Prediction horizon in minutes
        threshold_mgdl: Hypoglycemia threshold (mg/dL)
        min_event_duration_min: Minimum event duration to count (prevents noise)
        merge_gap_min: Gap for merging adjacent episodes
        label_col: Name of output label column
    
    Returns:
        DataFrame with label_col added (1 = future hypo, 0 = no future hypo)
        Rows where labeling is impossible (end of data) are DROPPED.
    """
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.sort_values(ts_col).reset_index(drop=True)
    
    glucose_vals = pd.to_numeric(df[glucose_col], errors="coerce").to_numpy(dtype=float)
    t_arr = pd.to_datetime(df[ts_col]).to_numpy(dtype="datetime64[ns]")
    
    n_total = len(df)
    labels = np.full(n_total, np.nan)
    
    if n_total > 0:
        from collections import deque
        t_end = t_arr + np.timedelta64(prediction_horizon_min, "m")
        j_arr = np.searchsorted(t_arr, t_end, side="right")
        
        q = deque()
        r = 1
        for l in range(n_total):
            target_r = j_arr[l]
            while r < target_r:
                val = glucose_vals[r]
                if not np.isnan(val):
                    while q and q[-1][1] >= val:
                        q.pop()
                    q.append((r, val))
                r += 1
            while q and q[0][0] <= l:
                q.popleft()
            if q:
                labels[l] = 1.0 if q[0][1] < threshold_mgdl else 0.0
    
    df[label_col] = labels
    
    # Drop rows where label could not be determined
    n_unlabeled = df[label_col].isna().sum()
    if n_unlabeled > 0:
        logger.debug(
            f"Dropping {n_unlabeled} rows at end of series where future window is unavailable"
        )
        df = df.dropna(subset=[label_col]).copy()
    
    df[label_col] = df[label_col].astype(int)
    
    n_positive = int(df[label_col].sum())
    n_total_labeled = len(df)
    pos_rate = n_positive / n_total_labeled * 100 if n_total_labeled > 0 else 0
    
    logger.info(
        f"Labels generated: {n_total_labeled} samples | "
        f"Positive (hypo): {n_positive} ({pos_rate:.1f}%) | "
        f"Horizon: {prediction_horizon_min} min | Threshold: {threshold_mgdl} mg/dL"
    )
    
    return df


def generate_nocturnal_labels(
    df: pd.DataFrame,
    glucose_col: str,
    ts_col: str,
    sleep_col: Optional[str],
    prediction_horizon_min: int = 60,
    threshold_mgdl: float = 70.0,
    nocturnal_start_hour: int = 22,
    nocturnal_end_hour: int = 7,
    label_col: str = "label_nocturnal_hypo",
) -> pd.DataFrame:
    """
    Generate nocturnal hypoglycemia prediction labels.
    
    A label is nocturnal if it occurs during sleep/nighttime hours.
    
    If sleep_col is available and contains valid sleep stage/state information,
    it is used to identify sleep periods. Otherwise, falls back to time-of-day
    heuristic (nocturnal_start_hour to nocturnal_end_hour).
    
    Args:
        df: Patient DataFrame
        glucose_col: Glucose column name
        ts_col: Timestamp column name
        sleep_col: Sleep state column (None if unavailable)
        prediction_horizon_min: Prediction horizon in minutes
        threshold_mgdl: Hypoglycemia threshold
        nocturnal_start_hour: Hour to begin nocturnal period (default 22:00)
        nocturnal_end_hour: Hour to end nocturnal period (default 07:00)
        label_col: Output label column name
    
    Returns:
        DataFrame with nocturnal label added.
        Non-nocturnal rows are excluded from labeling (label = NaN and dropped).
    """
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col])
    
    # Determine nocturnal flag
    if sleep_col is not None and sleep_col in df.columns:
        # Use sleep data directly
        sleep_vals = df[sleep_col].fillna("").astype(str).str.lower()
        # Common sleep state indicators in Fitbit data
        is_sleeping = sleep_vals.isin([
            "1", "asleep", "sleep", "light", "deep", "rem",
            "restless", "awake_during_sleep", "true"
        ]) | (pd.to_numeric(df[sleep_col], errors="coerce") > 0)
        df["is_nocturnal"] = is_sleeping
        n_nocturnal = is_sleeping.sum()
        source = "sleep_column"
    else:
        # Fallback: time-of-day heuristic
        hour = df[ts_col].dt.hour
        if nocturnal_start_hour > nocturnal_end_hour:
            # Wraps midnight: 22:00 - 07:00
            is_nocturnal = (hour >= nocturnal_start_hour) | (hour < nocturnal_end_hour)
        else:
            is_nocturnal = (hour >= nocturnal_start_hour) & (hour < nocturnal_end_hour)
        df["is_nocturnal"] = is_nocturnal
        n_nocturnal = is_nocturnal.sum()
        source = f"time_heuristic ({nocturnal_start_hour}:00-{nocturnal_end_hour}:00)"
    
    logger.info(f"Nocturnal rows: {n_nocturnal} / {len(df)} (source: {source})")
    
    # Only label nocturnal rows
    nocturnal_df = df[df["is_nocturnal"]].copy()
    
    if len(nocturnal_df) == 0:
        logger.warning("No nocturnal rows found — check sleep column or time heuristic")
        df[label_col] = np.nan
        return df
    
    # Generate standard prediction labels on nocturnal subset
    nocturnal_df = generate_prediction_labels(
        nocturnal_df,
        glucose_col=glucose_col,
        ts_col=ts_col,
        prediction_horizon_min=prediction_horizon_min,
        threshold_mgdl=threshold_mgdl,
        label_col=label_col,
    )
    
    # Merge back
    df[label_col] = np.nan
    df.loc[nocturnal_df.index, label_col] = nocturnal_df[label_col].values
    
    return df


def compute_label_statistics(
    df: pd.DataFrame,
    label_col: str = "label_hypo",
    patient_col: Optional[str] = "patient_id",
) -> Dict:
    """
    Compute label distribution statistics for reporting.
    
    Returns:
        Dict with overall and per-patient statistics
    """
    valid = df.dropna(subset=[label_col])
    
    n_total = len(valid)
    n_positive = int(valid[label_col].sum())
    n_negative = n_total - n_positive
    pos_rate = n_positive / n_total * 100 if n_total > 0 else 0
    imbalance_ratio = n_negative / n_positive if n_positive > 0 else float("inf")
    
    stats = {
        "n_total_labeled": n_total,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "positive_rate_pct": round(pos_rate, 2),
        "imbalance_ratio": round(imbalance_ratio, 1),
    }
    
    # Per-patient breakdown
    if patient_col and patient_col in df.columns:
        per_patient = {}
        for pid, group in valid.groupby(patient_col):
            n = len(group)
            pos = int(group[label_col].sum())
            per_patient[str(pid)] = {
                "n_samples": n,
                "n_positive": pos,
                "positive_rate_pct": round(pos / n * 100, 1) if n > 0 else 0,
            }
        stats["per_patient"] = per_patient
    
    logger.info(
        f"Label stats: n={n_total}, positive={n_positive} ({pos_rate:.1f}%), "
        f"imbalance={imbalance_ratio:.1f}:1"
    )
    
    return stats

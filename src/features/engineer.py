"""
GlucoTwin — Feature Engineering
src/features/engineer.py

Computes features from preprocessed patient time series for ML model training.

CRITICAL LEAKAGE PREVENTION:
- All features at time T are computed ONLY from data at or before T
- Lookback windows use ONLY past data (t < T)
- No future glucose, insulin, or meal data is used in features
- Rate-of-change features use only current and past measurements

Feature categories:
1. CGM-based (glucose history, trends, variability)
2. Insulin history (IOB proxy, bolus patterns)
3. Meal history (CoB proxy)
4. Activity (steps, heart rate)
5. Sleep context
6. Time/circadian features
7. Personal baseline deviation
8. Interaction features (e.g., low glucose + recent insulin)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── Default lookback windows ─────────────────────────────────────────────────
# Configurable — see config/model.yaml
DEFAULT_CGM_LOOKBACK_MIN = 120       # 2 hours of CGM history
DEFAULT_INSULIN_LOOKBACK_MIN = 240   # 4 hours (covers insulin action tail)
DEFAULT_MEAL_LOOKBACK_MIN = 180      # 3 hours (covers most carb absorption)
DEFAULT_ACTIVITY_LOOKBACK_MIN = 60   # 1 hour of steps/HR

# Insulin On Board (IOB) biexponential decay parameters (approximate)
# These are proxies — actual IOB depends on insulin type
IOB_PEAK_MIN = 75   # Rapid-acting insulin peak (minutes)
IOB_DIA_MIN = 300   # Duration of insulin action (5 hours)


def _get_window_mask(
    timestamps: pd.Series,
    current_time: pd.Timestamp,
    lookback_minutes: float,
    exclude_current: bool = True,
) -> pd.Series:
    """
    Return boolean mask for the lookback window [T - lookback, T).
    
    Args:
        timestamps: Series of timestamps
        current_time: Reference time T
        lookback_minutes: How many minutes to look back
        exclude_current: Exclude the exact current time (prevent leakage)
    
    Returns:
        Boolean mask
    """
    window_start = current_time - pd.Timedelta(minutes=lookback_minutes)
    if exclude_current:
        return (timestamps >= window_start) & (timestamps < current_time)
    else:
        return (timestamps >= window_start) & (timestamps <= current_time)


def compute_cgm_features(
    glucose: pd.Series,
    timestamps: pd.Series,
    current_time: pd.Timestamp,
    current_glucose: float,
    lookback_minutes: float = DEFAULT_CGM_LOOKBACK_MIN,
) -> Dict[str, float]:
    """
    Compute CGM-based features at time T.
    
    Features (all computed from data BEFORE T):
    - glucose_current: Current glucose (at T)
    - glucose_mean_{N}min: Mean glucose in past N minutes
    - glucose_std_{N}min: Standard deviation
    - glucose_min_{N}min: Minimum
    - glucose_max_{N}min: Maximum
    - glucose_velocity_15min: Rate of change in last 15 min (mg/dL/min)
    - glucose_velocity_30min: Rate of change in last 30 min
    - glucose_accel_30min: Second derivative (acceleration)
    - glucose_trend: Binned trend category (falling fast/slow, stable, rising)
    - n_low_readings_{N}min: Number of readings < 70 mg/dL in past N minutes
    - pct_low_{N}min: Fraction of readings < 70 mg/dL
    - time_below_70_min: Time spent below 70 mg/dL (minutes)
    """
    features: Dict[str, float] = {}
    
    features["glucose_current"] = current_glucose if not np.isnan(current_glucose) else np.nan
    
    # Historical window
    mask = _get_window_mask(timestamps, current_time, lookback_minutes)
    window_glucose = glucose[mask].dropna()
    window_ts = timestamps[mask]
    
    # Statistics
    if len(window_glucose) >= 2:
        features[f"glucose_mean_{int(lookback_minutes)}min"] = float(window_glucose.mean())
        features[f"glucose_std_{int(lookback_minutes)}min"] = float(window_glucose.std())
        features[f"glucose_min_{int(lookback_minutes)}min"] = float(window_glucose.min())
        features[f"glucose_max_{int(lookback_minutes)}min"] = float(window_glucose.max())
        features[f"glucose_range_{int(lookback_minutes)}min"] = float(
            window_glucose.max() - window_glucose.min()
        )
    else:
        for suffix in ["mean", "std", "min", "max", "range"]:
            features[f"glucose_{suffix}_{int(lookback_minutes)}min"] = np.nan
    
    # Velocity (rate of change) — 15 and 30 min
    for window_min in [15, 30]:
        mask_v = _get_window_mask(timestamps, current_time, window_min)
        gluc_v = glucose[mask_v].dropna()
        ts_v = timestamps[mask_v]
        
        if len(gluc_v) >= 2:
            dt_min = (ts_v.max() - ts_v.min()).total_seconds() / 60
            if dt_min > 0:
                dg = gluc_v.iloc[-1] - gluc_v.iloc[0]
                velocity = dg / dt_min  # mg/dL per minute
                features[f"glucose_velocity_{window_min}min"] = float(velocity)
            else:
                features[f"glucose_velocity_{window_min}min"] = 0.0
        else:
            features[f"glucose_velocity_{window_min}min"] = np.nan
    
    # Acceleration (second derivative)
    v15 = features.get("glucose_velocity_15min", np.nan)
    v30 = features.get("glucose_velocity_30min", np.nan)
    if not (np.isnan(v15) or np.isnan(v30)):
        features["glucose_accel_30min"] = float(v15 - v30)
    else:
        features["glucose_accel_30min"] = np.nan
    
    # Hypoglycemia proximity features
    if not np.isnan(current_glucose):
        features["glucose_dist_to_70"] = float(current_glucose - 70.0)
        features["glucose_dist_to_54"] = float(current_glucose - 54.0)
        features["is_near_hypo"] = int(current_glucose < 90)  # Warning zone
        features["is_low"] = int(current_glucose < 70)
        features["is_very_low"] = int(current_glucose < 54)
    else:
        features["glucose_dist_to_70"] = np.nan
        features["glucose_dist_to_54"] = np.nan
        features["is_near_hypo"] = np.nan
        features["is_low"] = np.nan
        features["is_very_low"] = np.nan
    
    # Time spent low in lookback window
    if len(window_glucose) >= 1:
        n_low = int((window_glucose < 70.0).sum())
        features["n_low_readings_window"] = float(n_low)
        features["pct_low_readings_window"] = float(n_low / len(window_glucose))
        
        # Estimate time below threshold (assumes uniform sampling at 15 min)
        features["time_below_70_min"] = float(n_low * 15)
    else:
        features["n_low_readings_window"] = np.nan
        features["pct_low_readings_window"] = np.nan
        features["time_below_70_min"] = np.nan
    
    return features


def iob_proxy(
    insulin_doses: pd.Series,
    dose_times: pd.Series,
    current_time: pd.Timestamp,
    peak_min: float = IOB_PEAK_MIN,
    dia_min: float = IOB_DIA_MIN,
) -> float:
    """
    Compute Insulin On Board (IOB) proxy using a simple biexponential decay model.
    
    This is a PROXY — not a full pharmacokinetic model.
    Real IOB depends on individual insulin sensitivity, injection site, etc.
    
    The model: IOB(t) = dose × (1 - t/DIA) for t < DIA, else 0
    (Linear decay approximation — good enough for ML features)
    
    Args:
        insulin_doses: Insulin dose amounts (units)
        dose_times: Times of doses
        current_time: Time T at which to compute IOB
        peak_min: Time to peak activity (not used in linear model but kept for documentation)
        dia_min: Duration of insulin action (minutes)
    
    Returns:
        Estimated IOB at time T (units)
    """
    total_iob = 0.0
    
    for dose, dose_time in zip(insulin_doses.values, dose_times.values):
        dose = float(dose)
        if np.isnan(dose) or dose <= 0:
            continue
        
        dose_time = pd.Timestamp(dose_time)
        if dose_time >= current_time:
            continue  # Future dose — NOT included (leakage prevention)
        
        elapsed_min = (current_time - dose_time).total_seconds() / 60
        
        if elapsed_min >= dia_min:
            continue  # Insulin has fully cleared
        
        # Linear decay approximation
        remaining_fraction = 1.0 - (elapsed_min / dia_min)
        total_iob += dose * remaining_fraction
    
    return max(0.0, total_iob)


def compute_insulin_features(
    df: pd.DataFrame,
    current_time: pd.Timestamp,
    insulin_basal_col: Optional[str] = "insulin_basal",
    insulin_bolus_col: Optional[str] = "insulin_bolus",
    ts_col: str = "timestamp",
    lookback_minutes: float = DEFAULT_INSULIN_LOOKBACK_MIN,
) -> Dict[str, float]:
    """
    Compute insulin-related features at time T.
    
    Features:
    - iob_proxy: Estimated insulin on board (units)
    - bolus_sum_{N}min: Total bolus insulin in past N minutes
    - n_bolus_{N}min: Number of bolus events
    - last_bolus_dose: Last bolus dose amount
    - last_bolus_min_ago: Minutes since last bolus
    """
    features: Dict[str, float] = {}
    timestamps = pd.to_datetime(df[ts_col])
    
    mask = _get_window_mask(timestamps, current_time, lookback_minutes)
    
    # Bolus features
    if insulin_bolus_col and insulin_bolus_col in df.columns:
        bolus = pd.to_numeric(df.loc[mask, insulin_bolus_col], errors="coerce")
        bolus_times = timestamps[mask]
        
        nonzero_mask = bolus > 0
        
        features[f"bolus_sum_{int(lookback_minutes)}min"] = float(bolus.sum(min_count=1) or 0)
        features[f"n_bolus_{int(lookback_minutes)}min"] = float(nonzero_mask.sum())
        
        # Last bolus
        if nonzero_mask.any():
            last_bolus_idx = bolus_times[nonzero_mask].idxmax()
            features["last_bolus_dose"] = float(bolus[last_bolus_idx])
            features["last_bolus_min_ago"] = float(
                (current_time - bolus_times[last_bolus_idx]).total_seconds() / 60
            )
        else:
            features["last_bolus_dose"] = 0.0
            features["last_bolus_min_ago"] = lookback_minutes  # Max lookback as proxy
        
        # IOB proxy
        if nonzero_mask.any():
            features["iob_proxy"] = iob_proxy(
                bolus[nonzero_mask],
                bolus_times[nonzero_mask],
                current_time,
            )
        else:
            features["iob_proxy"] = 0.0
    else:
        for key in [f"bolus_sum_{int(lookback_minutes)}min", 
                    f"n_bolus_{int(lookback_minutes)}min",
                    "last_bolus_dose", "last_bolus_min_ago", "iob_proxy"]:
            features[key] = np.nan
    
    return features


def compute_meal_features(
    df: pd.DataFrame,
    current_time: pd.Timestamp,
    carbs_col: Optional[str] = "carbs_g",
    ts_col: str = "timestamp",
    lookback_minutes: float = DEFAULT_MEAL_LOOKBACK_MIN,
) -> Dict[str, float]:
    """
    Compute meal/carbohydrate features at time T.
    
    Features:
    - carbs_sum_{N}min: Total carbs in past N minutes (CoB proxy)
    - n_meals_{N}min: Number of meal events
    - last_meal_carbs: Last meal carbohydrate amount
    - last_meal_min_ago: Minutes since last meal
    """
    features: Dict[str, float] = {}
    timestamps = pd.to_datetime(df[ts_col])
    mask = _get_window_mask(timestamps, current_time, lookback_minutes)
    
    if carbs_col and carbs_col in df.columns:
        carbs = pd.to_numeric(df.loc[mask, carbs_col], errors="coerce")
        carbs_times = timestamps[mask]
        nonzero = carbs > 0
        
        features[f"carbs_sum_{int(lookback_minutes)}min"] = float(carbs.sum(min_count=1) or 0)
        features[f"n_meals_{int(lookback_minutes)}min"] = float(nonzero.sum())
        
        if nonzero.any():
            last_idx = carbs_times[nonzero].idxmax()
            features["last_meal_carbs"] = float(carbs[last_idx])
            features["last_meal_min_ago"] = float(
                (current_time - carbs_times[last_idx]).total_seconds() / 60
            )
        else:
            features["last_meal_carbs"] = 0.0
            features["last_meal_min_ago"] = lookback_minutes
    else:
        for key in [f"carbs_sum_{int(lookback_minutes)}min",
                    f"n_meals_{int(lookback_minutes)}min",
                    "last_meal_carbs", "last_meal_min_ago"]:
            features[key] = np.nan
    
    return features


def compute_activity_features(
    df: pd.DataFrame,
    current_time: pd.Timestamp,
    steps_col: Optional[str] = "steps",
    hr_col: Optional[str] = "heart_rate",
    ts_col: str = "timestamp",
    lookback_minutes: float = DEFAULT_ACTIVITY_LOOKBACK_MIN,
) -> Dict[str, float]:
    """
    Compute activity features at time T.
    
    Features:
    - steps_sum_{N}min: Total steps in lookback window
    - hr_mean_{N}min: Mean heart rate
    - hr_max_{N}min: Max heart rate
    """
    features: Dict[str, float] = {}
    timestamps = pd.to_datetime(df[ts_col])
    mask = _get_window_mask(timestamps, current_time, lookback_minutes)
    
    if steps_col and steps_col in df.columns:
        steps = pd.to_numeric(df.loc[mask, steps_col], errors="coerce").dropna()
        features[f"steps_sum_{int(lookback_minutes)}min"] = float(steps.sum()) if len(steps) > 0 else np.nan
    else:
        features[f"steps_sum_{int(lookback_minutes)}min"] = np.nan
    
    if hr_col and hr_col in df.columns:
        hr = pd.to_numeric(df.loc[mask, hr_col], errors="coerce").dropna()
        if len(hr) >= 2:
            features[f"hr_mean_{int(lookback_minutes)}min"] = float(hr.mean())
            features[f"hr_max_{int(lookback_minutes)}min"] = float(hr.max())
            features[f"hr_std_{int(lookback_minutes)}min"] = float(hr.std())
        else:
            for k in ["mean", "max", "std"]:
                features[f"hr_{k}_{int(lookback_minutes)}min"] = np.nan
    else:
        for k in ["mean", "max", "std"]:
            features[f"hr_{k}_{int(lookback_minutes)}min"] = np.nan
    
    return features


def compute_sleep_features(
    df: pd.DataFrame,
    current_time: pd.Timestamp,
    sleep_col: Optional[str] = "sleep_seconds",
    sleep_stage_col: Optional[str] = "sleep_stage",
    ts_col: str = "timestamp",
) -> Dict[str, float]:
    """
    Compute sleep context features at time T.
    
    Uses the most recent sleep data available before T.
    Sleep data is typically low-frequency (1-per-minute or less).
    
    Features (if sleep data available):
    - is_currently_sleeping: 0/1
    - sleep_duration_current_night: Total sleep so far this night (hours)
    - sleep_stage_encoded: Numeric encoding of current sleep stage
    - time_since_sleep_start_min: Minutes into current sleep period
    """
    features: Dict[str, float] = {}
    timestamps = pd.to_datetime(df[ts_col])
    
    # Sleep: look at current state (8-hour window)
    mask_sleep = _get_window_mask(timestamps, current_time, 480)  # 8 hours
    
    if sleep_col and sleep_col in df.columns:
        sleep_vals = pd.to_numeric(df.loc[mask_sleep, sleep_col], errors="coerce")
        if len(sleep_vals.dropna()) > 0:
            features["sleep_sum_8h"] = float(sleep_vals.sum(min_count=1) or 0)
            features["is_currently_sleeping"] = int(sleep_vals.iloc[-1] > 0 
                                                     if not sleep_vals.empty else 0)
        else:
            features["sleep_sum_8h"] = np.nan
            features["is_currently_sleeping"] = np.nan
    else:
        features["sleep_sum_8h"] = np.nan
        features["is_currently_sleeping"] = np.nan
    
    if sleep_stage_col and sleep_stage_col in df.columns:
        # Encode sleep stages as ordinal: 0=awake, 1=light, 2=deep, 3=REM
        stage_map = {"0": 0, "awake": 0, "wake": 0,
                     "1": 1, "light": 1, "restless": 1,
                     "2": 2, "deep": 2, "nrem": 2,
                     "3": 3, "rem": 3}
        
        sleep_stages = df.loc[mask_sleep, sleep_stage_col].astype(str).str.lower()
        encoded = sleep_stages.map(lambda x: stage_map.get(x.strip(), np.nan))
        
        if encoded.dropna().shape[0] > 0:
            features["sleep_stage_current"] = float(encoded.iloc[-1])
        else:
            features["sleep_stage_current"] = np.nan
    else:
        features["sleep_stage_current"] = np.nan
    
    return features


def compute_personal_baseline_features(
    df: pd.DataFrame,
    current_time: pd.Timestamp,
    glucose_col: str = "glucose_mgdl",
    ts_col: str = "timestamp",
    n_time_bins: int = 8,
) -> Dict[str, float]:
    """
    Compute deviation from personal baseline.

    Baseline is computed from historical data BEFORE the current time
    (no leakage), split by time-of-day bin.

    Features:
    - glucose_deviation_from_baseline: Current - personal_tod_mean
    - personal_mean_all_time: Overall patient mean glucose
    - personal_std_all_time: Overall patient std glucose
    - personal_ewm_baseline: Exponentially-weighted expanding mean (alpha=0.05)
    - glucose_deviation_from_ewm: Current - EWM baseline
    - personal_nocturnal_mean: Mean glucose during 23:00-06:00 in history
    - glucose_nocturnal_deviation: Current - personal_nocturnal_mean (if nocturnal now)

    TEMPORAL GUARANTEE:
      All baseline statistics are computed from data strictly before current_time.
      No current or future observation is included in any baseline statistic.
    """
    features: Dict[str, float] = {}

    timestamps = pd.to_datetime(df[ts_col])
    glucose = pd.to_numeric(df[glucose_col], errors="coerce")

    # Historical data (strictly before current time)
    hist_mask = timestamps < current_time
    hist_glucose = glucose[hist_mask].dropna()
    hist_ts = timestamps[hist_mask]

    if len(hist_glucose) < 20:
        # Not enough history for baseline
        for key in [
            "glucose_deviation_from_baseline", "personal_mean_all_time",
            "personal_std_all_time", "personal_ewm_baseline",
            "glucose_deviation_from_ewm", "personal_nocturnal_mean",
            "glucose_nocturnal_deviation", "baseline_bin_mean",
        ]:
            features[key] = np.nan
        return features

    # ── Overall statistics ────────────────────────────────────────────────────
    features["personal_mean_all_time"] = float(hist_glucose.mean())
    features["personal_std_all_time"] = float(hist_glucose.std())

    # ── Get current glucose value (at or just before current_time) ───────────
    at_or_before = glucose[timestamps <= current_time].dropna()
    current_val: float = float(at_or_before.iloc[-1]) if len(at_or_before) > 0 else np.nan

    # ── Time-of-day bin baseline ──────────────────────────────────────────────
    bin_size_hours = 24 / n_time_bins
    current_hour = current_time.hour + current_time.minute / 60
    current_bin = int(current_hour / bin_size_hours)

    hist_bins = ((hist_ts.dt.hour + hist_ts.dt.minute / 60) / bin_size_hours).astype(int)
    same_bin_glucose = hist_glucose[hist_bins == current_bin]

    if len(same_bin_glucose) >= 5:
        bin_mean = float(same_bin_glucose.mean())
        features["baseline_bin_mean"] = bin_mean
        features["glucose_deviation_from_baseline"] = (
            current_val - bin_mean if not np.isnan(current_val) else np.nan
        )
    else:
        features["glucose_deviation_from_baseline"] = np.nan
        features["baseline_bin_mean"] = np.nan

    # ── Exponentially-weighted moving average baseline (strictly backward) ────
    # alpha=0.05: weights decay with ~20-reading (~100 min) half-life
    # Uses expanding window on past data only; result is the EWM value just before T
    ewm_vals = hist_glucose.reset_index(drop=True).ewm(alpha=0.05, adjust=True).mean()
    if len(ewm_vals) > 0:
        ewm_baseline = float(ewm_vals.iloc[-1])
        features["personal_ewm_baseline"] = ewm_baseline
        features["glucose_deviation_from_ewm"] = (
            current_val - ewm_baseline if not np.isnan(current_val) else np.nan
        )
    else:
        features["personal_ewm_baseline"] = np.nan
        features["glucose_deviation_from_ewm"] = np.nan

    # ── Nocturnal baseline (23:00 – 06:00) ───────────────────────────────────
    NOCTURNAL_START = 23
    NOCTURNAL_END = 6
    noc_mask = (hist_ts.dt.hour >= NOCTURNAL_START) | (hist_ts.dt.hour < NOCTURNAL_END)
    noc_glucose = hist_glucose[noc_mask]
    if len(noc_glucose) >= 10:
        noc_mean = float(noc_glucose.mean())
        features["personal_nocturnal_mean"] = noc_mean
        is_now_nocturnal = (current_time.hour >= NOCTURNAL_START) or (current_time.hour < NOCTURNAL_END)
        if is_now_nocturnal and not np.isnan(current_val):
            features["glucose_nocturnal_deviation"] = current_val - noc_mean
        else:
            features["glucose_nocturnal_deviation"] = np.nan
    else:
        features["personal_nocturnal_mean"] = np.nan
        features["glucose_nocturnal_deviation"] = np.nan

    return features


def compute_all_features_for_row(
    df: pd.DataFrame,
    row_idx: int,
    column_map: Dict[str, Optional[str]],
    lookback_cgm: float = DEFAULT_CGM_LOOKBACK_MIN,
    lookback_insulin: float = DEFAULT_INSULIN_LOOKBACK_MIN,
    lookback_meal: float = DEFAULT_MEAL_LOOKBACK_MIN,
    lookback_activity: float = DEFAULT_ACTIVITY_LOOKBACK_MIN,
) -> Dict[str, float]:
    """
    Compute all features for a single row (time point T).
    
    This is the main feature extraction function.
    
    Args:
        df: Full patient DataFrame (sorted by timestamp)
        row_idx: Row index for time T
        column_map: Standard name → actual column name
        lookback_*: Lookback window sizes in minutes
    
    Returns:
        Dict of feature_name → value
    """
    ts_col = "timestamp" if "timestamp" in df.columns else column_map.get("timestamp", "timestamp")
    glucose_col = "glucose_mgdl" if "glucose_mgdl" in df.columns else column_map.get("glucose_mgdl")
    
    if ts_col not in df.columns:
        return {}
    
    timestamps = pd.to_datetime(df[ts_col])
    current_time = timestamps.iloc[row_idx]
    
    features: Dict[str, float] = {}
    
    # ── CGM features ──────────────────────────────────────────────────────────
    if glucose_col and glucose_col in df.columns:
        glucose = pd.to_numeric(df[glucose_col], errors="coerce")
        current_glucose = float(glucose.iloc[row_idx]) if pd.notna(glucose.iloc[row_idx]) else np.nan
        
        cgm_feats = compute_cgm_features(
            glucose=glucose,
            timestamps=timestamps,
            current_time=current_time,
            current_glucose=current_glucose,
            lookback_minutes=lookback_cgm,
        )
        features.update(cgm_feats)
    
    # ── Insulin features ──────────────────────────────────────────────────────
    insulin_feats = compute_insulin_features(
        df=df,
        current_time=current_time,
        insulin_basal_col=column_map.get("insulin_basal") or "insulin_basal",
        insulin_bolus_col=column_map.get("insulin_bolus") or "insulin_bolus",
        ts_col=ts_col,
        lookback_minutes=lookback_insulin,
    )
    features.update(insulin_feats)
    
    # ── Meal features ─────────────────────────────────────────────────────────
    meal_feats = compute_meal_features(
        df=df,
        current_time=current_time,
        carbs_col=column_map.get("carbs_g") or "carbs_g",
        ts_col=ts_col,
        lookback_minutes=lookback_meal,
    )
    features.update(meal_feats)
    
    # ── Activity features ─────────────────────────────────────────────────────
    activity_feats = compute_activity_features(
        df=df,
        current_time=current_time,
        steps_col=column_map.get("steps") or "steps",
        hr_col=column_map.get("heart_rate") or "heart_rate",
        ts_col=ts_col,
        lookback_minutes=lookback_activity,
    )
    features.update(activity_feats)
    
    # ── Sleep features ────────────────────────────────────────────────────────
    sleep_feats = compute_sleep_features(
        df=df,
        current_time=current_time,
        sleep_col=column_map.get("sleep_seconds") or "sleep_seconds",
        sleep_stage_col=column_map.get("sleep_stage") or "sleep_stage",
        ts_col=ts_col,
    )
    features.update(sleep_feats)
    
    # ── Time/circadian features (from preprocessed columns) ───────────────────
    time_cols = [
        "hour_of_day", "hour_sin", "hour_cos", "day_of_week",
        "is_weekend", "time_in_day_fraction", "is_nocturnal_heuristic",
    ]
    for col in time_cols:
        if col in df.columns:
            val = df[col].iloc[row_idx]
            features[col] = float(val) if pd.notna(val) else np.nan
    
    return features


def build_feature_matrix_vectorized(
    df: pd.DataFrame,
    column_map: Dict[str, Optional[str]],
    label_col: str = "label_hypo",
    lookback_cgm: float = DEFAULT_CGM_LOOKBACK_MIN,
    lookback_insulin: float = DEFAULT_INSULIN_LOOKBACK_MIN,
    lookback_meal: float = DEFAULT_MEAL_LOOKBACK_MIN,
    lookback_activity: float = DEFAULT_ACTIVITY_LOOKBACK_MIN,
    include_personal_baseline: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    High-speed vectorized feature engineering for a single patient time series.
    Guarantees ZERO data leakage: all rolling features use closed='left' (strictly past data).
    """
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found. Run label generation first.")
    
    ts_col = "timestamp" if "timestamp" in df.columns else column_map.get("timestamp", "timestamp")
    glucose_col = "glucose_mgdl" if "glucose_mgdl" in df.columns else column_map.get("glucose_mgdl", "glucose")
    
    df_sorted = df.sort_values(ts_col).reset_index(drop=True)
    df_idx = df_sorted.set_index(ts_col).sort_index()

    labeled_mask = df_sorted[label_col].notna()
    if not labeled_mask.any():
        return pd.DataFrame(), pd.Series(dtype=float)

    X = pd.DataFrame(index=df_sorted.index)
    g = pd.to_numeric(df_idx[glucose_col], errors="coerce")
    
    # CGM current & backwards rolling statistics (strictly past: closed='left')
    X["glucose_current"] = g.values
    X["glucose_mean_120min"] = g.rolling("120min", closed="left").mean().values
    X["glucose_std_120min"] = g.rolling("120min", closed="left").std().values
    X["glucose_min_120min"] = g.rolling("120min", closed="left").min().values
    X["glucose_max_120min"] = g.rolling("120min", closed="left").max().values
    X["glucose_range_120min"] = X["glucose_max_120min"] - X["glucose_min_120min"]
    
    X["glucose_mean_60min"] = g.rolling("60min", closed="left").mean().values
    X["glucose_std_60min"] = g.rolling("60min", closed="left").std().values
    
    # Velocity (3 readings = 15 min, 6 readings = 30 min at 5-min nominal)
    X["glucose_velocity_15min"] = (g - g.shift(3)).values / 15.0
    X["glucose_velocity_30min"] = (g - g.shift(6)).values / 30.0
    X["glucose_accel_30min"] = X["glucose_velocity_15min"] - X["glucose_velocity_30min"]
    
    # Hypo distance & alert states
    X["glucose_dist_to_70"] = X["glucose_current"] - 70.0
    X["glucose_dist_to_54"] = X["glucose_current"] - 54.0
    X["is_near_hypo"] = (X["glucose_current"] < 90).astype(int)
    X["is_low"] = (X["glucose_current"] < 70).astype(int)
    X["is_very_low"] = (X["glucose_current"] < 54).astype(int)
    
    low_readings = (g < 70).astype(float)
    X["n_low_readings_window"] = low_readings.rolling("120min", closed="left").sum().values
    X["time_below_70_min"] = X["n_low_readings_window"] * 5.0
    
    # Insulin features
    bolus_col = column_map.get("insulin_bolus") or "insulin_bolus"
    if bolus_col in df_idx.columns:
        bolus = pd.to_numeric(df_idx[bolus_col], errors="coerce").fillna(0)
        X["bolus_sum_240min"] = bolus.rolling("240min", closed="left").sum().values
        X["bolus_sum_120min"] = bolus.rolling("120min", closed="left").sum().values
        X["n_bolus_240min"] = (bolus > 0).astype(float).rolling("240min", closed="left").sum().values
    else:
        X["bolus_sum_240min"] = 0.0
        X["bolus_sum_120min"] = 0.0
        X["n_bolus_240min"] = 0.0
        
    basal_col = column_map.get("insulin_basal") or "insulin_basal"
    if basal_col in df_idx.columns:
        X["basal_rate_current"] = pd.to_numeric(df_idx[basal_col], errors="coerce").fillna(0).values
    else:
        X["basal_rate_current"] = 0.0
        
    # Meal / Carbs
    carbs_col = column_map.get("carbs_g") or "carbs_g"
    if carbs_col in df_idx.columns:
        carbs = pd.to_numeric(df_idx[carbs_col], errors="coerce").fillna(0)
        X["carbs_sum_180min"] = carbs.rolling("180min", closed="left").sum().values
        X["carbs_sum_60min"] = carbs.rolling("60min", closed="left").sum().values
        X["n_meals_180min"] = (carbs > 0).astype(float).rolling("180min", closed="left").sum().values
    else:
        X["carbs_sum_180min"] = 0.0
        X["carbs_sum_60min"] = 0.0
        X["n_meals_180min"] = 0.0
        
    # Activity
    steps_col = column_map.get("steps") or "steps"
    if steps_col in df_idx.columns:
        steps = pd.to_numeric(df_idx[steps_col], errors="coerce").fillna(0)
        X["steps_sum_60min"] = steps.rolling("60min", closed="left").sum().values
    else:
        X["steps_sum_60min"] = 0.0
        
    hr_col = column_map.get("heart_rate") or "heart_rate"
    if hr_col in df_idx.columns:
        hr = pd.to_numeric(df_idx[hr_col], errors="coerce")
        X["hr_mean_60min"] = hr.rolling("60min", closed="left").mean().values
        X["hr_max_60min"] = hr.rolling("60min", closed="left").max().values
    else:
        X["hr_mean_60min"] = np.nan
        X["hr_max_60min"] = np.nan
        
    # Time / Circadian
    time_cols = [
        "hour_of_day", "hour_sin", "hour_cos", "day_of_week",
        "is_weekend", "time_in_day_fraction", "is_nocturnal_heuristic",
    ]
    for col in time_cols:
        if col in df_sorted.columns:
            X[col] = df_sorted[col].values
            
    # Interactions & Risk Proxies
    X["low_and_high_bolus"] = ((X["glucose_current"] < 100) & (X["bolus_sum_240min"] > 0.5)).astype(int)
    X["glucose_dropping_fast"] = (X["glucose_velocity_15min"] < -1.0).astype(int)
    if "is_nocturnal_heuristic" in X.columns:
        X["nocturnal_low"] = X["is_nocturnal_heuristic"] * X["is_low"]
    else:
        X["nocturnal_low"] = 0
        
    # ── Personal baseline features (strictly backward-looking) ─────────────────
    if include_personal_baseline:
        # Expanding mean: uses all past readings via shift(1) — no current-value leakage
        expanding_mean = g.shift(1).expanding(min_periods=20).mean().values
        X["personal_expanding_mean"] = expanding_mean
        X["glucose_deviation_from_baseline"] = X["glucose_current"] - expanding_mean

        # Expanding standard deviation (patient variability baseline)
        expanding_std = g.shift(1).expanding(min_periods=20).std().values
        X["personal_expanding_std"] = expanding_std

        # EWM baseline (alpha=0.05, ~100-minute half-life); shift(1) ensures no current leakage
        ewm_baseline = g.shift(1).ewm(alpha=0.05, adjust=True).mean().values
        X["personal_ewm_baseline"] = ewm_baseline
        X["glucose_deviation_from_ewm"] = X["glucose_current"] - ewm_baseline

        # Velocity deviation from personal mean velocity (individual drift tendency)
        expanding_mean_vel = X["glucose_velocity_15min"].shift(1).expanding(min_periods=10).mean().values
        X["personal_mean_velocity"] = expanding_mean_vel
        X["velocity_deviation_from_personal"] = X["glucose_velocity_15min"] - expanding_mean_vel

        # Nocturnal glucose baseline: rolling 7-day nocturnal mean (hour<6 or >=23)
        # Computed vectorized as expanding mean restricted to nocturnal hours, then forward-filled
        if "hour_of_day" in df_sorted.columns:
            is_noc = (
                (df_sorted["hour_of_day"].values >= 23) |
                (df_sorted["hour_of_day"].values < 6)
            )
            g_noc = g.copy()
            g_noc[~is_noc] = np.nan  # mask non-nocturnal periods
            noc_exp_mean = g_noc.shift(1).expanding(min_periods=10).mean().ffill().values
            X["personal_nocturnal_mean"] = noc_exp_mean
            # Nocturnal deviation: only meaningful during nocturnal periods
            noc_dev = np.where(
                is_noc,
                X["glucose_current"].values - noc_exp_mean,
                np.nan,
            )
            X["glucose_nocturnal_deviation"] = noc_dev
        else:
            X["personal_nocturnal_mean"] = np.nan
            X["glucose_nocturnal_deviation"] = np.nan

    # Retain only labeled rows
    X_out = X.loc[labeled_mask].copy()
    y_out = df_sorted.loc[labeled_mask, label_col].copy()

    return X_out, y_out


def build_feature_matrix(
    df: pd.DataFrame,
    column_map: Dict[str, Optional[str]],
    label_col: str = "label_hypo",
    lookback_cgm: float = DEFAULT_CGM_LOOKBACK_MIN,
    lookback_insulin: float = DEFAULT_INSULIN_LOOKBACK_MIN,
    lookback_meal: float = DEFAULT_MEAL_LOOKBACK_MIN,
    lookback_activity: float = DEFAULT_ACTIVITY_LOOKBACK_MIN,
    include_personal_baseline: bool = True,
    vectorized: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build feature matrix X and label vector y for a single patient.
    Uses high-speed vectorized path by default.
    """
    if vectorized:
        return build_feature_matrix_vectorized(
            df=df,
            column_map=column_map,
            label_col=label_col,
            lookback_cgm=lookback_cgm,
            lookback_insulin=lookback_insulin,
            lookback_meal=lookback_meal,
            lookback_activity=lookback_activity,
            include_personal_baseline=include_personal_baseline,
        )
    
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found. Run label generation first.")
    
    ts_col = "timestamp"
    if ts_col not in df.columns:
        raise ValueError("'timestamp' column required. Run preprocessing first.")
    
    labeled_mask = df[label_col].notna()
    labeled_df = df[labeled_mask].copy()
    
    if len(labeled_df) == 0:
        return pd.DataFrame(), pd.Series(dtype=float)
    
    all_features = []
    timestamps = pd.to_datetime(df[ts_col])
    
    for i, (orig_idx, row) in enumerate(labeled_df.iterrows()):
        row_features = compute_all_features_for_row(
            df=df,
            row_idx=orig_idx,
            column_map=column_map,
            lookback_cgm=lookback_cgm,
            lookback_insulin=lookback_insulin,
            lookback_meal=lookback_meal,
            lookback_activity=lookback_activity,
        )
        if include_personal_baseline:
            current_time = timestamps.iloc[orig_idx] if orig_idx < len(timestamps) else pd.Timestamp(row[ts_col])
            baseline_feats = compute_personal_baseline_features(
                df=df,
                current_time=current_time,
                glucose_col=column_map.get("glucose_mgdl") or "glucose_mgdl",
                ts_col=ts_col,
            )
            row_features.update(baseline_feats)
        all_features.append(row_features)
    
    X = pd.DataFrame(all_features, index=labeled_df.index)
    y = labeled_df[label_col]
    return X, y

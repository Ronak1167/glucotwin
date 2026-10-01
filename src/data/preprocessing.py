"""
GlucoTwin — Preprocessing Pipeline
src/data/preprocessing.py

Transforms raw patient DataFrames into clean, aligned, analysis-ready DataFrames.

Principles:
- Every transformation is documented
- Never blindly interpolate large gaps
- Timestamps drive all alignment (no row-index assumptions)
- Patient-aware (each patient is processed independently)
- Deterministic given same config/seed
- Leakage-safe: no future information used in preprocessing
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_preprocessing_config(config_path: str = "config/data.yaml") -> dict:
    """Load preprocessing parameters from config."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def standardize_column_names(
    df: pd.DataFrame,
    column_map: Dict[str, Optional[str]],
) -> pd.DataFrame:
    """
    Rename actual column names to standard names.
    
    After dataset audit, column_map contains the actual column names.
    This function creates standardized column names for downstream processing.
    
    Args:
        df: Raw patient DataFrame
        column_map: Maps standard names to actual column names
                    e.g., {"glucose_mgdl": "Historic Glucose mg/dL", ...}
    
    Returns:
        DataFrame with standardized column names added (originals kept)
    """
    df = df.copy()
    
    for std_name, actual_name in column_map.items():
        if actual_name is None:
            continue
        if actual_name not in df.columns:
            logger.warning(f"Expected column '{actual_name}' not found — skipping '{std_name}'")
            continue
        if actual_name != std_name:
            # Add standardized column (don't drop original for traceability)
            df[std_name] = df[actual_name]
    
    return df


def parse_and_sort_timestamps(
    df: pd.DataFrame,
    ts_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Parse timestamps and sort chronologically.
    
    CRITICAL: All subsequent operations depend on correct chronological ordering.
    
    Args:
        df: DataFrame with timestamp column
        ts_col: Name of timestamp column
    
    Returns:
        Sorted DataFrame with parsed timestamps
    """
    df = df.copy()
    
    if ts_col not in df.columns:
        raise ValueError(f"Timestamp column '{ts_col}' not found. Available: {list(df.columns)}")
    
    # Parse with UTC awareness stripped (we treat all times as local)
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=False)
    
    # Report and drop NaT
    n_nat = df[ts_col].isna().sum()
    if n_nat > 0:
        logger.warning(f"Dropping {n_nat} rows with unparseable timestamps")
        df = df[df[ts_col].notna()].copy()
    
    # Drop duplicate timestamps (keep first occurrence)
    n_dup = df[ts_col].duplicated().sum()
    if n_dup > 0:
        logger.warning(f"Dropping {n_dup} duplicate timestamps (keeping first)")
        df = df.loc[~df[ts_col].duplicated(keep="first")].copy()
    
    df = df.sort_values(ts_col).reset_index(drop=True)
    
    return df


def clip_physiological_bounds(
    df: pd.DataFrame,
    column: str,
    lo: float,
    hi: float,
    set_nan: bool = True,
) -> pd.DataFrame:
    """
    Replace out-of-range values with NaN (or clip to bounds).
    
    For CGM and physiological signals, values outside clinically plausible
    ranges are almost certainly sensor errors or data entry mistakes.
    
    Args:
        df: DataFrame
        column: Column to clean
        lo: Lower bound (inclusive)
        hi: Upper bound (inclusive)
        set_nan: If True, set out-of-range to NaN; otherwise clip to bounds
    
    Returns:
        DataFrame with cleaned column
    """
    if column not in df.columns:
        return df
    
    df = df.copy()
    values = pd.to_numeric(df[column], errors="coerce")
    
    out_of_range = (values < lo) | (values > hi)
    n_out = out_of_range.sum()
    
    if n_out > 0:
        logger.debug(f"Column '{column}': {n_out} values outside [{lo}, {hi}]")
        if set_nan:
            values[out_of_range] = np.nan
        else:
            values = values.clip(lo, hi)
        df[column] = values
    
    return df


def interpolate_cgm_gaps(
    df: pd.DataFrame,
    glucose_col: str = "glucose_mgdl",
    ts_col: str = "timestamp",
    max_gap_minutes: float = 30.0,
    method: str = "linear",
) -> pd.DataFrame:
    """
    Interpolate small CGM gaps using timestamp-aware linear interpolation.
    
    IMPORTANT DESIGN DECISION:
    - Only interpolate gaps <= max_gap_minutes (default 30 min = 2 CGM intervals)
    - Larger gaps are left as NaN (features computed in those windows will be NaN)
    - Never extrapolate beyond the last known value
    - Interpolation is time-aware (uses actual time difference, not row count)
    
    Rationale: FreeStyle Libre 2 samples every 15 min. A single missed reading
    (30-min gap) can be reasonably interpolated. Longer gaps represent genuine
    missing data periods and should not be fabricated.
    
    Args:
        df: Patient DataFrame sorted by timestamp
        glucose_col: Glucose column name
        ts_col: Timestamp column name
        max_gap_minutes: Maximum gap to interpolate (longer gaps = NaN)
        method: Interpolation method ("linear" recommended)
    
    Returns:
        DataFrame with interpolated glucose values (gaps > max_gap marked as NaN)
    """
    if glucose_col not in df.columns or ts_col not in df.columns:
        return df
    
    df = df.copy()
    glucose = pd.to_numeric(df[glucose_col], errors="coerce").copy()
    timestamps = pd.to_datetime(df[ts_col])
    
    # Identify gap sizes at each NaN position
    # For each NaN, find the surrounding valid values and compute gap
    is_nan = glucose.isna()
    
    if is_nan.sum() == 0:
        return df  # Nothing to interpolate
    
    # Vectorized gap detection (O(N) time complexity)
    valid_ts = timestamps.where(~is_nan)
    prev_valid_ts = valid_ts.ffill()
    next_valid_ts = valid_ts.bfill()
    gap_minutes = (next_valid_ts - prev_valid_ts).dt.total_seconds() / 60.0
    
    # Linear interpolation using time index
    ts_series = pd.Series(glucose.values, index=timestamps)
    interpolated = ts_series.interpolate(method="time", limit_direction="forward")
    
    # Mask back large gaps or edge boundaries where gap > max_gap_minutes or nan
    mask_keep_nan = is_nan & ((gap_minutes > max_gap_minutes) | gap_minutes.isna())
    interpolated_vals = interpolated.to_numpy(dtype=float)
    interpolated_vals[mask_keep_nan.to_numpy()] = np.nan
    df[glucose_col] = interpolated_vals
    
    n_filled = int(is_nan.sum() - df[glucose_col].isna().sum())
    n_kept_nan = int(df[glucose_col].isna().sum())
    
    logger.debug(
        f"CGM interpolation: {n_filled} gaps filled, "
        f"{n_kept_nan} large gaps kept as NaN (max_gap={max_gap_minutes} min)"
    )
    
    return df


def resample_to_common_grid(
    df: pd.DataFrame,
    ts_col: str = "timestamp",
    interval_minutes: int = 15,
    method: str = "nearest",
) -> pd.DataFrame:
    """
    Resample DataFrame to a regular time grid.
    
    Used to align multi-signal data (CGM is 15-min, HR is 1-min, etc.)
    to a common sampling grid for feature computation.
    
    Args:
        df: Time-indexed patient DataFrame
        ts_col: Timestamp column name
        interval_minutes: Target grid interval
        method: Resampling method ("nearest", "mean", "sum")
    
    Returns:
        DataFrame resampled to regular grid
    
    Note: For CGM, "nearest" is appropriate (sensor readings are discrete events).
    For heart rate, "mean" captures the average over the interval.
    For steps/calories, "sum" gives the total in the interval.
    """
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.set_index(ts_col).sort_index()
    
    freq = f"{interval_minutes}min"
    
    # Resample non-numeric columns separately
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]
    
    if method == "nearest":
        # For nearest, we resample and forward-fill
        resampled = df[numeric_cols].resample(freq).nearest()
    elif method == "mean":
        resampled = df[numeric_cols].resample(freq).mean()
    elif method == "sum":
        resampled = df[numeric_cols].resample(freq).sum(min_count=1)
    else:
        resampled = df[numeric_cols].resample(freq).first()
    
    resampled = resampled.reset_index()
    resampled.rename(columns={"index": ts_col}, errors="ignore")
    if resampled.index.name == ts_col:
        resampled = resampled.reset_index()
    
    return resampled


def add_time_features(
    df: pd.DataFrame,
    ts_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Add time-based contextual features.
    
    These are NOT future-looking — they only use the current timestamp.
    
    Features added:
    - hour_of_day: 0-23
    - hour_sin, hour_cos: Circular encoding of hour (captures midnight continuity)
    - day_of_week: 0-6 (Monday=0)
    - is_weekend: 0/1
    - time_in_day_fraction: 0.0-1.0
    - is_nocturnal: 1 if hour in [22, 7), 0 otherwise
    """
    if ts_col not in df.columns:
        return df
    
    df = df.copy()
    ts = pd.to_datetime(df[ts_col])
    
    df["hour_of_day"] = ts.dt.hour
    
    # Circular hour encoding (important: 23 and 0 are adjacent)
    hour_rad = 2 * np.pi * ts.dt.hour / 24
    df["hour_sin"] = np.sin(hour_rad)
    df["hour_cos"] = np.cos(hour_rad)
    
    df["day_of_week"] = ts.dt.dayofweek  # 0=Monday
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    df["time_in_day_fraction"] = (ts.dt.hour * 3600 + ts.dt.minute * 60 + ts.dt.second) / 86400
    
    # Nocturnal heuristic: 22:00 to 07:00
    hour = ts.dt.hour
    df["is_nocturnal_heuristic"] = ((hour >= 22) | (hour < 7)).astype(int)
    
    return df


def preprocess_patient(
    df: pd.DataFrame,
    column_map: Dict[str, Optional[str]],
    config: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Full preprocessing pipeline for a single patient.
    
    Steps:
    1. Standardize column names
    2. Parse and sort timestamps
    3. Clip physiological bounds
    4. Interpolate small CGM gaps (time-aware)
    5. Add time features
    
    Does NOT:
    - Add features that require future data
    - Normalize/scale (done in feature pipeline with train-only fit)
    - Generate labels (done separately in labels.py)
    
    Args:
        df: Raw patient DataFrame
        column_map: Standard name → actual column name
        config: Optional data config dict
    
    Returns:
        Preprocessed DataFrame
    """
    from src.data.validation import PHYSIOLOGICAL_BOUNDS
    
    if config is None:
        try:
            config = load_preprocessing_config()
        except Exception:
            config = {}
    
    # ── 1. Standardize column names ───────────────────────────────────────────
    df = standardize_column_names(df, column_map)
    
    # ── 2. Parse and sort timestamps ──────────────────────────────────────────
    ts_col = "timestamp"
    if ts_col in df.columns:
        df = parse_and_sort_timestamps(df, ts_col)
    else:
        logger.error("No 'timestamp' column after standardization — preprocessing aborted")
        return df
    
    # ── 3. Clip physiological bounds ─────────────────────────────────────────
    for signal, (lo, hi) in PHYSIOLOGICAL_BOUNDS.items():
        if signal in df.columns:
            df = clip_physiological_bounds(df, signal, lo, hi, set_nan=True)
    
    # ── 4. Interpolate small CGM gaps ─────────────────────────────────────────
    cgm_config = config.get("missing_data", {})
    max_gap = cgm_config.get("cgm_max_interpolation_gap_minutes", 30.0)
    
    if "glucose_mgdl" in df.columns:
        df = interpolate_cgm_gaps(
            df, 
            glucose_col="glucose_mgdl",
            ts_col=ts_col,
            max_gap_minutes=max_gap,
        )
    
    # ── 5. Add time features ──────────────────────────────────────────────────
    df = add_time_features(df, ts_col)
    
    logger.debug(f"Preprocessing complete: {len(df)} rows × {len(df.columns)} columns")
    
    return df


def preprocess_all_patients(
    patients: Dict[str, pd.DataFrame],
    column_map: Dict[str, Optional[str]],
    config: Optional[dict] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Apply preprocessing to all patients.
    
    Args:
        patients: Dict of patient_id → raw DataFrame
        column_map: Column name mapping
        config: Data configuration
    
    Returns:
        Dict of patient_id → preprocessed DataFrame
    """
    processed = {}
    errors = []
    
    for pid, df in patients.items():
        try:
            df_clean = preprocess_patient(df, column_map, config)
            processed[pid] = df_clean
        except Exception as e:
            logger.error(f"[{pid}] Preprocessing failed: {e}")
            errors.append(pid)
    
    logger.info(f"Preprocessed {len(processed)} patients. Errors: {len(errors)}")
    if errors:
        logger.warning(f"Failed preprocessing: {errors}")
    
    return processed

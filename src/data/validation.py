"""
GlucoTwin — Data Validation
src/data/validation.py

Validates raw and processed patient data for:
- Required signal presence
- Timestamp validity
- Physiological plausibility
- Data completeness
- Duplicate detection

All validation failures are reported, not raised silently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─── Physiological bounds ────────────────────────────────────────────────────
# Clinically plausible ranges. Values outside these are flagged as outliers.
# Sources: ADA guidelines, FreeStyle Libre 2 sensor spec, Fitbit device spec.
PHYSIOLOGICAL_BOUNDS = {
    "glucose_mgdl": (20.0, 600.0),    # Sensor range: 40-400, allow wider for raw
    "heart_rate": (20.0, 250.0),       # RHR typical 40-100; exercise up to 220
    "steps": (0.0, 100_000.0),         # Steps per interval; daily max ~100k
    "calories": (0.0, 10_000.0),       # Daily calories burned (active adult)
    "insulin_basal": (0.0, 50.0),      # Units/hour; typical 0-2 U/hr
    "insulin_bolus": (0.0, 100.0),     # Units; typical 0-20 U per meal
    "carbs_g": (0.0, 500.0),           # Grams per meal; rarely >200g
}

# Minimum CGM readings to consider patient usable
MIN_CGM_READINGS = 200

# Maximum allowable gap in CGM data before flagging
MAX_CGM_GAP_MINUTES = 60


@dataclass
class ValidationResult:
    """Holds validation results for a single patient."""
    patient_id: str
    is_valid: bool = True
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    stats: Dict = field(default_factory=dict)
    
    def add_warning(self, msg: str):
        self.warnings.append(msg)
        logger.warning(f"[{self.patient_id}] WARNING: {msg}")
    
    def add_error(self, msg: str):
        self.errors.append(msg)
        self.is_valid = False
        logger.error(f"[{self.patient_id}] ERROR: {msg}")
    
    def add_stat(self, key: str, value):
        self.stats[key] = value
    
    def summary(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        return (f"[{self.patient_id}] {status} | "
                f"{len(self.errors)} errors | {len(self.warnings)} warnings")


def validate_timestamps(
    df: pd.DataFrame,
    ts_col: str,
    patient_id: str,
    result: ValidationResult,
) -> pd.DataFrame:
    """
    Validate and clean timestamps.
    
    Checks:
    - Column exists
    - Parseable as datetime
    - No NaT values
    - Sorted (after sort)
    - No duplicates
    - Reasonable time range
    
    Returns DataFrame with valid timestamps only.
    """
    if ts_col not in df.columns:
        result.add_error(f"Timestamp column '{ts_col}' not found. Columns: {list(df.columns)}")
        return df
    
    # Parse timestamps
    ts = pd.to_datetime(df[ts_col], errors="coerce")
    n_nat = ts.isna().sum()
    
    if n_nat > 0:
        pct = n_nat / len(ts) * 100
        result.add_warning(f"Timestamp: {n_nat} NaT values ({pct:.1f}%) — dropping")
        df = df[ts.notna()].copy()
        ts = ts.dropna()
    
    result.add_stat("n_valid_timestamps", len(ts))
    
    # Duplicate timestamps
    n_dup = ts.duplicated().sum()
    if n_dup > 0:
        result.add_warning(f"Timestamp: {n_dup} duplicate timestamps — keeping first")
        df = df.loc[~ts.duplicated(keep="first")].copy()
    
    # Sort by timestamp
    df = df.sort_values(ts_col).reset_index(drop=True)
    
    # Reasonable time range
    ts_sorted = pd.to_datetime(df[ts_col])
    ts_min = ts_sorted.min()
    ts_max = ts_sorted.max()
    duration_days = (ts_max - ts_min).total_seconds() / 86400
    
    result.add_stat("start_time", str(ts_min))
    result.add_stat("end_time", str(ts_max))
    result.add_stat("duration_days", round(duration_days, 1))
    
    if duration_days < 7:
        result.add_warning(f"Only {duration_days:.1f} days of data — may be insufficient")
    
    if duration_days < 1:
        result.add_error(f"Less than 1 day of data — patient unusable")
    
    return df


def validate_glucose_signal(
    df: pd.DataFrame,
    glucose_col: str,
    ts_col: str,
    patient_id: str,
    result: ValidationResult,
) -> pd.DataFrame:
    """
    Validate glucose (CGM) signal.
    
    Checks:
    - Column exists
    - Minimum number of readings
    - Physiological bounds
    - Large gaps
    - Sampling interval consistency
    """
    if glucose_col not in df.columns:
        result.add_error(f"Glucose column '{glucose_col}' not found")
        return df
    
    glucose = pd.to_numeric(df[glucose_col], errors="coerce")
    n_total = len(glucose)
    n_valid = glucose.notna().sum()
    n_missing = n_total - n_valid
    missing_pct = n_missing / n_total * 100
    
    result.add_stat("n_cgm_readings_total", n_total)
    result.add_stat("n_cgm_readings_valid", int(n_valid))
    result.add_stat("cgm_missing_pct", round(missing_pct, 1))
    
    if n_valid < MIN_CGM_READINGS:
        result.add_error(
            f"Only {n_valid} valid CGM readings (minimum: {MIN_CGM_READINGS})"
        )
    
    if missing_pct > 30:
        result.add_warning(f"CGM missing: {missing_pct:.1f}% — high proportion")
    
    # Physiological bounds
    lo, hi = PHYSIOLOGICAL_BOUNDS["glucose_mgdl"]
    n_out_low = (glucose < lo).sum()
    n_out_high = (glucose > hi).sum()
    
    result.add_stat("cgm_below_20", int(n_out_low))
    result.add_stat("cgm_above_600", int(n_out_high))
    
    if n_out_low > 0:
        result.add_warning(f"CGM: {n_out_low} readings below {lo} mg/dL (physiologically impossible)")
    if n_out_high > 0:
        result.add_warning(f"CGM: {n_out_high} readings above {hi} mg/dL (likely sensor error)")
    
    # Glucose statistics (on valid readings)
    valid_glucose = glucose.dropna()
    if len(valid_glucose) > 0:
        result.add_stat("cgm_mean_mgdl", round(float(valid_glucose.mean()), 1))
        result.add_stat("cgm_std_mgdl", round(float(valid_glucose.std()), 1))
        result.add_stat("cgm_min_mgdl", round(float(valid_glucose.min()), 1))
        result.add_stat("cgm_max_mgdl", round(float(valid_glucose.max()), 1))
        result.add_stat("cgm_pct_below_70", round(float((valid_glucose < 70).mean() * 100), 2))
        result.add_stat("cgm_pct_below_54", round(float((valid_glucose < 54).mean() * 100), 2))
    
    # Large gaps (using timestamps)
    if ts_col in df.columns:
        ts = pd.to_datetime(df[ts_col])
        # Only look at rows with valid glucose
        ts_valid = ts[glucose.notna()].sort_values()
        if len(ts_valid) > 1:
            gaps_min = ts_valid.diff().dt.total_seconds().dropna() / 60
            large_gaps = (gaps_min > MAX_CGM_GAP_MINUTES).sum()
            max_gap = gaps_min.max()
            
            result.add_stat("cgm_n_large_gaps", int(large_gaps))
            result.add_stat("cgm_max_gap_min", round(float(max_gap), 1))
            
            if large_gaps > 0:
                result.add_warning(
                    f"CGM: {large_gaps} gap(s) > {MAX_CGM_GAP_MINUTES} min "
                    f"(max gap: {max_gap:.0f} min)"
                )
            
            # Estimate sampling interval
            median_gap = gaps_min.median()
            result.add_stat("cgm_median_interval_min", round(float(median_gap), 1))
    
    return df


def validate_signal(
    df: pd.DataFrame,
    col: str,
    signal_name: str,
    patient_id: str,
    result: ValidationResult,
    bounds: Optional[Tuple[float, float]] = None,
) -> None:
    """Generic signal validation for non-CGM signals."""
    if col not in df.columns:
        result.add_warning(f"Signal '{signal_name}' not found (column '{col}')")
        return
    
    values = pd.to_numeric(df[col], errors="coerce")
    n_valid = values.notna().sum()
    n_total = len(values)
    pct_missing = (n_total - n_valid) / n_total * 100
    
    result.add_stat(f"{signal_name}_n_valid", int(n_valid))
    result.add_stat(f"{signal_name}_pct_missing", round(pct_missing, 1))
    
    if n_valid == 0:
        result.add_warning(f"Signal '{signal_name}': ALL values missing")
        return
    
    if pct_missing > 50:
        result.add_warning(f"Signal '{signal_name}': {pct_missing:.1f}% missing")
    
    # Physiological bounds check
    check_bounds = bounds or PHYSIOLOGICAL_BOUNDS.get(signal_name)
    if check_bounds:
        lo, hi = check_bounds
        valid_vals = values.dropna()
        n_out = ((valid_vals < lo) | (valid_vals > hi)).sum()
        if n_out > 0:
            result.add_warning(
                f"Signal '{signal_name}': {n_out} values outside [{lo}, {hi}]"
            )
        
        result.add_stat(f"{signal_name}_mean", round(float(valid_vals.mean()), 2))
        result.add_stat(f"{signal_name}_std", round(float(valid_vals.std()), 2))


def validate_patient(
    df: pd.DataFrame,
    patient_id: str,
    column_map: Dict[str, Optional[str]],
) -> ValidationResult:
    """
    Run full validation on a single patient's data.
    
    Args:
        df: Patient DataFrame (raw)
        patient_id: Patient identifier
        column_map: Mapping of standard names to actual column names
    
    Returns:
        ValidationResult with all findings
    """
    result = ValidationResult(patient_id=patient_id)
    result.add_stat("patient_id", patient_id)
    result.add_stat("n_rows_raw", len(df))
    result.add_stat("n_columns", len(df.columns))
    result.add_stat("columns", list(df.columns))
    
    # ── Timestamps ────────────────────────────────────────────────────────────
    ts_col = column_map.get("timestamp")
    if ts_col:
        df = validate_timestamps(df, ts_col, patient_id, result)
    else:
        result.add_error("No timestamp column identified")
    
    # ── Glucose (required) ────────────────────────────────────────────────────
    glucose_col = column_map.get("glucose_mgdl")
    if glucose_col:
        df = validate_glucose_signal(df, glucose_col, ts_col or "", patient_id, result)
    else:
        result.add_error("No glucose/CGM column identified — this signal is REQUIRED")
    
    # ── Optional signals ──────────────────────────────────────────────────────
    optional_signals = {
        "insulin_basal": "insulin_basal",
        "insulin_bolus": "insulin_bolus",
        "carbs_g": "carbs_g",
        "steps": "steps",
        "calories": "calories",
        "heart_rate": "heart_rate",
        "sleep_seconds": "sleep",
        "sleep_stage": "sleep_stage",
    }
    
    for std_name, stat_prefix in optional_signals.items():
        col = column_map.get(std_name)
        if col and col in df.columns:
            validate_signal(df, col, stat_prefix, patient_id, result)
    
    # ── Final summary ─────────────────────────────────────────────────────────
    result.add_stat("n_errors", len(result.errors))
    result.add_stat("n_warnings", len(result.warnings))
    
    logger.info(result.summary())
    return result


def validate_all_patients(
    patients: Dict[str, pd.DataFrame],
    column_map: Dict[str, Optional[str]],
) -> Tuple[Dict[str, ValidationResult], List[str], List[str]]:
    """
    Validate all patients and return results.
    
    Returns:
        Tuple of (results_dict, valid_patient_ids, invalid_patient_ids)
    """
    results: Dict[str, ValidationResult] = {}
    valid_ids: List[str] = []
    invalid_ids: List[str] = []
    
    for pid, df in patients.items():
        result = validate_patient(df, pid, column_map)
        results[pid] = result
        
        if result.is_valid:
            valid_ids.append(pid)
        else:
            invalid_ids.append(pid)
    
    logger.info(f"Validation complete: {len(valid_ids)} valid, {len(invalid_ids)} invalid")
    if invalid_ids:
        logger.warning(f"Invalid patients (excluded): {invalid_ids}")
    
    return results, valid_ids, invalid_ids


def summarize_validation(results: Dict[str, ValidationResult]) -> pd.DataFrame:
    """Create a summary DataFrame from validation results."""
    rows = []
    for pid, r in results.items():
        row = {"patient_id": pid, "valid": r.is_valid}
        row.update(r.stats)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    if "patient_id" in df.columns:
        df = df.set_index("patient_id")
    return df

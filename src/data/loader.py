"""
GlucoTwin — Data Loader
src/data/loader.py

Handles loading of HUPA-UCM preprocessed CSV files.
Each patient CSV contains multiple signals in a long-format or wide-format table.
The actual format is determined during dataset audit.

This module is deliberately agnostic about the exact column layout —
actual column names are populated after audit and stored in config/data.yaml.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# ─── Column name constants (verified from HUPA-UCM dataset audit) ───────────
COLUMN_MAP: Dict[str, str] = {
    "timestamp": "time",
    "glucose_mgdl": "glucose",
    "insulin_basal": "basal_rate",
    "insulin_bolus": "bolus_volume_delivered",
    "carbs_g": "carb_input",
    "steps": "steps",
    "calories": "calories",
    "heart_rate": "heart_rate",
    "sleep_stage": None,
    "sleep_seconds": None,
}


def load_config(config_path: str = "config/data.yaml") -> dict:
    """Load data configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_patient_files(data_dir: str | Path) -> List[Path]:
    """
    Discover all patient CSV files in the data directory.
    
    HUPA-UCM preprocessed format uses naming convention:
    HUPA{id}P.csv  e.g., HUPA0001P.csv
    
    Args:
        data_dir: Path to the directory containing patient CSV files
    
    Returns:
        Sorted list of patient file paths
    """
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            "Run: python scripts/download_dataset.py"
        )
    
    # Look recursively for HUPA preprocessed CSV files
    all_files = sorted(data_dir.rglob("Preprocessed/HUPA*P.csv"))
    if not all_files:
        all_files = sorted(data_dir.rglob("HUPA*P.csv"))
    if not all_files:
        all_files = sorted(data_dir.rglob("HUPA*.csv"))
    if not all_files:
        all_files = sorted(data_dir.rglob("patient_*.csv"))
    
    if not all_files:
        raise FileNotFoundError(
            f"No patient CSV files found in {data_dir}\n"
            "Expected files matching pattern: HUPA*.csv\n"
            "Run: python scripts/download_dataset.py"
        )
    
    logger.info(f"Found {len(all_files)} patient files in {data_dir}")
    return all_files


def extract_patient_id(filepath: Path) -> str:
    """
    Extract patient ID from filename.
    
    HUPA-UCM: HUPA0001P.csv → "HUPA0001"
    """
    stem = filepath.stem
    # Remove trailing 'P' if present (preprocessed indicator)
    if stem.endswith("P"):
        return stem[:-1]
    return stem


def load_patient_raw(
    filepath: Path,
    timestamp_col: Optional[str] = None,
    parse_dates: bool = True,
) -> pd.DataFrame:
    """
    Load a single patient CSV file.
    
    Performs minimal parsing — full cleaning is handled in preprocessing.py.
    
    Args:
        filepath: Path to patient CSV file
        timestamp_col: Name of timestamp column (auto-detected if None)
        parse_dates: Whether to attempt timestamp parsing
    
    Returns:
        DataFrame with raw patient data, sorted by timestamp
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file cannot be parsed
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Patient file not found: {filepath}")
    
    patient_id = extract_patient_id(filepath)
    
    try:
        # Auto-detect delimiter (semicolon in HUPA-UCM preprocessed)
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            first_line = f.readline()
        sep = ";" if ";" in first_line else ","
        df = pd.read_csv(filepath, sep=sep, low_memory=False)
        logger.debug(f"[{patient_id}] Loaded {len(df)} rows × {len(df.columns)} columns")
        
        # Auto-detect timestamp column if not provided
        if timestamp_col is None:
            timestamp_col = _detect_timestamp_column(df, patient_id)
        
        if timestamp_col and timestamp_col in df.columns:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce", utc=False)
            df = df.sort_values(timestamp_col).reset_index(drop=True)
        
        # Add patient identifier
        df["patient_id"] = patient_id
        
        return df
    
    except Exception as e:
        raise ValueError(f"Failed to load {filepath}: {e}") from e


def _detect_timestamp_column(df: pd.DataFrame, patient_id: str) -> Optional[str]:
    """
    Auto-detect the timestamp/datetime column.
    
    Tries common naming patterns used in diabetes datasets.
    Updated after actual column inspection in audit.
    """
    # Candidate column names (case-insensitive search)
    candidates = [
        "datetime", "date_time", "timestamp", "time", "date",
        "dateTime", "DateTime", "Timestamp",
        "Date", "Time",
        "LocalDtTm",  # Some CGM exports
        "Device Timestamp",  # FreeStyle Libre format
        "Device Time",
    ]
    
    col_lower_map = {c.lower(): c for c in df.columns}
    
    for cand in candidates:
        if cand.lower() in col_lower_map:
            actual = col_lower_map[cand.lower()]
            logger.debug(f"[{patient_id}] Detected timestamp column: '{actual}'")
            return actual
    
    # Fallback: look for datetime-parseable columns
    for col in df.columns:
        sample = df[col].dropna().head(3)
        if len(sample) > 0:
            try:
                pd.to_datetime(sample, errors="raise")
                logger.info(f"[{patient_id}] Detected timestamp column by parsing: '{col}'")
                return col
            except (ValueError, TypeError):
                continue
    
    logger.warning(f"[{patient_id}] Could not detect timestamp column. "
                   f"Columns: {list(df.columns)}")
    return None


def load_all_patients(
    data_dir: str | Path,
    patient_ids: Optional[List[str]] = None,
    timestamp_col: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load all patient data into a dictionary.
    
    Args:
        data_dir: Directory containing patient CSV files
        patient_ids: Specific patients to load (load all if None)
        timestamp_col: Timestamp column name (auto-detected if None)
    
    Returns:
        Dict mapping patient_id → raw DataFrame
    """
    files = get_patient_files(data_dir)
    
    if patient_ids is not None:
        patient_id_set = set(patient_ids)
        files = [f for f in files if extract_patient_id(f) in patient_id_set]
        logger.info(f"Filtered to {len(files)} patients: {patient_ids}")
    
    patients: Dict[str, pd.DataFrame] = {}
    errors: List[str] = []
    
    for filepath in files:
        pid = extract_patient_id(filepath)
        try:
            df = load_patient_raw(filepath, timestamp_col)
            patients[pid] = df
            logger.debug(f"[{pid}] OK — {len(df)} rows")
        except Exception as e:
            logger.warning(f"[{pid}] Failed to load: {e}")
            errors.append(pid)
    
    logger.info(f"Loaded {len(patients)} patients. Errors: {len(errors)}")
    if errors:
        logger.warning(f"Failed patients: {errors}")
    
    return patients


def detect_column_schema(df: pd.DataFrame, patient_id: str = "?") -> Dict[str, Optional[str]]:
    """
    Inspect a patient DataFrame and attempt to identify standard signal columns.
    
    This is used during the dataset audit to auto-populate COLUMN_MAP.
    Returns a best-guess mapping of standard names to actual column names.
    
    Args:
        df: Patient DataFrame (raw)
        patient_id: For logging
    
    Returns:
        Dict mapping standard signal names to detected column names
    """
    found: Dict[str, Optional[str]] = {k: None for k in COLUMN_MAP}
    col_lower = {c.lower().strip(): c for c in df.columns}
    
    # ── Glucose ───────────────────────────────────────────────────────────────
    glucose_candidates = [
        "historic glucose mg/dl", "glucose mg/dl", "glucose_mgdl", "cgm",
        "glucose", "historic glucose mmol/l", "glucose mmol/l",
        "bgl", "cbg", "blood glucose",
    ]
    for c in glucose_candidates:
        if c in col_lower:
            found["glucose_mgdl"] = col_lower[c]
            break
    
    # ── Insulin ───────────────────────────────────────────────────────────────
    for c_low, actual in col_lower.items():
        if "basal" in c_low:
            found["insulin_basal"] = actual
        elif "bolus" in c_low:
            found["insulin_bolus"] = actual
    
    # ── Meals/Carbs ───────────────────────────────────────────────────────────
    for c_low, actual in col_lower.items():
        if any(kw in c_low for kw in ["carb", "cho", "meal", "food"]):
            found["carbs_g"] = actual
            break
    
    # ── Activity ──────────────────────────────────────────────────────────────
    for c_low, actual in col_lower.items():
        if "step" in c_low:
            found["steps"] = actual
        elif "calori" in c_low:
            found["calories"] = actual
    
    # ── Heart rate ────────────────────────────────────────────────────────────
    for c_low, actual in col_lower.items():
        if any(kw in c_low for kw in ["heart rate", "hr", "heartrate", "pulse"]):
            found["heart_rate"] = actual
            break
    
    # ── Sleep ─────────────────────────────────────────────────────────────────
    for c_low, actual in col_lower.items():
        if "sleep stage" in c_low or "sleep_stage" in c_low:
            found["sleep_stage"] = actual
        elif "sleep" in c_low:
            found["sleep_seconds"] = actual  # May be minutes or seconds

    # ── Timestamp (already handled in load_patient_raw) ───────────────────────
    ts_col = _detect_timestamp_column(df, patient_id)
    found["timestamp"] = ts_col
    
    detected = {k: v for k, v in found.items() if v is not None}
    missing = [k for k, v in found.items() if v is None]
    
    logger.debug(f"[{patient_id}] Detected {len(detected)} columns: {detected}")
    if missing:
        logger.debug(f"[{patient_id}] Not found: {missing}")
    
    return found


def get_data_summary(patients: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Generate a summary DataFrame with per-patient data statistics.
    
    Args:
        patients: Dict of patient_id → DataFrame
    
    Returns:
        Summary DataFrame
    """
    rows = []
    for pid, df in patients.items():
        row = {
            "patient_id": pid,
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "columns": list(df.columns),
        }
        
        # Try to get time range
        ts_col = _detect_timestamp_column(df, pid)
        if ts_col and ts_col in df.columns:
            ts = pd.to_datetime(df[ts_col], errors="coerce")
            ts_valid = ts.dropna()
            if len(ts_valid) > 0:
                row["start_time"] = ts_valid.min()
                row["end_time"] = ts_valid.max()
                duration = (ts_valid.max() - ts_valid.min()).total_seconds() / 86400
                row["duration_days"] = round(duration, 1)
        
        rows.append(row)
    
    return pd.DataFrame(rows).set_index("patient_id")

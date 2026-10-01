r"""
GlucoTwin — Full-Cohort TTT Validation + Error Distribution Analysis
scripts/validate_ttt_full_cohort.py

Phase 2 & 3 from Checkpoint 7:
- Attempt TTT validation across all 25 patients
- Document which patients have valid threshold-crossing pairs
- Compute full error distribution (not just MAE)
- Stratify errors by glucose level, velocity, time-of-day, patient

SCIENTIFIC LANGUAGE:
- "model-derived estimate" — not "prediction" or "diagnosis"
- Bias reported without clinical interpretation
- Exclusion reasons documented honestly
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from src.data.loader import get_patient_files, load_patient_raw, COLUMN_MAP
from src.data.preprocessing import preprocess_patient
from src.data.labels import generate_prediction_labels, identify_hypoglycemic_episodes
from src.features.engineer import build_feature_matrix
from src.twin.engine import _estimate_time_to_threshold

DATA_DIR = Path("data/raw/hupa_ucm")
THRESHOLD_MGDL = 70.0
HORIZON_MIN = 120  # How far ahead we look for actual crossing (wider than prediction window for ground-truth)


def find_ttt_pairs_for_patient(
    df: pd.DataFrame,
    patient_id: str,
    ttt_threshold: float = THRESHOLD_MGDL,
    min_glucose_before: float = 75.0,
    horizon_minutes: int = HORIZON_MIN,
) -> tuple[list[dict], str]:
    """
    For a patient, find moments where:
    1. Glucose is above threshold (glucose >= min_glucose_before, i.e., not already below)
    2. Within horizon_minutes, glucose crosses threshold
    3. We have a valid velocity estimate

    Returns (pairs_list, exclusion_reason_if_empty)
    """
    if "glucose_mgdl" not in df.columns or "timestamp" not in df.columns:
        return [], "missing_columns"

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    glucose = pd.to_numeric(df["glucose_mgdl"], errors="coerce").values
    timestamps = df["timestamp"].values.astype("datetime64[ns]")

    # Check how many readings cross threshold at all
    n_below = int((glucose < ttt_threshold).sum())
    if n_below == 0:
        return [], "no_threshold_crossings_in_record"

    # Build velocity feature: simple 15-min window
    # v_15min[i] = (glucose[i] - glucose[i-3]) / 15 (3 readings × 5 min = 15 min)
    velocity = np.full(len(glucose), np.nan)
    for i in range(3, len(glucose)):
        dt_min = (timestamps[i] - timestamps[i - 3]).astype(float) / 1e9 / 60
        if 10 <= dt_min <= 20 and not np.isnan(glucose[i]) and not np.isnan(glucose[i - 3]):
            velocity[i] = (glucose[i] - glucose[i - 3]) / dt_min

    pairs = []
    n_candidate = 0

    for i in range(len(glucose) - 1):
        g0 = glucose[i]
        v0 = velocity[i]
        t0 = timestamps[i]

        # Must be above threshold with valid velocity
        if np.isnan(g0) or g0 < min_glucose_before or np.isnan(v0):
            continue
        # Must be falling
        if v0 >= 0:
            continue

        n_candidate += 1

        # Find actual first crossing in forward window
        t_horizon = t0 + np.timedelta64(int(horizon_minutes * 60e9), "ns")
        fwd_mask = (timestamps > t0) & (timestamps <= t_horizon) & ~np.isnan(glucose)
        if not fwd_mask.any():
            continue

        fwd_glucose = glucose[fwd_mask]
        fwd_ts = timestamps[fwd_mask]

        # Find first crossing (first reading < threshold)
        below_mask = fwd_glucose < ttt_threshold
        if not below_mask.any():
            continue

        first_cross_idx = int(np.argmax(below_mask))
        t_cross = fwd_ts[first_cross_idx]
        actual_ttt_min = float((t_cross - t0).astype(float) / 1e9 / 60)

        if actual_ttt_min <= 0:
            continue

        # Compute model TTT estimate
        model_ttt = _estimate_time_to_threshold(
            current_glucose=float(g0),
            velocity_mgdl_per_min=float(v0),
            threshold=ttt_threshold,
        )

        if model_ttt is None:
            continue

        # Hour for circadian stratification
        t0_pd = pd.Timestamp(t0)
        hour = t0_pd.hour
        is_nocturnal = int((hour >= 22) or (hour < 7))

        pairs.append({
            "patient_id": patient_id,
            "timestamp": str(t0_pd),
            "glucose_at_query": round(float(g0), 1),
            "velocity_at_query": round(float(v0), 3),
            "model_ttt_min": round(float(model_ttt), 1),
            "actual_ttt_min": round(float(actual_ttt_min), 1),
            "error_min": round(float(model_ttt - actual_ttt_min), 1),  # positive = overestimate
            "abs_error_min": round(abs(model_ttt - actual_ttt_min), 1),
            "hour_of_day": int(hour),
            "is_nocturnal": is_nocturnal,
        })

    if not pairs and n_candidate == 0:
        return [], "no_falling_candidates_above_threshold"

    if not pairs:
        return [], f"candidates_found({n_candidate})_but_no_valid_pairs_in_window"

    return pairs, ""


def compute_error_distribution(errors: np.ndarray) -> dict:
    """Compute full error distribution statistics."""
    if len(errors) == 0:
        return {}
    return {
        "n": int(len(errors)),
        "mean": round(float(errors.mean()), 2),
        "median": round(float(np.median(errors)), 2),
        "std": round(float(errors.std()), 2),
        "p25": round(float(np.percentile(errors, 25)), 2),
        "p75": round(float(np.percentile(errors, 75)), 2),
        "p90": round(float(np.percentile(errors, 90)), 2),
        "min": round(float(errors.min()), 2),
        "max": round(float(errors.max()), 2),
        "within_10min_pct": round(float((np.abs(errors) <= 10).mean() * 100), 1),
        "within_15min_pct": round(float((np.abs(errors) <= 15).mean() * 100), 1),
        "within_30min_pct": round(float((np.abs(errors) <= 30).mean() * 100), 1),
    }


def main():
    files = get_patient_files(DATA_DIR)
    print(f"Processing {len(files)} patients...")

    all_pairs = []
    patient_results = {}
    eligible_patients = 0
    patients_with_pairs = 0
    exclusion_reasons = {}

    for f in files:
        pid = f.stem
        try:
            df_raw = load_patient_raw(f)
            df_clean = preprocess_patient(df_raw, COLUMN_MAP)
            eligible_patients += 1
        except Exception as e:
            exclusion_reasons[pid] = f"preprocessing_error: {e}"
            continue

        pairs, excl = find_ttt_pairs_for_patient(df_clean, pid)

        if not pairs:
            exclusion_reasons[pid] = excl
            patient_results[pid] = {
                "n_pairs": 0,
                "excluded": True,
                "exclusion_reason": excl,
            }
        else:
            patients_with_pairs += 1
            all_pairs.extend(pairs)
            errors = np.array([p["abs_error_min"] for p in pairs])
            signed_errors = np.array([p["error_min"] for p in pairs])
            patient_results[pid] = {
                "n_pairs": len(pairs),
                "excluded": False,
                "abs_error_dist": compute_error_distribution(errors),
                "signed_error_dist": compute_error_distribution(signed_errors),
            }
            print(f"  {pid}: {len(pairs)} pairs | MAE={errors.mean():.1f} min | Bias={signed_errors.mean():.1f} min")

    if not all_pairs:
        print("ERROR: No valid pairs found across all patients.")
        return

    df_pairs = pd.DataFrame(all_pairs)
    abs_err = df_pairs["abs_error_min"].values
    signed_err = df_pairs["error_min"].values

    # Full error distribution
    overall_dist = compute_error_distribution(abs_err)
    signed_dist = compute_error_distribution(signed_err)

    # Stratified analysis: glucose level at query
    strat_glucose = {}
    for label, lo, hi in [("below_85", 70, 85), ("85_to_100", 85, 100), ("above_100", 100, 999)]:
        mask = (df_pairs["glucose_at_query"] >= lo) & (df_pairs["glucose_at_query"] < hi)
        if mask.sum() >= 5:
            strat_glucose[label] = compute_error_distribution(abs_err[mask])

    # Stratified: velocity
    strat_vel = {}
    for label, lo, hi in [("slow_drop", -1.0, 0.0), ("moderate_drop", -2.0, -1.0), ("fast_drop", -999, -2.0)]:
        mask = (df_pairs["velocity_at_query"] >= lo) & (df_pairs["velocity_at_query"] < hi)
        if mask.sum() >= 5:
            strat_vel[label] = compute_error_distribution(abs_err[mask])

    # Stratified: nocturnal
    strat_circ = {}
    for label, val in [("nocturnal", 1), ("daytime", 0)]:
        mask = df_pairs["is_nocturnal"] == val
        if mask.sum() >= 5:
            strat_circ[label] = compute_error_distribution(abs_err[mask])

    # Per-patient errors
    per_patient_error = {}
    for pid in df_pairs["patient_id"].unique():
        p_err = abs_err[df_pairs["patient_id"] == pid]
        p_bias = signed_err[df_pairs["patient_id"] == pid]
        per_patient_error[pid] = {
            "n": int(len(p_err)),
            "mae": round(float(p_err.mean()), 1),
            "bias": round(float(p_bias.mean()), 1),
        }

    result = {
        "validation_type": "TTT Full-Cohort Validation",
        "dataset": "HUPA-UCM",
        "threshold_mgdl": THRESHOLD_MGDL,
        "horizon_used_min": HORIZON_MIN,
        "summary": {
            "total_patients_in_dataset": len(files),
            "eligible_patients_preprocessed": eligible_patients,
            "patients_with_valid_pairs": patients_with_pairs,
            "patients_excluded": eligible_patients - patients_with_pairs,
            "total_matched_pairs": len(all_pairs),
        },
        "scientific_statement": (
            "The model-derived time-to-threshold estimate was compared against "
            "actual threshold-crossing times. A negative signed error indicates "
            "the model estimate occurred earlier than the observed crossing. "
            "No clinical interpretation of this bias is implied."
        ),
        "overall_absolute_error": overall_dist,
        "signed_error_distribution": signed_dist,
        "stratified_by_glucose_level": strat_glucose,
        "stratified_by_velocity": strat_vel,
        "stratified_by_circadian": strat_circ,
        "per_patient_summary": per_patient_error,
        "patient_details": patient_results,
        "exclusion_reasons": exclusion_reasons,
    }

    out_path = Path("reports/ttt_validation_full_cohort.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"TTT VALIDATION RESULTS — FULL COHORT")
    print(f"{'='*60}")
    print(f"Patients evaluated:   {eligible_patients}")
    print(f"Patients with pairs:  {patients_with_pairs} / {eligible_patients}")
    print(f"Total matched pairs:  {len(all_pairs)}")
    print(f"\nAbsolute Error Distribution:")
    for k, v in overall_dist.items():
        print(f"  {k}: {v}")
    print(f"\nSigned Error (model - actual):")
    print(f"  mean bias: {signed_dist.get('mean','N/A')} min")
    print(f"  median bias: {signed_dist.get('median','N/A')} min")
    print(f"\nStratified by velocity:")
    for k, v in strat_vel.items():
        print(f"  {k}: MAE={v.get('mean','N/A')} n={v.get('n','N/A')}")
    print(f"\nStratified by circadian:")
    for k, v in strat_circ.items():
        print(f"  {k}: MAE={v.get('mean','N/A')} n={v.get('n','N/A')}")
    print(f"\nExclusion reasons:")
    for pid, reason in exclusion_reasons.items():
        print(f"  {pid}: {reason}")
    print(f"\nSaved to {out_path}")
    return result


if __name__ == "__main__":
    main()

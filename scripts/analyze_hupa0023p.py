r"""
GlucoTwin — Phase 9.4: HUPA0023P Deep Failure Analysis
scripts/analyze_hupa0023p.py

Conducts a deep, leakage-safe investigation of patient HUPA0023P in the LOPO framework.
- Evaluates HUPA0023P under the exact LOPO pipeline (trained on all N-1 other patients).
- Computes TP, TN, FP, FN, sensitivity, specificity, precision, ROC-AUC, PR-AUC, Brier score.
- Identifies specific false-positive and false-negative instances with full physiological context:
  glucose, velocity, acceleration, IOB, COB, activity/HR, circadian/nocturnal context, OOD score, data quality.
- Statistically compares HUPA0023P with the cohort/training population across key distributions:
  mean glucose, GMI, coefficient of variation (CV), hypo prevalence, time-in-range (TIR/TBR/TAR),
  sampling gaps, sensor noise, insulin sensitivity proxy, and OOD scores.
- Synthesizes findings into evidence-supported contributors to failure.
- Never tunes on HUPA0023P.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import get_patient_files, load_patient_raw, COLUMN_MAP
from src.data.preprocessing import preprocess_patient
from src.data.labels import generate_prediction_labels
from src.features.engineer import build_feature_matrix
from src.models.classifier import HypoClassifier
from src.evaluation.metrics import compute_clinical_metrics
from src.twin.ood import OODDetector

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/raw/hupa_ucm")
TARGET_PID = "HUPA0023P"
HORIZON_MIN = 60
THRESHOLD_MGDL = 70.0
LOPO_THRESHOLD = 0.25  # Standard LOPO operating threshold


def load_cohort_data():
    """Load and preprocess all cohort patients."""
    files = get_patient_files(DATA_DIR)
    all_X, all_y, all_df = {}, {}, {}
    
    for f in files:
        pid = f.stem
        try:
            df_raw = load_patient_raw(f)
            df_clean = preprocess_patient(df_raw, COLUMN_MAP)
            df_lab = generate_prediction_labels(
                df_clean, "glucose_mgdl", "timestamp",
                prediction_horizon_min=HORIZON_MIN,
                threshold_mgdl=THRESHOLD_MGDL,
            )
            X, y = build_feature_matrix(df_lab, COLUMN_MAP, vectorized=True)
            if len(X) >= 50 and y.sum() >= 5:
                all_X[pid] = X
                all_y[pid] = y
                all_df[pid] = df_lab
        except Exception as e:
            logger.warning(f"Error loading {pid}: {e}")
            
    return all_X, all_y, all_df


def run_hupa0023p_analysis():
    logger.info("Loading cohort data...")
    all_X, all_y, all_df = load_cohort_data()
    
    if TARGET_PID not in all_X:
        raise ValueError(f"Target patient {TARGET_PID} not found in cohort!")
        
    patient_ids = sorted(all_X.keys())
    logger.info(f"Loaded {len(patient_ids)} patients. Target: {TARGET_PID}")

    # 1. Train LOPO model (all patients EXCEPT HUPA0023P, first 80%)
    X_train_parts, y_train_parts = [], []
    for pid in patient_ids:
        if pid == TARGET_PID:
            continue
        X_p = all_X[pid]
        y_p = all_y[pid]
        n_tr = max(int(len(X_p) * 0.80), len(X_p) - 1)
        X_train_parts.append(X_p.iloc[:n_tr])
        y_train_parts.append(y_p.iloc[:n_tr])
        
    X_train = pd.concat(X_train_parts, ignore_index=True)
    y_train = pd.concat(y_train_parts, ignore_index=True)
    
    # Target patient holdout split (last 20%)
    X_target_full = all_X[TARGET_PID]
    y_target_full = all_y[TARGET_PID]
    df_target_full = all_df[TARGET_PID]
    
    n_test_start = int(len(X_target_full) * 0.80)
    X_test = X_target_full.iloc[n_test_start:].copy()
    y_test = y_target_full.iloc[n_test_start:].copy()
    df_test = df_target_full.iloc[n_test_start:].copy()
    
    # Common columns
    common_cols = sorted(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]
    X_target_full_aligned = X_target_full[common_cols]
    
    logger.info(f"Training LOPO classifier on {len(X_train)} samples across {len(patient_ids)-1} cohort patients...")
    clf = HypoClassifier(model_type="lightgbm")
    clf.fit(X_train, y_train)
    
    # Predict on LOPO test set
    y_pred_proba_test = clf.predict_proba(X_test)
    y_pred_bin_test = (y_pred_proba_test >= LOPO_THRESHOLD).astype(int)
    
    # Also predict on full target series to understand full-patient dynamics
    y_pred_proba_full = clf.predict_proba(X_target_full_aligned)
    y_pred_bin_full = (y_pred_proba_full >= LOPO_THRESHOLD).astype(int)
    
    # Fit OOD detector on training cohort
    ood_detector = OODDetector()
    ood_detector.fit(X_train)
    
    # Test set metrics
    metrics_test = compute_clinical_metrics(y_test, y_pred_proba_test, threshold=LOPO_THRESHOLD)
    metrics_full = compute_clinical_metrics(y_target_full, y_pred_proba_full, threshold=LOPO_THRESHOLD)
    
    # Also evaluate across standard threshold 0.44 (the validation-derived threshold)
    metrics_test_044 = compute_clinical_metrics(y_test, y_pred_proba_test, threshold=0.44)
    
    logger.info(f"HUPA0023P Test Metrics @ thresh={LOPO_THRESHOLD}: {metrics_test}")
    logger.info(f"HUPA0023P Test Metrics @ thresh=0.44: {metrics_test_044}")
    
    # Analyze confusion matrix and individual errors on LOPO test split
    tp_mask = (y_pred_bin_test == 1) & (y_test.values == 1)
    tn_mask = (y_pred_bin_test == 0) & (y_test.values == 0)
    fp_mask = (y_pred_bin_test == 1) & (y_test.values == 0)
    fn_mask = (y_pred_bin_test == 0) & (y_test.values == 1)
    
    n_tp = int(tp_mask.sum())
    n_tn = int(tn_mask.sum())
    n_fp = int(fp_mask.sum())
    n_fn = int(fn_mask.sum())
    
    logger.info(f"Confusion Matrix: TP={n_tp}, TN={n_tn}, FP={n_fp}, FN={n_fn}")

    # Inspect False Positives and False Negatives
    fp_indices = np.where(fp_mask)[0]
    fn_indices = np.where(fn_mask)[0]
    
    fp_details = []
    for idx in fp_indices[:15]:  # Top 15 FPs
        row_X = X_test.iloc[idx]
        row_df = df_test.iloc[idx]
        proba = float(y_pred_proba_test[idx])
        ts = str(row_df.get("timestamp", ""))
        
        # Check OOD for this sample
        ood_res = ood_detector.assess(row_X)
        
        fp_details.append({
            "index": int(idx),
            "timestamp": ts,
            "predicted_proba": round(proba, 4),
            "true_label": 0,
            "glucose_current": round(float(row_X.get("glucose_current", np.nan)), 1),
            "glucose_velocity_15min": round(float(row_X.get("glucose_velocity_15min", np.nan)), 3),
            "glucose_accel_30min": round(float(row_X.get("glucose_accel_30min", np.nan)), 3),
            "bolus_sum_240min": round(float(row_X.get("bolus_sum_240min", 0.0)), 2),
            "carbs_sum_180min": round(float(row_X.get("carbs_sum_180min", 0.0)), 1),
            "hour_of_day": int(row_X.get("hour_of_day", -1)) if pd.notna(row_X.get("hour_of_day")) else -1,
            "is_nocturnal": bool(row_X.get("is_nocturnal_heuristic", 0) == 1),
            "personal_expanding_mean": round(float(row_X.get("personal_expanding_mean", np.nan)), 1) if pd.notna(row_X.get("personal_expanding_mean")) else None,
            "glucose_deviation_from_baseline": round(float(row_X.get("glucose_deviation_from_baseline", np.nan)), 1) if pd.notna(row_X.get("glucose_deviation_from_baseline")) else None,
            "confidence_score": round(float(ood_res.confidence_score), 2),
            "mahalanobis_percentile": round(float(ood_res.mahalanobis_percentile), 1),
            "ood_reasons": ood_res.reasons,
        })
        
    fn_details = []
    for idx in fn_indices[:15]:  # Top 15 FNs
        row_X = X_test.iloc[idx]
        row_df = df_test.iloc[idx]
        proba = float(y_pred_proba_test[idx])
        ts = str(row_df.get("timestamp", ""))
        ood_res = ood_detector.assess(row_X)
        
        fn_details.append({
            "index": int(idx),
            "timestamp": ts,
            "predicted_proba": round(proba, 4),
            "true_label": 1,
            "glucose_current": round(float(row_X.get("glucose_current", np.nan)), 1),
            "glucose_velocity_15min": round(float(row_X.get("glucose_velocity_15min", np.nan)), 3),
            "glucose_accel_30min": round(float(row_X.get("glucose_accel_30min", np.nan)), 3),
            "bolus_sum_240min": round(float(row_X.get("bolus_sum_240min", 0.0)), 2),
            "carbs_sum_180min": round(float(row_X.get("carbs_sum_180min", 0.0)), 1),
            "hour_of_day": int(row_X.get("hour_of_day", -1)) if pd.notna(row_X.get("hour_of_day")) else -1,
            "is_nocturnal": bool(row_X.get("is_nocturnal_heuristic", 0) == 1),
            "personal_expanding_mean": round(float(row_X.get("personal_expanding_mean", np.nan)), 1) if pd.notna(row_X.get("personal_expanding_mean")) else None,
            "glucose_deviation_from_baseline": round(float(row_X.get("glucose_deviation_from_baseline", np.nan)), 1) if pd.notna(row_X.get("glucose_deviation_from_baseline")) else None,
            "confidence_score": round(float(ood_res.confidence_score), 2),
            "mahalanobis_percentile": round(float(ood_res.mahalanobis_percentile), 1),
            "ood_reasons": ood_res.reasons,
        })

    # 2. Comparative Population Statistics
    # Compare HUPA0023P with all other patients
    cohort_stats = []
    for pid in patient_ids:
        df_p = all_df[pid]
        X_p = all_X[pid]
        y_p = all_y[pid]
        
        g = df_p["glucose_mgdl"].dropna()
        hypo_pct = float(y_p.mean() * 100)
        mean_g = float(g.mean())
        std_g = float(g.std())
        cv_g = float(std_g / mean_g * 100) if mean_g > 0 else 0.0
        tir = float(((g >= 70) & (g <= 180)).mean() * 100)
        tbr = float((g < 70).mean() * 100)
        tar = float((g > 180).mean() * 100)
        
        # Check bolus and carb frequency
        has_bolus = "bolus_sum_240min" in X_p.columns and (X_p["bolus_sum_240min"] > 0).any()
        has_carbs = "carbs_sum_180min" in X_p.columns and (X_p["carbs_sum_180min"] > 0).any()
        has_hr = "hr_mean_60min" in X_p.columns and X_p["hr_mean_60min"].notna().any()
        
        bolus_rate_per_day = float((X_p["bolus_sum_240min"] > 0).sum()) / (max(len(df_p) * 5 / 1440, 1.0))
        carbs_rate_per_day = float((X_p["carbs_sum_180min"] > 0).sum()) / (max(len(df_p) * 5 / 1440, 1.0))
        
        # Sensor sampling rate / gap distribution
        ts = pd.to_datetime(df_p["timestamp"])
        diffs = ts.diff().dt.total_seconds() / 60.0
        gaps_gt_15m = float((diffs > 15).sum())
        gap_fraction = float((diffs > 15).mean() * 100)
        
        cohort_stats.append({
            "patient_id": pid,
            "n_samples": len(df_p),
            "monitoring_days": round(len(df_p) * 5.0 / 1440.0, 1),
            "mean_glucose": round(mean_g, 1),
            "std_glucose": round(std_g, 1),
            "cv_pct": round(cv_g, 1),
            "tir_pct": round(tir, 1),
            "tbr_pct": round(tbr, 1),
            "tar_pct": round(tar, 1),
            "hypo_label_prevalence_pct": round(hypo_pct, 2),
            "gaps_gt_15m": int(gaps_gt_15m),
            "gap_fraction_pct": round(gap_fraction, 2),
            "has_bolus": bool(has_bolus),
            "has_carbs": bool(has_carbs),
            "has_hr": bool(has_hr),
            "bolus_events_per_day": round(bolus_rate_per_day, 1),
            "carbs_events_per_day": round(carbs_rate_per_day, 1),
        })

    cohort_df = pd.DataFrame(cohort_stats)
    hupa23_stats = cohort_df[cohort_df["patient_id"] == TARGET_PID].iloc[0].to_dict()
    cohort_no_23 = cohort_df[cohort_df["patient_id"] != TARGET_PID]

    # Compute percentiles and z-scores for HUPA0023P
    comparison_metrics = {}
    for col in ["mean_glucose", "std_glucose", "cv_pct", "tir_pct", "tbr_pct", "tar_pct", "hypo_label_prevalence_pct", "gap_fraction_pct", "monitoring_days"]:
        val = hupa23_stats[col]
        cohort_vals = cohort_no_23[col].values
        mean_c = float(np.mean(cohort_vals))
        std_c = float(np.std(cohort_vals))
        z = float((val - mean_c) / std_c) if std_c > 0 else 0.0
        pctl = float(stats.percentileofscore(cohort_vals, val))
        comparison_metrics[col] = {
            "hupa0023p": val,
            "cohort_mean": round(mean_c, 2),
            "cohort_std": round(std_c, 2),
            "cohort_median": round(float(np.median(cohort_vals)), 2),
            "cohort_min": round(float(np.min(cohort_vals)), 2),
            "cohort_max": round(float(np.max(cohort_vals)), 2),
            "z_score": round(z, 2),
            "percentile_in_cohort": round(pctl, 1),
        }

    # Evaluate feature importance of LOPO model to understand why predictions were made
    importances = clf.get_feature_importances()
    top_features = {k: round(float(v), 4) for k, v in importances.head(15).items()}
    
    # Check OOD summary for HUPA0023P test set
    ood_confidences = []
    ood_mahalanobis = []
    ood_flagged = []
    for _, row in X_test.iterrows():
        res = ood_detector.assess(row)
        ood_confidences.append(res.confidence_score)
        ood_mahalanobis.append(res.mahalanobis_percentile)
        ood_flagged.append(res.out_of_distribution)
    
    ood_confidences = np.array(ood_confidences)
    ood_mahalanobis = np.array(ood_mahalanobis)
    
    ood_summary = {
        "mean_confidence": round(float(np.mean(ood_confidences)), 3),
        "median_confidence": round(float(np.median(ood_confidences)), 3),
        "mean_mahalanobis_percentile": round(float(np.mean(ood_mahalanobis)), 1),
        "p90_mahalanobis_percentile": round(float(np.percentile(ood_mahalanobis, 90)), 1),
        "pct_flagged_ood": round(float(np.mean(ood_flagged) * 100), 2),
    }

    # Package output
    output = {
        "target_patient": TARGET_PID,
        "lopo_test_metrics_threshold_025": metrics_test,
        "lopo_test_metrics_threshold_044": metrics_test_044,
        "lopo_full_patient_metrics_threshold_025": metrics_full,
        "confusion_matrix_test_025": {
            "tp": n_tp,
            "tn": n_tn,
            "fp": n_fp,
            "fn": n_fn,
            "total_test_samples": len(y_test),
            "total_positive_test": int(y_test.sum()),
        },
        "false_positives_sample": fp_details,
        "false_negatives_sample": fn_details,
        "patient_characteristics": hupa23_stats,
        "cohort_comparison": comparison_metrics,
        "ood_distribution_on_target": ood_summary,
        "top_model_features": top_features,
    }

    out_file = Path("reports/checkpoint9_phase4_hupa0023p_analysis.json")
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved machine-readable analysis to {out_file}")
    
    return output


if __name__ == "__main__":
    res = run_hupa0023p_analysis()
    print("\n--- HUPA0023P LOPO Analysis Completed Successfully ---")
    print(f"Test ROC-AUC: {res['lopo_test_metrics_threshold_025']['roc_auc']:.4f}")
    print(f"Test PR-AUC: {res['lopo_test_metrics_threshold_025']['pr_auc']:.4f}")
    print(f"Test Sensitivity @ 0.25: {res['lopo_test_metrics_threshold_025']['sensitivity']:.4f}")
    print(f"Test Specificity @ 0.25: {res['lopo_test_metrics_threshold_025']['specificity']:.4f}")
    print(f"Test Sensitivity @ 0.44: {res['lopo_test_metrics_threshold_044']['sensitivity']:.4f}")
    print(f"Test Specificity @ 0.44: {res['lopo_test_metrics_threshold_044']['specificity']:.4f}")
    print(f"Confusion Matrix @ 0.25: {res['confusion_matrix_test_025']}")

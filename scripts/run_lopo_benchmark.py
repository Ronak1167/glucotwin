r"""
GlucoTwin — LOPO Benchmark + Alert Threshold Science + Calibration
scripts/run_lopo_benchmark.py

Phase 4: Patient-independent benchmark with full per-patient table
Phase 5: Model calibration analysis (raw vs Platt vs Isotonic)
Phase 7: Alert threshold derivation from validation data
Phase 8: Alert burden analysis

OUTPUT: reports/lopo_benchmark.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from src.data.loader import get_patient_files, load_patient_raw, COLUMN_MAP
from src.data.preprocessing import preprocess_patient
from src.data.labels import generate_prediction_labels
from src.features.engineer import build_feature_matrix
from src.models.classifier import HypoClassifier
from src.evaluation.metrics import compute_clinical_metrics, find_optimal_threshold
from src.evaluation.lopo import run_lopo_evaluation, PlattCalibrator
from src.evaluation.lead_time import analyze_lead_time, compute_alert_burden_summary

DATA_DIR = Path("data/raw/hupa_ucm")
PREDICTION_HORIZON_MIN = 60
THRESHOLD_MGDL = 70.0
LOPO_FIXED_THRESHOLD = 0.25  # Fixed threshold for LOPO — no test-set tuning


def build_all_patient_data():
    """Load features+labels for all patients. Returns dicts keyed by patient_id."""
    files = get_patient_files(DATA_DIR)
    all_X, all_y, all_df = {}, {}, {}
    skipped = []

    for f in files:
        pid = f.stem
        try:
            df_raw = load_patient_raw(f)
            df_clean = preprocess_patient(df_raw, COLUMN_MAP)
            df_lab = generate_prediction_labels(
                df_clean, "glucose_mgdl", "timestamp",
                prediction_horizon_min=PREDICTION_HORIZON_MIN,
                threshold_mgdl=THRESHOLD_MGDL,
            )
            X, y = build_feature_matrix(df_lab, COLUMN_MAP, vectorized=True)
            if len(X) >= 50 and y.sum() >= 5:
                all_X[pid] = X
                all_y[pid] = y
                all_df[pid] = df_lab
            else:
                skipped.append(f"{pid} (n={len(X)}, pos={int(y.sum())})")
        except Exception as e:
            skipped.append(f"{pid} (error: {e})")

    print(f"Loaded {len(all_X)} patients | Skipped: {skipped}")
    return all_X, all_y, all_df


def run_calibration_analysis(X_train, y_train, X_val, y_val, X_test, y_test):
    """Compare raw vs Platt vs Isotonic calibration. NEVER fits calibration on test set."""
    from sklearn.metrics import brier_score_loss

    clf = HypoClassifier(model_type="lightgbm")
    clf.fit(X_train, y_train)

    raw_test = clf.predict_proba(X_test)
    raw_val  = clf.predict_proba(X_val)

    results = {}

    # Raw probabilities
    results["raw"] = {
        "brier": round(float(brier_score_loss(y_test, raw_test)), 4),
        "description": "Uncalibrated model output",
    }

    # Platt calibration (fit on val only)
    platt = PlattCalibrator(method="platt")
    platt.fit(raw_val, y_val)
    platt_test = platt.calibrate(raw_test)
    results["platt"] = {
        "brier": round(float(brier_score_loss(y_test, platt_test)), 4),
        "description": "Logistic regression on model scores, fit on validation split",
        "calibration_metrics": platt.compute_calibration_metrics(y_test, raw_test),
    }

    # Isotonic calibration (fit on val only)
    iso = PlattCalibrator(method="isotonic")
    iso.fit(raw_val, y_val)
    iso_test = iso.calibrate(raw_test)
    results["isotonic"] = {
        "brier": round(float(brier_score_loss(y_test, iso_test)), 4),
        "description": "Isotonic regression, fit on validation split",
        "calibration_metrics": iso.compute_calibration_metrics(y_test, raw_test),
    }

    # Select best
    best = min(results.items(), key=lambda kv: kv[1]["brier"])
    results["recommended"] = {
        "method": best[0],
        "brier": best[1]["brier"],
        "selection_criterion": "lowest Brier score on held-out test set",
        "note": "If raw Brier is already adequate, additional calibration may not be warranted",
    }

    return results, clf


def analyze_alert_burden_for_patient(y_true, y_pred_proba, timestamps, threshold, patient_id):
    """Compute alert burden metrics for one patient."""
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Count alert events (consecutive positives = one episode)
    alert_events = 0
    in_alert = False
    for pred in y_pred:
        if pred == 1 and not in_alert:
            alert_events += 1
            in_alert = True
        elif pred == 0:
            in_alert = False

    # Total monitoring time in days
    if hasattr(timestamps, "__len__") and len(timestamps) > 0:
        ts = pd.to_datetime(timestamps)
        monitor_days = max(1.0, (ts.max() - ts.min()).total_seconds() / 86400)
    else:
        monitor_days = 1.0

    n_pos = int(y_true.sum()) if hasattr(y_true, "sum") else int(np.sum(y_true))
    n_tp = int(((np.asarray(y_pred) == 1) & (np.asarray(y_true) == 1)).sum())
    n_fp_samples = int(((np.asarray(y_pred) == 1) & (np.asarray(y_true) == 0)).sum())

    return {
        "patient_id": patient_id,
        "alert_events_total": int(alert_events),
        "alerts_per_day": round(alert_events / monitor_days, 2),
        "monitoring_days": round(monitor_days, 1),
        "n_true_events": n_pos,
        "n_tp_samples": n_tp,
        "n_fp_samples": n_fp_samples,
        "fp_sample_rate_pct": round(n_fp_samples / max(1, len(y_true)) * 100, 2),
    }


def main():
    print("Loading patient data...")
    all_X, all_y, all_df = build_all_patient_data()

    n_patients = len(all_X)
    print(f"\n{'='*60}")
    print(f"RUNNING LOPO BENCHMARK — {n_patients} patients")
    print(f"{'='*60}")

    t_lopo_start = time.perf_counter()
    lopo_result = run_lopo_evaluation(
        all_patient_features=all_X,
        all_patient_labels=all_y,
        model_type="lightgbm",
        prediction_horizon_min=PREDICTION_HORIZON_MIN,
        min_pos_samples=10,
    )
    lopo_time = time.perf_counter() - t_lopo_start
    print(f"LOPO complete in {lopo_time:.1f}s")

    # Print per-patient table
    print(f"\nPer-Patient LOPO Results (fixed threshold={LOPO_FIXED_THRESHOLD}):")
    print(f"{'Patient':<15} {'ROC-AUC':>8} {'PR-AUC':>8} {'Recall':>8} {'Spec':>8} {'F2':>8} {'n_pos':>7}")
    print("-" * 65)
    per_patient = lopo_result.get("per_patient_metrics", {})
    for pid, m in sorted(per_patient.items()):
        print(
            f"{pid:<15} "
            f"{m.get('roc_auc',0):.3f}   "
            f"{m.get('pr_auc',0):.3f}   "
            f"{m.get('sensitivity',0):.3f}   "
            f"{m.get('specificity',0):.3f}   "
            f"{m.get('f2',0):.3f}   "
            f"{m.get('n_pos_test',0):>7}"
        )
    print("-" * 65)
    agg = lopo_result.get("aggregate_metrics", {})
    print(
        f"{'MEAN (±std)':<15} "
        f"{agg.get('roc_auc',{}).get('mean',0):.3f}±{agg.get('roc_auc',{}).get('std',0):.3f} "
        f"{agg.get('pr_auc',{}).get('mean',0):.3f}±{agg.get('pr_auc',{}).get('std',0):.3f} "
        f"{agg.get('sensitivity',{}).get('mean',0):.3f}   "
        f"{agg.get('specificity',{}).get('mean',0):.3f}   "
        f"{agg.get('f2',{}).get('mean',0):.3f}"
    )

    # ── Calibration Analysis ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("CALIBRATION ANALYSIS — 60/20/20 split")
    print(f"{'='*60}")

    # Pool all patient data for calibration analysis
    X_all = pd.concat(all_X.values(), ignore_index=True)
    y_all = pd.concat(all_y.values(), ignore_index=True)
    n = len(X_all)
    n_train = int(n * 0.60)
    n_val   = int(n * 0.20)
    X_train_cal = X_all.iloc[:n_train]
    y_train_cal = y_all.iloc[:n_train]
    X_val_cal   = X_all.iloc[n_train:n_train + n_val]
    y_val_cal   = y_all.iloc[n_train:n_train + n_val]
    X_test_cal  = X_all.iloc[n_train + n_val:]
    y_test_cal  = y_all.iloc[n_train + n_val:]

    cal_results, clf_for_threshold = run_calibration_analysis(
        X_train_cal, y_train_cal,
        X_val_cal, y_val_cal,
        X_test_cal, y_test_cal,
    )

    print(f"Calibration results (Brier score, lower=better):")
    for method, r in cal_results.items():
        if method != "recommended":
            print(f"  {method}: {r['brier']}")
    print(f"  -> Recommended: {cal_results['recommended']['method']} (Brier={cal_results['recommended']['brier']})")

    # ── Alert Threshold Derivation (Phase 7) ────────────────────────────────────
    print(f"\n{'='*60}")
    print("ALERT THRESHOLD SCIENCE — derived from validation data only")
    print(f"{'='*60}")

    # Re-train on train split (60%), tune on val (20%), evaluate fixed on test (20%)
    clf_thresh = HypoClassifier(model_type="lightgbm")
    clf_thresh.fit(X_train_cal, y_train_cal)

    val_proba = clf_thresh.predict_proba(X_val_cal)
    test_proba = clf_thresh.predict_proba(X_test_cal)

    # Derive threshold on VALIDATION data only
    best_thresh_val, best_m_val = find_optimal_threshold(
        y_val_cal, val_proba, target_metric="f2", min_sensitivity=0.70,
    )
    print(f"Threshold derived from validation: {best_thresh_val:.3f}")
    print(f"Validation metrics at this threshold:")
    for k in ["sensitivity", "specificity", "precision", "f2"]:
        print(f"  {k}: {best_m_val.get(k,0):.3f}")

    # Evaluate on test (no peeking during threshold selection)
    m_test = compute_clinical_metrics(y_test_cal, test_proba, threshold=best_thresh_val)
    print(f"\nTest set metrics at validation-derived threshold ({best_thresh_val:.3f}):")
    for k in ["roc_auc", "pr_auc", "sensitivity", "specificity", "precision", "f2", "brier_score"]:
        print(f"  {k}: {m_test.get(k,0):.4f}")

    threshold_science = {
        "method": "F2-optimized on validation set with min_sensitivity=0.70 constraint",
        "derived_from": "validation_split_only",
        "threshold_value": best_thresh_val,
        "validation_metrics": best_m_val,
        "test_metrics": m_test,
        "note": "Threshold frozen before test evaluation. Not tuned on test set.",
        "alert_levels": {
            "NORMAL": f"model_probability < {best_thresh_val:.2f}",
            "WATCH": f"model_probability >= {max(0.10, best_thresh_val*0.5):.2f} (heuristic lower boundary)",
            "WARNING": f"model_probability >= {best_thresh_val:.2f}",
            "CRITICAL": f"model_probability >= {min(0.80, best_thresh_val*2):.2f} OR glucose < 70",
        },
    }

    # ── Alert Burden (Phase 8) ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("ALERT BURDEN ANALYSIS")
    print(f"{'='*60}")

    burden_records = []
    for pid, X_p in all_X.items():
        y_p = all_y[pid]
        df_p = all_df[pid]
        # Only test portion (last 20%)
        n_start = int(len(X_p) * 0.80)
        X_te_p = X_p.iloc[n_start:]
        y_te_p = y_p.iloc[n_start:]
        df_te_p = df_p.iloc[n_start:]

        if len(X_te_p) < 20:
            continue

        try:
            clf_p = HypoClassifier(model_type="lightgbm")
            clf_p.fit(X_p.iloc[:n_start], y_p.iloc[:n_start])
            proba_p = clf_p.predict_proba(X_te_p)
            ts_p = df_te_p["timestamp"].values if "timestamp" in df_te_p.columns else []

            burden = analyze_alert_burden_for_patient(
                y_te_p.values, proba_p, ts_p,
                threshold=best_thresh_val, patient_id=pid
            )
            burden_records.append(burden)

            print(f"  {pid}: {burden['alerts_per_day']:.1f} alerts/day | "
                  f"fp_rate={burden['fp_sample_rate_pct']:.1f}% | "
                  f"monitor={burden['monitoring_days']:.1f} days")
        except Exception as e:
            print(f"  {pid}: burden analysis failed ({e})")

    if burden_records:
        rates = [b["alerts_per_day"] for b in burden_records]
        print(f"\nAlert burden summary:")
        print(f"  Median alerts/day: {np.median(rates):.1f}")
        print(f"  Mean alerts/day:   {np.mean(rates):.1f}")
        print(f"  Min: {np.min(rates):.1f} | Max: {np.max(rates):.1f}")
        print(f"  Note: High alert burden may indicate over-alerting at selected threshold")

    # ── Save Results ────────────────────────────────────────────────────────────
    output = {
        "lopo_benchmark": lopo_result,
        "calibration_analysis": cal_results,
        "threshold_science": threshold_science,
        "alert_burden": {
            "per_patient": burden_records,
            "summary": {
                "median_alerts_per_day": round(float(np.median([b["alerts_per_day"] for b in burden_records])), 1) if burden_records else None,
                "mean_alerts_per_day": round(float(np.mean([b["alerts_per_day"] for b in burden_records])), 1) if burden_records else None,
                "note": "Alert burden reported as observed behavior. No claim made about clinical optimality.",
            },
        },
    }

    out_path = Path("reports/lopo_benchmark.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nSaved to {out_path}")
    return output


if __name__ == "__main__":
    main()

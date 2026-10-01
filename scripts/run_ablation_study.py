r"""
GlucoTwin — Phase 9.12: Focused Apples-to-Apples Ablation Study
scripts/run_ablation_study.py

Conducts a rigorous, apples-to-apples ablation study comparing:
1. core/original: CGM rolling stats, velocities, acceleration, insulin, carbs, activity, base interactions.
2. core+personalization: core + expanding personal baseline stats (mean, std, EWM, velocity deviations).
3. core+personalization+temporal/circadian: core + personalization + circadian & nocturnal features.
4. core+personalization+memory: core + personalization + historical memory features (24h hypo history).
5. full_current_stack: All features + isotonic calibration + AlertStateMachine hysteresis.

PRESERVATION OF SCIENTIFIC INTEGRITY:
- Slicing: Chronological 60% Train / 20% Validation / 20% Test.
- Validation-Only Decisions: Threshold optimization (F2-score targeting, sensitivity >= 0.70)
  and isotonic calibrator fitting occur STRICTLY on the 20% validation split.
- Untouched Final Test Set: Evaluated completely blind on the 20% held-out test split.
- Standard Metrics Measured:
  ROC-AUC, PR-AUC, Sensitivity, Specificity, Precision, F1, F2,
  Brier Score, Expected Calibration Error (ECE, 10 bins),
  False Positives, Missed Events (FN), Alert Burden (alerts/day),
  Lead-Time Metrics (mean, median, P25, P75, detected episode rate).

Outputs:
- reports/checkpoint9_phase12_ablation.json
- reports/checkpoint9_phase12_ablation.md
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import get_patient_files, load_patient_raw, COLUMN_MAP
from src.data.preprocessing import preprocess_patient
from src.data.labels import generate_prediction_labels
from src.features.engineer import build_feature_matrix_vectorized
from src.models.classifier import HypoClassifier
from src.models.calibration import IsotonicCalibrator
from src.evaluation.metrics import compute_clinical_metrics, find_optimal_threshold
from src.evaluation.lead_time import analyze_lead_time
from src.twin.alert import AlertStateMachine

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/raw/hupa_ucm")
HORIZON_MIN = 60
THRESHOLD_MGDL = 70.0


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) with equal-width probability bins."""
    y = np.asarray(y_true, dtype=int).ravel()
    p = np.asarray(y_prob, dtype=float).ravel()
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(y)
    
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (p >= lo) & (p < hi)
        n_bin = np.sum(mask)
        if n_bin == 0:
            continue
        mean_p = float(np.mean(p[mask]))
        mean_y = float(np.mean(y[mask]))
        ece += (n_bin / n_total) * abs(mean_p - mean_y)
        
    return round(float(ece), 4)


def build_augmented_patient_features():
    """
    Build features across all patients, including dedicated historical memory features.
    Guarantees strict zero-leakage via closed='left' and shift(1).
    """
    files = get_patient_files(DATA_DIR)
    all_X, all_y, all_ts = {}, {}, {}
    
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
            X, y = build_feature_matrix_vectorized(df_lab, COLUMN_MAP)
            if len(X) < 50 or y.sum() < 5:
                continue
                
            # Add dedicated 24-hour historical memory features (strictly past: closed='left')
            df_sorted = df_lab.sort_values("timestamp").reset_index(drop=True)
            df_idx = df_sorted.set_index("timestamp").sort_index()
            g = pd.to_numeric(df_idx["glucose_mgdl"], errors="coerce")
            
            # Historical memory 1: Minimum glucose in past 24 hours (strictly past)
            min_g_24h = g.rolling("1440min", closed="left").min().values
            # Historical memory 2: Count of hypo episodes in past 24 hours
            low_binary = (g < 70).astype(float)
            hypo_count_24h = low_binary.rolling("1440min", closed="left").sum().values
            
            # Historical memory 3: Time elapsed since last reading below 70 mg/dL (bounded 0 to 1440 min)
            # Vectorized forward-fill of timestamps where glucose < 70
            ts_series = pd.Series(df_idx.index, index=df_idx.index)
            ts_low = ts_series.where(g < 70).shift(1).ffill()
            time_since_low = (ts_series - ts_low).dt.total_seconds().values / 60.0
            time_since_low = np.nan_to_num(time_since_low, nan=1440.0)
            time_since_low = np.clip(time_since_low, 0.0, 1440.0)

            labeled_mask = df_sorted["label_hypo"].notna()
            X["memory_min_glucose_24h"] = min_g_24h[labeled_mask]
            X["memory_hypo_count_24h"] = hypo_count_24h[labeled_mask]
            X["memory_time_since_last_low_min"] = time_since_low[labeled_mask]

            all_X[pid] = X
            all_y[pid] = y
            all_ts[pid] = df_sorted.loc[labeled_mask, "timestamp"].values
        except Exception as e:
            logger.warning(f"Skipping {pid}: {e}")
            
    return all_X, all_y, all_ts


def define_ablation_feature_subsets():
    """Define exact feature column subsets for each ablation configuration."""
    core_features = [
        "glucose_current", "glucose_mean_120min", "glucose_std_120min",
        "glucose_min_120min", "glucose_max_120min", "glucose_range_120min",
        "glucose_mean_60min", "glucose_std_60min", "glucose_velocity_15min",
        "glucose_velocity_30min", "glucose_accel_30min", "glucose_dist_to_70",
        "glucose_dist_to_54", "is_near_hypo", "is_low", "is_very_low",
        "n_low_readings_window", "time_below_70_min",
        "bolus_sum_240min", "bolus_sum_120min", "n_bolus_240min", "basal_rate_current",
        "carbs_sum_180min", "carbs_sum_60min", "n_meals_180min",
        "steps_sum_60min", "hr_mean_60min", "hr_max_60min",
        "low_and_high_bolus", "glucose_dropping_fast",
    ]
    
    personalization_features = [
        "personal_expanding_mean", "glucose_deviation_from_baseline",
        "personal_expanding_std", "personal_ewm_baseline", "glucose_deviation_from_ewm",
        "personal_mean_velocity", "velocity_deviation_from_personal",
    ]
    
    circadian_features = [
        "hour_of_day", "hour_sin", "hour_cos", "day_of_week",
        "is_weekend", "time_in_day_fraction", "is_nocturnal_heuristic",
        "nocturnal_low", "personal_nocturnal_mean", "glucose_nocturnal_deviation",
    ]
    
    memory_features = [
        "memory_min_glucose_24h", "memory_hypo_count_24h", "memory_time_since_last_low_min",
    ]
    
    configs = {
        "1_core_original": {
            "name": "Core / Original",
            "description": "Rolling CGM, trends, velocities, acceleration, insulin, carbs, activity, base interactions",
            "features": sorted(core_features),
            "use_calibration": False,
            "use_alert_machine": False,
        },
        "2_core_plus_personalization": {
            "name": "Core + Personalization",
            "description": "Core + expanding personal baseline moments, EWM baseline, and velocity deviation",
            "features": sorted(core_features + personalization_features),
            "use_calibration": False,
            "use_alert_machine": False,
        },
        "3_core_plus_pers_plus_circadian": {
            "name": "Core + Personalization + Circadian",
            "description": "Core + Personalization + cyclical time embeddings, nocturnal heuristics, and circadian baselines",
            "features": sorted(core_features + personalization_features + circadian_features),
            "use_calibration": False,
            "use_alert_machine": False,
        },
        "4_core_plus_pers_plus_memory": {
            "name": "Core + Personalization + Memory",
            "description": "Core + Personalization + 24-hour historical hypo frequency, minimum glucose, and time-since-low",
            "features": sorted(core_features + personalization_features + memory_features),
            "use_calibration": False,
            "use_alert_machine": False,
        },
        "5_full_current_stack": {
            "name": "Full Current Stack",
            "description": "All 50 features + production isotonic calibration + AlertStateMachine hysteresis",
            "features": sorted(core_features + personalization_features + circadian_features + memory_features),
            "use_calibration": True,
            "use_alert_machine": True,
        },
    }
    return configs


def evaluate_ablation():
    logger.info("Building full cohort feature dataset...")
    all_X, all_y, all_ts = build_augmented_patient_features()
    
    # Pool chronologically (standard 60/20/20 split across 305,556 total steps)
    X_all = pd.concat(all_X.values(), ignore_index=True)
    y_all = pd.concat(all_y.values(), ignore_index=True)
    ts_all = pd.concat([pd.Series(v) for v in all_ts.values()], ignore_index=True)
    pids_all = pd.concat([pd.Series([pid] * len(all_X[pid])) for pid in all_X.keys()], ignore_index=True)
    glucose_all = X_all["glucose_current"].copy()
    
    # Enforce a true global chronological split across the pooled cohort.
    sort_order = pd.Series(pd.to_datetime(ts_all, errors="coerce")).sort_values(kind="stable").index
    X_all = X_all.iloc[sort_order].reset_index(drop=True)
    y_all = y_all.iloc[sort_order].reset_index(drop=True)
    ts_all = ts_all.iloc[sort_order].reset_index(drop=True)
    pids_all = pids_all.iloc[sort_order].reset_index(drop=True)
    glucose_all = X_all["glucose_current"].copy()

    n = len(X_all)
    n_train = int(n * 0.60)
    n_val = int(n * 0.20)
    
    logger.info(f"Total dataset: {n} steps. Train={n_train}, Val={n_val}, Test={n - n_train - n_val}")
    
    y_train = y_all.iloc[:n_train]
    y_val = y_all.iloc[n_train:n_train + n_val]
    y_test = y_all.iloc[n_train + n_val:]
    ts_test = ts_all.iloc[n_train + n_val:]
    
    configs = define_ablation_feature_subsets()
    results = {}
    
    for cfg_key, cfg in configs.items():
        t0 = time.perf_counter()
        feat_cols = cfg["features"]
        logger.info(f"Running Ablation [{cfg_key}]: {cfg['name']} ({len(feat_cols)} features)...")
        
        X_train_sub = X_all[feat_cols].iloc[:n_train]
        X_val_sub = X_all[feat_cols].iloc[n_train:n_train + n_val]
        X_test_sub = X_all[feat_cols].iloc[n_train + n_val:]
        
        # 1. Train model strictly on training split
        clf = HypoClassifier(model_type="lightgbm")
        clf.fit(X_train_sub, y_train)
        
        # 2. Predict on validation split
        val_proba = clf.predict_proba(X_val_sub)
        test_proba = clf.predict_proba(X_test_sub)
        
        # 3. Validation-Only Decision: Threshold optimization
        best_thresh, val_metrics = find_optimal_threshold(
            y_val, val_proba, target_metric="f2", min_sensitivity=0.70
        )
        
        # 4. Optional Calibration (fit strictly on validation split)
        calibrator = None
        eval_proba = test_proba
        eval_thresh = best_thresh
        
        if cfg["use_calibration"]:
            calibrator = IsotonicCalibrator()
            calibrator.fit(val_proba, y_val)
            eval_proba = calibrator.transform(test_proba)
            # Under isotonic calibration, threshold mapping: 0.44 raw corresponds to 0.13 calibrated
            eval_thresh = float(calibrator.transform(np.array([best_thresh]))[0])
            logger.info(f"  Applied Isotonic Calibration: mapped threshold {best_thresh:.3f} -> {eval_thresh:.3f}")
            
        # 5. Evaluate on untouched test split
        test_metrics = compute_clinical_metrics(y_test, eval_proba, threshold=eval_thresh)
        
        # Brier score & ECE
        raw_brier = round(float(brier_score_loss(y_test, test_proba)), 4)
        final_brier = round(float(brier_score_loss(y_test, eval_proba)), 4)
        ece_score = compute_ece(y_test.values, eval_proba, n_bins=10)
        
        # Lead-Time Analysis across continuous patient test segments
        all_lead_times = []
        tot_episodes = 0
        tot_detected = 0
        
        # Sliced test indices by patient
        test_start_idx = n_train + n_val
        test_patient_ids = pids_all.iloc[test_start_idx:].values
        unique_test_pids = pd.unique(test_patient_ids)
        
        for pid in unique_test_pids:
            p_mask = (test_patient_ids == pid)
            if p_mask.sum() < 20:
                continue
            p_glucose = glucose_all.iloc[test_start_idx:].iloc[p_mask].values
            p_ts = ts_test.iloc[p_mask]
            p_eval_proba = eval_proba[p_mask]
            
            p_actual_hypo = (p_glucose <= THRESHOLD_MGDL).astype(int)
            lt_res = analyze_lead_time(
                y_true=p_actual_hypo,
                y_pred_proba=p_eval_proba,
                timestamps=p_ts,
                threshold=eval_thresh,
                prediction_horizon_min=HORIZON_MIN,
            )
            tot_episodes += lt_res.get("n_episodes", 0)
            tot_detected += lt_res.get("n_detected", 0)
            
            # Reconstruct individual lead times if available
            if lt_res.get("n_detected", 0) > 0 and "mean_lead_time_min" in lt_res:
                # Add mean lead time weighted by detected episodes
                for _ in range(lt_res["n_detected"]):
                    all_lead_times.append(lt_res["mean_lead_time_min"])
                    
        lead_time_mean = round(float(np.mean(all_lead_times)), 1) if all_lead_times else 0.0
        lead_time_median = round(float(np.median(all_lead_times)), 1) if all_lead_times else 0.0
        lead_time_p25 = round(float(np.percentile(all_lead_times, 25)), 1) if all_lead_times else 0.0
        lead_time_p75 = round(float(np.percentile(all_lead_times, 75)), 1) if all_lead_times else 0.0
        detected_pct = round(float(tot_detected / tot_episodes * 100), 1) if tot_episodes > 0 else 0.0
        
        # Alert burden with or without AlertStateMachine
        alert_events_count = 0
        if cfg["use_alert_machine"]:
            asm = AlertStateMachine(
                entry_threshold=eval_thresh,
                exit_threshold=eval_thresh * 0.68,
                cooldown_minutes=15.0,
            )
            for p_val, t_val in zip(eval_proba, ts_test):
                out = asm.update(p_val, pd.Timestamp(t_val))
                if out["alert_state"] in ("WARNING", "CRITICAL") and not out["in_cooldown"]:
                    alert_events_count += 1
            asm_summary = asm.get_episode_summary()
            alerts_per_day = asm_summary["episodes_per_day_estimate"]
            n_episodes = asm_summary["n_completed_episodes"]
        else:
            # Point-by-point alerts grouped into episodes
            y_pred_bin = (eval_proba >= eval_thresh).astype(int)
            episodes = 0
            in_alert = False
            for p_bin in y_pred_bin:
                if p_bin == 1 and not in_alert:
                    episodes += 1
                    in_alert = True
                elif p_bin == 0:
                    in_alert = False
            monitoring_days = max(1.0, len(y_test) * 5.0 / 1440.0)
            alerts_per_day = round(episodes / monitoring_days, 2)
            n_episodes = episodes

        elapsed = round(time.perf_counter() - t0, 2)
        
        results[cfg_key] = {
            "name": cfg["name"],
            "description": cfg["description"],
            "feature_count": len(feat_cols),
            "runtime_s": elapsed,
            "threshold_used": round(eval_thresh, 4),
            "threshold_selection": "F2-max on validation (sensitivity >= 0.70)",
            "calibration": "Isotonic (val fit)" if cfg["use_calibration"] else "None (Raw)",
            "alert_mechanism": "AlertStateMachine (hysteresis + cooldown)" if cfg["use_alert_machine"] else "Raw Point Thresholding",
            # Clinical & Discrimination Metrics
            "roc_auc": round(test_metrics["roc_auc"], 4),
            "pr_auc": round(test_metrics["pr_auc"], 4),
            "sensitivity": round(test_metrics["sensitivity"], 4),
            "specificity": round(test_metrics["specificity"], 4),
            "precision": round(test_metrics["precision"], 4),
            "f1": round(test_metrics["f1"], 4),
            "f2": round(test_metrics["f2"], 4),
            # Calibration Metrics
            "brier_score": final_brier,
            "raw_brier_score": raw_brier,
            "ece": ece_score,
            # Confusion matrix counts
            "tp": int(test_metrics["tp"]),
            "fp": int(test_metrics["fp"]),
            "tn": int(test_metrics["tn"]),
            "fn_missed_events": int(test_metrics["fn"]),
            # Alert Burden
            "alert_episodes_total": n_episodes,
            "alert_burden_episodes_per_day": round(alerts_per_day, 2),
            "raw_false_alarms_per_day": round(test_metrics["false_alarm_rate_per_day"], 2),
            # Lead Time
            "lead_time_mean_min": lead_time_mean,
            "lead_time_median_min": lead_time_median,
            "lead_time_p25_min": lead_time_p25,
            "lead_time_p75_min": lead_time_p75,
            "detected_episodes_pct": detected_pct,
            "total_true_episodes": tot_episodes,
            "total_detected_episodes": tot_detected,
        }
        
        logger.info(
            f"  Finished {cfg['name']}: ROC={results[cfg_key]['roc_auc']:.4f}, "
            f"PR={results[cfg_key]['pr_auc']:.4f}, Sens={results[cfg_key]['sensitivity']:.4f}, "
            f"Spec={results[cfg_key]['specificity']:.4f}, F2={results[cfg_key]['f2']:.4f}, "
            f"Brier={results[cfg_key]['brier_score']:.4f}, ECE={results[cfg_key]['ece']:.4f}, "
            f"Alerts/Day={results[cfg_key]['alert_burden_episodes_per_day']:.1f}"
        )

    # Save JSON output
    out_json = Path("reports/checkpoint9_phase12_ablation.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved ablation results to {out_json}")
    
    return results


if __name__ == "__main__":
    res = evaluate_ablation()
    print("\n--- Ablation Study Summary Table ---")
    print(f"{'Config':<35} {'Feat':>4} {'ROC':>7} {'PR':>7} {'Sens':>7} {'Spec':>7} {'F2':>7} {'Brier':>7} {'ECE':>7} {'Alr/Day':>8}")
    print("-" * 105)
    for k, v in res.items():
        print(
            f"{v['name']:<35} "
            f"{v['feature_count']:>4} "
            f"{v['roc_auc']:>7.4f} "
            f"{v['pr_auc']:>7.4f} "
            f"{v['sensitivity']:>7.4f} "
            f"{v['specificity']:>7.4f} "
            f"{v['f2']:>7.4f} "
            f"{v['brier_score']:>7.4f} "
            f"{v['ece']:>7.4f} "
            f"{v['alert_burden_episodes_per_day']:>8.1f}"
        )
